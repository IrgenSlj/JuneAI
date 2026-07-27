"""Action classification and the approval policy (ADR 0021, S6.2).

Framing (S6.1) tells the model that tool results are untrusted; the action gates
limit the blast radius when a model is fooled anyway. Every tool call is
classified into one of five action classes, and a policy decides which need
explicit user approval ("defers" — inversion 1, implemented as control flow).

A deliberate refinement of the rebuild plan's "gate all write_*": June's local
writes are the user's *own* memory on their *own* machine, written because they
asked — gating every `save_goal` would be naggy and is not the threat. The real
damage/exfiltration vectors are **network egress** (`write_network`),
**code execution** (`execute`), and **tainted network reads** — a `fetch_url`
whose target was supplied by a prior (untrusted) tool result, i.e. "follow the
link the injected web page told you to." Those are what the gate stops; local
reads/writes flow freely (still surfaced by the privacy dial + provenance).

Pure and dependency-free (no loop import — the loop imports the guard, not the
reverse), so the taxonomy is unit-tested without the rest of the stack.
"""

from __future__ import annotations

from typing import Any, Literal

from .injection import scan_all

ActionClass = Literal[
    "read_local", "read_network", "write_local", "write_network", "execute"
]

# The one definition of "invoking this reaches the network". It lives in the
# guard because the guard is the lowest layer that needs it (the loop imports
# the guard, never the reverse), and because the alternative — a second copy in
# `loop/wiring.py` — meant adding a networked tool to one list and not the other
# left it either ungated or unsurfaced, silently and in opposite directions.
NETWORK_TOOLS = frozenset({"web_search", "fetch_url", "read_webpage"})

_EXECUTE_PREFIXES = ("run_", "exec_", "execute_", "shell_", "eval_")
_NETWORK_WRITE_PREFIXES = ("send_", "post_", "publish_", "email_", "notify_", "sms_", "tweet_")
_READ_PREFIXES = ("get_", "list_", "search_", "read_", "find_", "show_")
_WRITE_PREFIXES = (
    "save_", "log_", "create_", "update_", "delete_", "set_", "clear_",
    "add_", "remove_", "mark_",
)

# Ignore taint matches on short argument values — they collide with ordinary
# words and would flag everything.
_MIN_TAINT_LEN = 8

# Least to most consequential. Used to ask "is this class covered by what the
# skill declared?", never to sum or average — the classes are a lattice of
# capability, not a score.
RISK_ORDER: tuple[ActionClass, ...] = (
    "read_local",
    "write_local",
    "read_network",
    "write_network",
    "execute",
)

_NETWORK_SCOPES = frozenset({"read_network", "write_network"})


def classify_action(
    tool_name: str, *, network_tools: frozenset[str] | None = None
) -> ActionClass:
    """Classify a tool call into an action class by name convention.

    The name is the only input, deliberately: this value is what the UI shows
    and what the ledger records, so it has to describe what the call *is*, not
    how cautious June has decided to be about it. Reading a local PDF is a
    ``read_local`` even when the skill offering it can also reach the network —
    labelling it otherwise would put a false statement in the audit trail.

    Caution about who is asking lives in :func:`requires_approval`, which takes
    the skill's capabilities as a separate argument. Skill tools whose semantics
    cannot be inferred fall through to ``write_local``.
    """
    name = (tool_name or "").strip().lower()
    nt = network_tools if network_tools is not None else NETWORK_TOOLS
    if any(name.startswith(p) for p in _EXECUTE_PREFIXES):
        return "execute"
    if any(name.startswith(p) for p in _NETWORK_WRITE_PREFIXES):
        return "write_network"
    if name in nt:
        return "read_network"
    if any(name.startswith(p) for p in _READ_PREFIXES):
        return "read_local"
    if any(name.startswith(p) for p in _WRITE_PREFIXES):
        return "write_local"
    return "write_local"


def is_network_capable(declared_scopes: tuple[str, ...] | frozenset[str]) -> bool:
    """Whether a skill's contract lets it reach the network at all."""
    return bool(_NETWORK_SCOPES & frozenset(declared_scopes))


def _flatten_str_values(value: Any) -> list[str]:
    """Collect string leaves from an argument value (dict/list/scalar)."""
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(_flatten_str_values(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            out.extend(_flatten_str_values(v))
    return out


def is_tainted(args: dict[str, Any], prior_results: list[str]) -> bool:
    """True when an argument value was derived from a prior tool result.

    This is the exfiltration tell: the model is feeding untrusted tool output
    (a URL, a token, a payload it just read from a web page) back into a new
    action. Short values are ignored to avoid false positives on common words.
    """
    if not prior_results:
        return False
    blob = "\n".join(r for r in prior_results if r)
    if not blob:
        return False
    for value in _flatten_str_values(args):
        candidate = value.strip()
        if len(candidate) >= _MIN_TAINT_LEN and candidate in blob:
            return True
    return False


def requires_approval(
    action_class: ActionClass,
    *,
    tainted: bool = False,
    injected: bool = False,
    network_capable: bool = False,
) -> tuple[bool, str]:
    """Return (needs_approval, human-readable reason) for an action class.

    ``injected`` means a prior tool result carried an injection shape (see
    ``guard.injection``). It closes the gap taint alone leaves: taint catches a
    URL copied verbatim out of a page, but not one the page *described* in prose
    for the model to reconstruct. After an injection signal, a network read asks
    even when nothing was copied.

    ``network_capable`` says the caller is a skill whose contract permits
    network access. A name is not proof of what a call does, so a "read" from
    such a skill is treated as a possible network read — a tool named
    ``get_page_content`` that fetches URLs otherwise classifies ``read_local``
    and escapes taint gating entirely. This only bites when the arguments are
    tainted or a prior result looked hostile, so it costs no extra prompts in
    ordinary use, and the *class* stays honest for the UI and the ledger.
    """
    if action_class == "execute":
        return True, "Runs code or a shell command"
    if action_class == "write_network":
        reason = "Sends data off your device"
        if tainted:
            reason += " using content from a tool result"
        elif injected:
            reason += ", after a tool result that looked like an injection attempt"
        return True, reason
    if action_class == "read_network":
        if tainted:
            return True, "Fetches a network resource chosen by a prior tool result"
        if injected:
            return True, "Fetches a network resource after a suspicious tool result"
    if action_class == "read_local" and network_capable:
        if tainted:
            return True, (
                "Reads through a skill that can reach the network, "
                "using content from a tool result"
            )
        if injected:
            return True, (
                "Reads through a skill that can reach the network, "
                "after a suspicious tool result"
            )
    return False, ""


def exceeds_declared_scopes(
    action_class: ActionClass, declared_scopes: tuple[str, ...] | frozenset[str]
) -> str:
    """Return a reason when a call exceeds the skill's contract, else "".

    A skill declares in the manifest which action classes it may use. Until now
    that contract was *reported* — the UI showed drift — and the call went
    through anyway. Reporting a violation after allowing it is not a permission
    system, so the contract is enforced here.

    The attack this stops is the update: a skill the user installed and granted
    ``read_local`` ships a new version that advertises ``send_report``, and
    inherits the trust the user extended to the old one. It now blocks until the
    user widens the contract deliberately.

    A skill with no declared scopes has no contract to violate, and nothing here
    applies — that stays true so existing installs do not break on upgrade.
    """
    declared = frozenset(declared_scopes)
    if not declared or action_class in declared:
        return ""
    return (
        f"This skill did not declare the '{action_class}' capability. "
        f"It declared: {', '.join(sorted(declared)) or 'nothing'}."
    )


def evaluate_call(
    name: str,
    args: dict[str, Any],
    prior_results: list[str],
    *,
    allow_list: frozenset[str] = frozenset(),
    network_tools: frozenset[str] | None = None,
    injected: bool | None = None,
    declared_scopes: tuple[str, ...] | frozenset[str] = (),
) -> tuple[bool, str, str]:
    """Decide whether a tool call may execute. Returns (allowed, action_class, reason).

    ``allow_list`` is the per-conversation set of tool names the user already
    approved; it can waive future approvals except where the exfiltration
    pattern is present, which always asks. ``reason`` is non-empty only when the
    call is blocked.

    ``injected`` is scanned from ``prior_results`` when not supplied. Callers
    dispatching a batch of calls should scan once and pass the result, since the
    evidence is the same for every call in the batch.
    """
    action_class = classify_action(name, network_tools=network_tools)

    # Checked before approval, and not waivable by it: an "always allow" is the
    # user permitting a tool they were shown, not permitting a skill to step
    # outside the contract they agreed to when they installed it.
    breach = exceeds_declared_scopes(action_class, declared_scopes)
    if breach:
        return False, action_class, breach

    network_capable = is_network_capable(declared_scopes)
    tainted = is_tainted(args, prior_results)
    if injected is None:
        injected = scan_all(prior_results).suspicious
    needs, reason = requires_approval(
        action_class, tainted=tainted, injected=injected, network_capable=network_capable
    )
    if not needs:
        return True, action_class, ""
    if name in allow_list and is_waivable(
        action_class, tainted=tainted, injected=injected, network_capable=network_capable
    ):
        return True, action_class, ""
    return False, action_class, reason


def is_waivable(
    action_class: ActionClass,
    *,
    tainted: bool = False,
    injected: bool = False,
    network_capable: bool = False,
) -> bool:
    """Whether a per-conversation 'always allow' may waive future approvals.

    Two conditions revoke a standing approval, and they are the whole point of
    having one: taint (an argument copied out of a tool result) and injection (a
    tool result that read like an attempt to steer June). "Always allow" is a
    statement about a tool, made before the untrusted content arrived; it cannot
    also be consent for what that content later asks the tool to do.

    Local reads and writes stay waivable either way — gating June's own memory
    on her own machine is naggy and is not the threat. Content that arrives
    poisoned is Phase 6's quarantine problem, not this gate's.
    """
    consequential = ("write_network", "read_network", "execute")
    if (tainted or injected) and action_class in consequential:
        return False
    # A network-capable skill's "read" is treated the same way, for the same
    # reason it is gated at all: the name is not evidence about the call.
    if (tainted or injected) and network_capable and action_class == "read_local":
        return False
    return True
