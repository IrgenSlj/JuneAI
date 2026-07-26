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

# Kept local to the guard package to avoid a guard -> loop import cycle.
_DEFAULT_NETWORK_TOOLS = frozenset({"web_search", "fetch_url", "read_webpage"})

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


def classify_action(
    tool_name: str, *, network_tools: frozenset[str] | None = None
) -> ActionClass:
    """Classify a tool call into an action class by name convention.

    Skill tools whose semantics we cannot infer fall through to ``write_local``;
    their real permissions come from the skill manifest (S6.3).
    """
    name = (tool_name or "").strip().lower()
    nt = network_tools if network_tools is not None else _DEFAULT_NETWORK_TOOLS
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
    action_class: ActionClass, *, tainted: bool = False, injected: bool = False
) -> tuple[bool, str]:
    """Return (needs_approval, human-readable reason) for an action class.

    ``injected`` means a prior tool result carried an injection shape (see
    ``guard.injection``). It closes the gap taint alone leaves: taint catches a
    URL copied verbatim out of a page, but not one the page *described* in prose
    for the model to reconstruct. After an injection signal, a network read asks
    even when nothing was copied.
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
    return False, ""


def evaluate_call(
    name: str,
    args: dict[str, Any],
    prior_results: list[str],
    *,
    allow_list: frozenset[str] = frozenset(),
    network_tools: frozenset[str] | None = None,
    injected: bool | None = None,
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
    tainted = is_tainted(args, prior_results)
    if injected is None:
        injected = scan_all(prior_results).suspicious
    needs, reason = requires_approval(action_class, tainted=tainted, injected=injected)
    if not needs:
        return True, action_class, ""
    if name in allow_list and is_waivable(action_class, tainted=tainted, injected=injected):
        return True, action_class, ""
    return False, action_class, reason


def is_waivable(
    action_class: ActionClass, *, tainted: bool = False, injected: bool = False
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
    return True
