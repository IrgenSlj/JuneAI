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
    action_class: ActionClass, *, tainted: bool = False
) -> tuple[bool, str]:
    """Return (needs_approval, human-readable reason) for an action class."""
    if action_class == "execute":
        return True, "Runs code or a shell command"
    if action_class == "write_network":
        reason = "Sends data off your device"
        if tainted:
            reason += " using content from a tool result"
        return True, reason
    if action_class == "read_network" and tainted:
        return True, "Fetches a network resource chosen by a prior tool result"
    return False, ""


def is_waivable(action_class: ActionClass, *, tainted: bool = False) -> bool:
    """Whether a per-conversation 'always allow' may waive future approvals.

    Taint-flagged network actions — the exact exfiltration pattern — always ask;
    everything else can be waived for the conversation once approved.
    """
    if tainted and action_class in ("write_network", "read_network"):
        return False
    return True
