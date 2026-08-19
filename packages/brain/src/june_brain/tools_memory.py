"""June's model-callable memory surface — the four tools of ADR 0032.

Before this module June had none: recall ran automatically in
``ContextAssembler`` and Promises were managed by the user through ``/tasks``,
so the product's two headline capabilities were things June could not do on
purpose when asked. What it had instead was seven v1 domain writers
(journal, relationships, goals, open loops, preferences, calendar, favorites)
that made the smallest model in the stack pick a table before it could store a
sentence.

These four replace them. ``remember`` and ``forget`` are inversion 3 (the user
decides what is kept, and forgetting is first-class and reversible);
``list_promises`` and ``update_promise`` are inversion 2 (standing intentions,
not terminating TODOs).
"""

from __future__ import annotations

import logging
import re
from typing import Annotated, Any

from .tools_base import Inject, ToolOutcome, tool

logger = logging.getLogger(__name__)

type AgentState = dict[str, Any] | None
InjectedAgentState = Annotated[AgentState, Inject]

# How close two recall scores must be before `forget` treats them as
# indistinguishable and asks instead of choosing. Forgetting the wrong memory
# is the one place in the product where being confidently wrong destroys user
# data, so the tool defers (inversion 1) rather than trusting a ranking.
_AMBIGUITY_MARGIN = 0.15

# Statuses the model may write. `running` is deliberately absent: it means the
# runtime is executing the promise, and a tool that could set it would make the
# Promises view assert work that nobody started.
_MODEL_SETTABLE_STATUSES = ("completed", "cancelled", "paused")


def _user_id(state: AgentState) -> str:
    if state is None:
        raise ValueError("Tool execution requires injected agent state.")
    return str(state["user_id"])


# ---------------------------------------------------------------------------
# Memory — remember and forget
# ---------------------------------------------------------------------------


@tool
def remember(
    text: str,
    state: InjectedAgentState = None,
) -> str:
    """Store one durable fact about the user so it can be recalled in later conversations. Use when the user asks June to remember something, or states something lasting about themselves. Write the fact as a complete sentence in the third person, e.g. 'Her sister is called Mira.' Do not use for passing details of the current conversation."""
    from .memory import MemoryManager

    text = (text or "").strip()
    if not text:
        return "Nothing to remember — no text was given."

    try:
        result = MemoryManager(_user_id(state)).write(
            {"kind": "fact", "fields": {"text": text}},
            source="tool:remember",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("remember: write failed")
        return f"Could not store that memory: {exc}"

    if not result.get("written"):
        detail = result.get("error") or "the memory store rejected it"
        return f"Could not store that memory — {detail}."
    return ToolOutcome(f"Remembered: {text}", wrote_memory=True)


@tool
def forget(
    description: str,
    state: InjectedAgentState = None,
) -> str:
    """Forget a stored memory the user has asked June to drop. Describe the memory in the user's own words; June finds the matching one. If several memories match, June lists them and forgets nothing, so ask the user which one they meant. Forgetting is reversible and the memory can be restored."""
    from .memory import MemoryManager
    from .memory.recall import prefixed_ref

    description = (description or "").strip()
    if not description:
        return "Nothing to forget — no description was given."

    try:
        manager = MemoryManager(_user_id(state))
        hits = manager.recall(description, k=5)
    except Exception as exc:  # noqa: BLE001
        logger.exception("forget: recall failed")
        return f"Could not search memory: {exc}"

    if not hits:
        return f"No stored memory matches '{description}', so nothing was forgotten."

    if len(hits) > 1 and _too_close_to_call(hits[0], hits[1]):
        listed = "\n".join(f"- {h.get('text', '')}" for h in hits[:3])
        return (
            f"Several memories match '{description}', so nothing was forgotten. "
            f"Ask which one is meant:\n{listed}"
        )

    target = hits[0]
    ref = prefixed_ref(target)
    text = str(target.get("text", "")).strip()
    try:
        removed = manager.forget(ref)
    except Exception as exc:  # noqa: BLE001
        logger.exception("forget: delete failed for ref=%s", ref)
        return f"Could not forget that memory: {exc}"

    if not removed:
        return f"Found '{text}' but could not forget it — it may already be gone."
    return ToolOutcome(
        f"Forgotten: {text}. It can be restored if this was a mistake.",
        wrote_memory=True,
    )


def _too_close_to_call(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """True when the top two hits are too near in score to pick between.

    Recall mixes distance-based and keyword sources, so the scores are only
    loosely comparable and their direction is not uniform. That is exactly why
    a near-tie must not be resolved by ranking: a missing or non-numeric score
    on either side counts as ambiguous, which fails toward asking the user.
    """
    a, b = first.get("score"), second.get("score")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return True
    if first.get("source") != second.get("source"):
        return True
    spread = max(abs(float(a)), abs(float(b)))
    if spread == 0:
        return True
    return abs(float(a) - float(b)) / spread < _AMBIGUITY_MARGIN


# ---------------------------------------------------------------------------
# Promises — read and update standing intentions
# ---------------------------------------------------------------------------


@tool
def list_promises(
    state: InjectedAgentState = None,
) -> str:
    """List the user's open promises — the standing intentions June is carrying — with their status and, where one is blocked, what it is waiting on."""
    from .tasks.store import TasksStore

    try:
        promises = TasksStore(user_id=_user_id(state)).active()
    except Exception as exc:  # noqa: BLE001
        logger.exception("list_promises: read failed")
        return f"Could not read promises: {exc}"

    if not promises:
        return "No open promises."

    lines = ["Open promises:"]
    for p in promises:
        line = f"- [{p.id[:8]}] {p.goal} — {p.status.value}"
        if p.due_at:
            line += f", due {p.due_at[:10]}"
        if p.blocked_reason:
            line += f"\n    blocked: {p.blocked_reason}"
        if p.next_action:
            line += f"\n    next: {p.next_action}"
        lines.append(line)
    return "\n".join(lines)


@tool
def update_promise(
    promise: str,
    status: str = "",
    next_action: str = "",
    state: InjectedAgentState = None,
) -> str:
    """Update one of the user's promises. Name the promise the way the user refers to it, e.g. 'the passport renewal' — an id is not needed. status may be 'completed', 'cancelled' or 'paused'. next_action records what the promise is waiting on. If several promises match, June lists them and changes nothing, so ask the user which one they meant. Only report a promise completed when the user says it is done."""
    from .tasks.models import TaskStatus
    from .tasks.store import TasksStore

    promise = (promise or "").strip()
    status = (status or "").strip().lower()
    next_action = (next_action or "").strip()

    if not promise:
        return "No promise was named."
    if not status and not next_action:
        return "Nothing to update — give a status or a next action."
    if status and status not in _MODEL_SETTABLE_STATUSES:
        allowed = ", ".join(_MODEL_SETTABLE_STATUSES)
        return f"'{status}' is not a status June can set. Use one of: {allowed}."

    try:
        store = TasksStore(user_id=_user_id(state))
        target, candidates = _resolve_promise(store, promise)
        if target is None:
            if candidates:
                listed = "\n".join(f"- {p.goal}" for p in candidates[:4])
                return (
                    f"Several promises match '{promise}', so nothing was changed. "
                    f"Ask which one is meant:\n{listed}"
                )
            return f"No open promise matches '{promise}'."

        if status:
            updated = store.set_status(
                target.id,
                TaskStatus(status),
                next_action=next_action or None,
            )
            if updated is None:
                return f"Promise '{target.goal}' could not be updated."
            return f"Promise '{updated.goal}' is now {status}."

        updated = store.set_blocked(
            target.id,
            reason=target.blocked_reason or "Waiting on the user.",
            next_action=next_action,
            final_deliverable=target.final_deliverable,
        )
        if updated is None:
            return f"Promise '{target.goal}' could not be updated."
        return f"Promise '{updated.goal}' is waiting on: {next_action}"
    except Exception as exc:  # noqa: BLE001
        logger.exception("update_promise: update failed for %r", promise)
        return f"Could not update that promise: {exc}"


def _resolve_promise(store: Any, reference: str) -> tuple[Any, list[Any]]:
    """Find the promise a user means. Returns (match, candidates_when_ambiguous).

    Taking an id alone was measured at 0/12 on the local model (D.5d): the id
    only exists in a `list_promises` result, so `update_promise` could not be
    reached without chaining two calls, and a 2B model does not chain reliably.
    Requiring the model to produce a handle it has never seen is a design
    defect, not a model failing. So a promise may also be named the way the
    user names it, and June does the matching — the same shape as `forget`,
    including the refusal to break a tie by ranking.
    """
    exact = store.get(reference)
    if exact is not None:
        return exact, []

    active = store.active()
    prefix = [p for p in active if p.id.startswith(reference)]
    if len(prefix) == 1:
        return prefix[0], []
    if len(prefix) > 1:
        return None, prefix

    scored = [(p, _goal_overlap(reference, p.goal)) for p in active]
    hits = sorted(
        [(p, n) for p, n in scored if n > 0], key=lambda item: item[1], reverse=True
    )
    if not hits:
        return None, []
    if len(hits) > 1 and hits[0][1] == hits[1][1]:
        return None, [p for p, _ in hits]
    return hits[0][0], []


def _content_words(text: str) -> set[str]:
    """Lowercased content words, function words dropped.

    Reuses recall's stop-word list rather than starting a second one; the two
    are answering the same question about the same user's English.
    """
    from .memory.recall import _STOPWORDS

    return {
        w
        for w in re.findall(r"[a-z][a-z'-]{2,}", text.lower())
        if w.strip("'-") not in _STOPWORDS
    }


def _goal_overlap(reference: str, goal: str) -> int:
    """How many content words the reference and the promise's goal share.

    Prefix-matched at four characters so "renewal"/"renew" and
    "booking"/"book" count — a user says a promise is done in different words
    than the promise was written in, which is the whole case for matching text
    rather than demanding an id.
    """
    ref, target = _content_words(reference), _content_words(goal)
    if not ref or not target:
        return 0
    score = 0
    for a in ref:
        if a in target or any(
            (a.startswith(b[:4]) or b.startswith(a[:4])) and min(len(a), len(b)) >= 4
            for b in target
        ):
            score += 1
    return score


JUNE_MEMORY_TOOLS = [remember, forget, list_promises, update_promise]
