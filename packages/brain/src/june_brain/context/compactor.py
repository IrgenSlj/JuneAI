"""Compactor — threshold-triggered compaction of session.messages into PinnedState.

Trigger: estimated tokens of session.messages >= context_window * threshold_ratio.

On trigger (and capability_ok):
  Summarize the oldest ~half of messages via local-fast provider,
  parse the result into PinnedState fields, merge (anchored), drop those turns.

Fallback (invariant 6 — same PR):
  If capability_ok() is False OR the provider call raises,
  drop oldest turns by salience_fn (ascending) or simple oldest-first,
  keeping PinnedState intact. Never raise; never escalate to cloud.
"""

from __future__ import annotations

from collections.abc import Callable

from june_brain.context.assembler import estimate_tokens
from june_brain.context.pinned_state import PinnedState
from june_brain.providers.base import GenerateRequest, Message
from june_brain.providers.registry import ProviderRegistry, get_registry

_SUMMARIZE_PROMPT = (
    "Summarize the following conversation turns for a personal assistant's memory. "
    "Output exactly these sections (omit a section if nothing applies):\n"
    "GOAL: <one-sentence user goal, if stated>\n"
    "CONSTRAINTS: <bullet list>\n"
    "CONFIRMED: <bullet list of confirmed facts>\n"
    "OPEN: <bullet list of open questions>\n"
    "Only include facts explicitly stated. Be concise."
)


def _parse_summary(text: str) -> dict[str, object]:
    """Parse the model's structured summary text into PinnedState merge kwargs.

    Tolerant: if parsing fails, puts the whole text into confirmed_facts.
    """
    goal: str | None = None
    constraints: list[str] = []
    confirmed: list[str] = []
    open_q: list[str] = []

    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("GOAL:"):
            val = line[5:].strip()
            if val:
                goal = val
            current = None
        elif upper.startswith("CONSTRAINTS:"):
            current = "constraints"
        elif upper.startswith("CONFIRMED:"):
            current = "confirmed"
        elif upper.startswith("OPEN:"):
            current = "open"
        elif line.startswith("-") and current:
            item = line.lstrip("- ").strip()
            if item:
                if current == "constraints":
                    constraints.append(item)
                elif current == "confirmed":
                    confirmed.append(item)
                elif current == "open":
                    open_q.append(item)
        elif current:
            item = line.strip()
            if item:
                if current == "constraints":
                    constraints.append(item)
                elif current == "confirmed":
                    confirmed.append(item)
                elif current == "open":
                    open_q.append(item)

    if not goal and not constraints and not confirmed and not open_q:
        confirmed = [text.strip()] if text.strip() else []

    result: dict[str, object] = {}
    if goal:
        result["user_goal"] = goal
    if constraints:
        result["constraints"] = constraints
    if confirmed:
        result["confirmed_facts"] = confirmed
    if open_q:
        result["open_questions"] = open_q
    return result


class Compactor:
    """Compacts session.messages once they approach the context window."""

    def __init__(
        self,
        registry: ProviderRegistry | None = None,
        role: str = "local-fast",
        context_window: int = 8192,
        threshold_ratio: float = 0.70,
        capability_ok: Callable[[], bool] | None = None,
        salience_fn: Callable[[Message], float] | None = None,
    ) -> None:
        self._registry = registry
        self._role = role
        self._context_window = context_window
        self._threshold_ratio = threshold_ratio
        self._capability_ok = capability_ok if capability_ok is not None else lambda: True
        self._salience_fn = salience_fn

    def _get_registry(self) -> ProviderRegistry:
        return self._registry if self._registry is not None else get_registry()

    async def compact(self, session: object) -> bool:
        """Compact session if above threshold. Returns True if anything was done."""
        messages: list[Message] = getattr(session, "messages", [])
        pinned: PinnedState = getattr(session, "pinned", PinnedState())

        total_tokens = sum(
            estimate_tokens(m.role + m.content) for m in messages
        )
        threshold = self._context_window * self._threshold_ratio

        if total_tokens < threshold:
            return False

        half = max(1, len(messages) // 2)
        oldest = messages[:half]
        rest = messages[half:]

        if self._capability_ok():
            try:
                conversation_text = "\n".join(
                    f"{m.role.upper()}: {m.content}" for m in oldest
                )
                prompt_messages = [
                    Message(role="system", content=_SUMMARIZE_PROMPT),
                    Message(role="user", content=conversation_text),
                ]
                provider = self._get_registry().get(self._role)
                result = await provider.generate(
                    GenerateRequest(messages=prompt_messages, max_tokens=400)
                )
                kwargs = _parse_summary(result.text)
                pinned.merge(**kwargs)  # type: ignore[arg-type]
                session.messages[:] = rest  # type: ignore[union-attr]
                return True
            except Exception:  # noqa: BLE001
                pass

        # Fallback: drop oldest turns by salience ascending (or oldest-first)
        if self._salience_fn is not None:
            messages_copy = list(messages)
            messages_copy.sort(key=self._salience_fn)
            keep_set: set[int] = set()
            remaining_tokens = 0
            for i in range(len(messages_copy) - 1, -1, -1):
                msg = messages_copy[i]
                t = estimate_tokens(msg.role + msg.content)
                if remaining_tokens + t <= threshold:
                    keep_set.add(id(msg))
                    remaining_tokens += t
            session.messages[:] = [m for m in messages if id(m) in keep_set]  # type: ignore[union-attr]
        else:
            kept = list(messages)
            total = sum(estimate_tokens(m.role + m.content) for m in kept)
            while kept and total >= threshold:
                dropped = kept.pop(0)
                total -= estimate_tokens(dropped.role + dropped.content)
            session.messages[:] = kept  # type: ignore[union-attr]

        return True
