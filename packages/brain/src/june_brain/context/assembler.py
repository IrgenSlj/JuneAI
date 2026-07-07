"""ContextAssembler — fixed 5-part context order for every harness turn.

Assembly order (stable prefix first, volatile last):
  1. system / persona block        (stable, cached)
  2. character block               (semi-stable, optional — C.5)
  3. pinned-state block            (the anchored summary)
  4. recalled memory               (volatile, this-turn — C.4)
  5. recent raw turns              (volatile, oldest trimmed to fit budget)
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from june_brain.providers.base import Message

from .temporal import build_temporal_block
from .tokens import estimate_tokens

_DEFAULT_SYSTEM_PROMPT = (
    "You are June, a personal assistant. "
    "Be helpful, honest, conversational, and concise. "
    "You remember what matters and tell the truth plainly and kindly. "
    "Take gentle initiative when it benefits the user, and avoid checklist-style "
    "replies unless structure truly helps."
)

# Asks the model to externalize its reasoning before answering. Works for any
# model (Gemma, Gemini) — it is the model's own natural reasoning, surfaced via
# <think> tags that the loop's ReasoningSplitter routes to the (hidable)
# reasoning channel rather than the answer. Kept brief so it doesn't dominate
# latency on simple turns.
_REASONING_INSTRUCTION = (
    "Think through the request first inside <think> and </think> tags — your own "
    "reasoning, kept brief and focused. Then, after the closing </think> tag, write "
    "your reply to the user. Do not mention the tags or that you are thinking."
)


def _msg_tokens(msg: Message) -> int:
    return estimate_tokens(msg.role + msg.content)


class ContextAssembler:
    """Composes the fixed 5-part context list for a harness turn."""

    def __init__(
        self,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        character_block: str | None = None,
        recall: Callable[..., list[Message]] | None = None,
        token_budget: int = 6000,
        tools_block: str | None = None,
        reason: bool = False,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._character_block = character_block
        self._recall = recall
        self._token_budget = token_budget
        self._tools_block = tools_block
        self._reason = reason
        # Injected read-time clock returning *local* time. When None, no temporal
        # block is added (kept out of tests that assert an exact context shape);
        # the live loop passes datetime.now. A clock that raises degrades to no
        # block rather than failing the turn (graceful-degradation invariant).
        self._clock = clock

    def set_reason(self, reason: bool) -> None:
        """Per-turn override of the reasoning instruction (difficulty-gated)."""
        self._reason = reason

    def assemble(self, session: object, user_msg: Message) -> list[Message]:
        """Return the ordered context list, trimming oldest raw turns to fit budget."""
        fixed: list[Message] = []

        # Section 1 — system / persona (+ the standing untrusted-content rule,
        # which is part of the behavioral safety floor and always present, and
        # the reasoning instruction when enabled).
        from june_brain.guard import UNTRUSTED_CONTENT_RULE

        system_content = f"{self._system_prompt}\n\n{UNTRUSTED_CONTENT_RULE}"
        if self._reason:
            system_content = f"{system_content}\n\n{_REASONING_INSTRUCTION}"
        fixed.append(Message(role="system", content=system_content))

        # Section 2 — character block (optional)
        if self._character_block:
            fixed.append(Message(role="system", content=self._character_block))

        # Section 2.5 — tool advertisement (optional). Tells the model which
        # tools it may call and the JSON to emit; without it the loop can
        # dispatch tools but the model never knows they exist.
        if self._tools_block:
            fixed.append(Message(role="system", content=self._tools_block))

        # Section 3 — pinned state (only when non-empty)
        pinned = getattr(session, "pinned", None)
        if pinned is not None:
            block_text = pinned.to_block()
            if block_text:
                fixed.append(Message(role="system", content=block_text))

        # Section 3.5 — temporal context (optional, read-time; D.1). Placed after
        # the stable prefix so it never busts the cache of the system/character/
        # tools blocks, only when a clock is injected. Best-effort: a clock that
        # raises degrades to no block rather than failing the turn.
        if self._clock is not None:
            try:
                fixed.append(
                    Message(role="system", content=build_temporal_block(self._clock()))
                )
            except Exception:
                pass

        # Section 4 — recalled memory (optional)
        if self._recall is not None:
            recalled = self._recall(session, user_msg)
            fixed.extend(recalled)

        # Budget consumed by fixed sections
        fixed_tokens = sum(_msg_tokens(m) for m in fixed)
        user_msg_tokens = _msg_tokens(user_msg)
        remaining = self._token_budget - fixed_tokens - user_msg_tokens

        # Section 5 — recent raw turns, trimmed oldest-first to fit budget
        raw: list[Message] = list(getattr(session, "messages", []))
        while raw and remaining < sum(_msg_tokens(m) for m in raw):
            raw.pop(0)

        return [*fixed, *raw, user_msg]
