"""Loop interface types for C.2.

These are the stable boundary types shared by both harness engines.
Do NOT import from handwritten.py or langgraph_loop.py here — this module
must remain free of engine-specific deps so callers can import it independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from june_brain.providers.base import Message


@dataclass
class SessionState:
    """Mutable conversation state passed into every run_turn call."""

    user_id: str
    messages: list[Message]
    skill: str = "default"


@dataclass
class ToolCall:
    """A single tool invocation extracted from a model response."""

    name: str
    args: dict


@dataclass
class TokenAccounting:
    """Accumulated token counts across all provider calls in one turn."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class TurnProvenance:
    """Minimal per-turn record of what happened.

    The full SSE TurnProvenance (C.6) adds cloud_payload_summary and will
    extend this shape; keep this minimal for now.
    """

    tiers_used: list[str]
    cloud_call: bool
    model_ids: list[str]
    memories_recalled: int = 0
    skills_called: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class TurnResult:
    """The complete output of one harness turn."""

    assistant_msg: Message
    tool_calls: list[ToolCall]
    provenance: TurnProvenance
    tokens: TokenAccounting
    compacted: bool


@runtime_checkable
class HarnessLoop(Protocol):
    """The one stable interface both engines must satisfy."""

    async def run_turn(self, session: SessionState, user_msg: Message) -> TurnResult: ...
