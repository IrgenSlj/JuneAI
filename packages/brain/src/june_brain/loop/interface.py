"""Loop interface types for C.2.

These are the stable boundary types shared by both harness engines.
Do NOT import from handwritten.py or langgraph_loop.py here — this module
must remain free of engine-specific deps so callers can import it independently.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from june_brain.context.pinned_state import PinnedState
from june_brain.providers.base import Message


@dataclass
class SessionState:
    """Mutable conversation state passed into every run_turn call."""

    user_id: str
    messages: list[Message]
    skill: str = "default"
    pinned: PinnedState = field(default_factory=PinnedState)


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
    """Per-turn record of what happened and why."""

    tiers_used: list[str]
    cloud_call: bool
    model_ids: list[str]
    memories_recalled: int = 0
    skills_called: list[str] = field(default_factory=list)
    rationale: str = ""
    latency_ms: int = 0
    cloud_payload_summary: str | None = None


@dataclass
class TurnResult:
    """The complete output of one harness turn."""

    assistant_msg: Message
    tool_calls: list[ToolCall]
    provenance: TurnProvenance
    tokens: TokenAccounting
    compacted: bool


@dataclass
class StreamEvent:
    """A single event emitted by stream_turn; maps onto SSE frames in the API layer.

    ``detail`` carries the full, expandable content for glass-box trace kinds
    (the rendered ``prompt``, per-``iteration`` internals, ``compaction``). For
    ``token`` it stays empty — tokens are the foreground answer, not trace.
    """

    type: Literal[
        "token",
        "tool_call",
        "tool_result",
        "recall",
        "provenance",
        "done",
        "reasoning",
        "prompt",
        "iteration",
        "compaction",
    ]
    content: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_result: str = ""
    recall_hits: list[dict] = field(default_factory=list)
    provenance: TurnProvenance | None = None
    detail: str = ""
    iteration: int = 0


@runtime_checkable
class HarnessLoop(Protocol):
    """The one stable interface both engines must satisfy."""

    async def run_turn(self, session: SessionState, user_msg: Message) -> TurnResult: ...

    def stream_turn(
        self, session: SessionState, user_msg: Message
    ) -> AsyncIterator[StreamEvent]: ...
