"""Loop interface types for the harness loop.

These are the stable boundary types the loop is built on. Do NOT import from
handwritten.py here — this module must remain free of engine-specific deps so
callers can import it independently.
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
    # Tool names the user approved for this conversation (the guard's
    # per-conversation allow-list, ADR 0021 S6.2). Empty by default — gated
    # actions ask until approved.
    approved_tools: set[str] = field(default_factory=set)


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
    # Memories June *wrote* this turn, reported by the tools that wrote them
    # rather than inferred from which tools ran. Recall has always been shown;
    # writes were not, so June could add to or delete from the user's memory
    # without the turn saying so.
    memories_written: int = 0
    skills_called: list[str] = field(default_factory=list)
    rationale: str = ""
    latency_ms: int = 0
    cloud_payload_summary: str | None = None
    # Networked tools dispatched this turn (e.g. web_search). The LLM stayed
    # local, but these tools sent data off the machine — surfaced separately
    # from cloud_call so local-only egress is never silent.
    egress: list[str] = field(default_factory=list)
    # Difficulty classification that drove routing + reasoning gating (S4):
    # the label (trivial/standard/hard/creative) and whether it came from the
    # model classifier, its cache, or the heuristic fallback.
    difficulty: str = ""
    difficulty_source: str = ""
    # Per-turn token counts surfaced in the Glass Box provenance frame.
    input_tokens: int = 0
    output_tokens: int = 0
    compacted: bool = False


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
        "tool_blocked",
        "model_call",
    ]
    content: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_result: str = ""
    recall_hits: list[dict] = field(default_factory=list)
    provenance: TurnProvenance | None = None
    detail: str = ""
    iteration: int = 0
    # True on tool_call events for tools that reach the network — lets the UI
    # flag the call as egress even when the LLM tier is local.
    network: bool = False
    # True on tool_blocked events the guard withheld pending the user's explicit
    # approval (ADR 0021, S6.2 — network egress, code execution, tainted reads).
    # Distinct from the Local-only block (network=True): the latter asks the user
    # to change the privacy dial; this asks them to approve one consequential act.
    needs_approval: bool = False
    # The guard's action class for a blocked call (e.g. "write_network",
    # "execute", "read_network"). Empty unless needs_approval is set.
    action_class: str = ""
    # Stable id of the turn's persisted trace; lets callers match a live stream
    # event to the later-written trace file at <datadir>/traces/<turn_id>.json.
    turn_id: str | None = None


@runtime_checkable
class HarnessLoop(Protocol):
    """The interface a harness loop satisfies.

    ``run_turn`` is a non-streaming convenience wrapper around ``stream_turn``,
    not a second entry point into a second engine (ADR 0018, D.4c). It used to
    be the latter, and the two had drifted.
    """

    async def run_turn(self, session: SessionState, user_msg: Message) -> TurnResult: ...

    def stream_turn(
        self, session: SessionState, user_msg: Message
    ) -> AsyncIterator[StreamEvent]: ...
