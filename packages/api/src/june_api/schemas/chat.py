"""Chat request and streaming event schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """A single user turn submitted to POST /chat."""

    user_id: str = Field(..., description="Stable identifier for the user. Used as the memory partition key.")
    message: str = Field(..., description="The user's message in plain text.")
    skill: str = Field(
        default="assistant",
        description="Optional skill hint to guide the agent. Defaults to general assistant behavior.",
    )


class RecallHit(BaseModel):
    """One memory June drew on while answering this turn.

    Carries the same ``ref`` shape used by ``/memory``, so a UI rendering
    these can deep-link to the source: ``semantic:<id>``, ``node:<id>``,
    ``goal:<title>``, etc.
    """

    ref: str = Field(default="", description="Stable identifier; resolves in /memory.")
    text: str = Field(default="", description="Human-readable snippet that was injected.")
    source: str = Field(
        default="",
        description="Which store the hit came from: 'vector', 'graph', or 'sqlite'.",
    )
    kind: str = Field(default="", description="Sub-type for filtering / iconography.")
    score: float | None = Field(
        default=None,
        description="Loose relevance score. Lower is more relevant for distance-based sources.",
    )
    feedback: str = Field(
        default="",
        description="Existing vote on this memory: 'up', 'down', or '' when none.",
    )


class ChatHistoryMessage(BaseModel):
    role: str = Field(default="", description="'user' or 'assistant'.")
    content: str = Field(default="", description="The message text.")
    timestamp: str = Field(default="", description="ISO-ish timestamp the message was stored.")


class ChatHistory(BaseModel):
    messages: list[ChatHistoryMessage] = Field(default_factory=list, description="Persisted chat messages, oldest first.")


class ChatEvent(BaseModel):
    """One item in the SSE stream returned by POST /chat.

    The wire format is one ``data:`` frame per event carrying a JSON
    object with this shape. ``type`` drives the UI.
    """

    type: Literal[
        "token",
        "tool_call",
        "tool_result",
        "recall",
        "provenance",
        "done",
        "error",
        "reasoning",
        "prompt",
        "iteration",
        "compaction",
        "tool_blocked",
    ] = Field(
        ...,
        description=(
            "Discriminator that determines the meaning of the payload. "
            "reasoning: June's chain-of-thought for this turn; shown selectively, not part of the answer. "
            "prompt/iteration/compaction: glass-box trace events; the collapsed line is in content, "
            "the full expandable body is in detail."
        ),
    )
    content: str = Field(
        default="",
        description="Textual content for token and error events; empty otherwise.",
    )
    tool_name: str = Field(
        default="",
        description="Name of the tool for tool_call and tool_result events.",
    )
    tool_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool for tool_call events.",
    )
    tool_result: str = Field(
        default="",
        description="Serialized result returned by the tool for tool_result events.",
    )
    recall_hits: list[RecallHit] = Field(
        default_factory=list,
        description=(
            "Memories June drew on this turn. Emitted once, before the first token, "
            "so the UI can render a 'memories used' affordance per assistant message."
        ),
    )
    provenance: dict[str, Any] | None = Field(
        default=None,
        description="Per-turn model provenance. Carries provider/model/tier/latency_ms when type == 'provenance'.",
    )
    detail: str = Field(
        default="",
        description=(
            "Full, expandable body for glass-box trace events (prompt/iteration/"
            "compaction/tool_call/tool_result/reasoning). Empty for token events — "
            "tokens are the foreground answer, never the trace."
        ),
    )
    network: bool = Field(
        default=False,
        description=(
            "True on tool_call events for tools that reach the network. The UI "
            "flags these as egress even when the LLM tier is local, so a query "
            "leaving the machine is never silent."
        ),
    )
    needs_approval: bool = Field(
        default=False,
        description=(
            "True on tool_blocked events the guard withheld pending the user's "
            "explicit approval (network egress, code execution, tainted reads). "
            "Distinct from a Local-only block (network=True), which is resolved "
            "by changing the privacy dial rather than approving a single action."
        ),
    )
    action_class: str = Field(
        default="",
        description=(
            "The guard's action class for an approval-needed tool_blocked event "
            "(e.g. 'write_network', 'execute', 'read_network'). Empty otherwise."
        ),
    )
