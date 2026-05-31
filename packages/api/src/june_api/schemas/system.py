"""System-status and activity-log schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SystemStatus(BaseModel):
    """What the shells need to display a honest runtime indicator."""

    provider: str = Field(..., description="Active preset key: 'gemma' or 'gemini'.")
    label: str = Field(..., description="Human-readable runtime label.")
    model: str = Field(..., description="Active model identifier.")
    mode: str = Field(..., description="'local' when inference stays on-device, 'api' otherwise.")
    privacy_label: str = Field(..., description="'local-only' or 'api-assisted'.")
    privacy_dial: str = Field(
        default="private_by_default",
        description="The user's persistent privacy dial: 'local_only', 'private_by_default', or 'cloud_first'. This is the mode that gates networked tools.",
    )
    base_url: str = Field(default="", description="Endpoint the brain is talking to.")
    ollama_reachable: bool = Field(
        default=False,
        description="True when the configured Ollama instance responds. Only meaningful for the gemma preset.",
    )
    ollama_has_model: bool = Field(
        default=False,
        description="True when the active Gemma tag is already pulled. Only meaningful for the gemma preset.",
    )
    api_key_present: bool = Field(
        default=False,
        description="True when the active runtime has the credentials it needs. Only meaningful for the gemini preset.",
    )


class ActivityEntryView(BaseModel):
    """One row in the activity log, returned by GET /system/activity."""

    id: int
    timestamp: str
    kind: str = Field(..., description="'request', 'tool', 'task', or 'skill'.")
    label: str = Field(..., description="Short summary, e.g. 'GET /skills'.")
    status: int | None = Field(default=None, description="HTTP status for requests.")
    latency_ms: int | None = None
    detail: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured context (tool args, error message, etc.).",
    )


class ActivityResponse(BaseModel):
    """GET /system/activity payload."""

    entries: list[ActivityEntryView] = Field(default_factory=list)
    count: int = 0


# ---------------------------------------------------------------------------
# Glass-box turn traces (GET /system/traces, /system/traces/{turn_id})
# ---------------------------------------------------------------------------


class TraceEventView(BaseModel):
    """One recorded step in a persisted turn trace."""

    seq: int = Field(..., description="Order within the turn, starting at 0.")
    ts: float = Field(..., description="Epoch seconds when the step was recorded.")
    kind: str = Field(
        ...,
        description="Step type: prompt, iteration, recall, tool_call, tool_result, reasoning, compaction, provenance, done, error.",
    )
    summary: str = Field(default="", description="Collapsed one-line label.")
    detail: str = Field(default="", description="Full expandable body for this step.")


class TraceSummary(BaseModel):
    """Lightweight entry in the trace list — no event bodies."""

    turn_id: str
    started_at: float
    event_count: int


class TraceListResponse(BaseModel):
    """GET /system/traces payload — recent turns, newest first."""

    traces: list[TraceSummary] = Field(default_factory=list)
    count: int = 0


class TurnTraceView(BaseModel):
    """GET /system/traces/{turn_id} payload — one turn's full trace."""

    turn_id: str
    user_id: str = ""
    started_at: float = 0.0
    events: list[TraceEventView] = Field(default_factory=list)
