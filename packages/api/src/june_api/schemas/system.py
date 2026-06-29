"""System-status and activity-log schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LedgerSummary(BaseModel):
    """Trust Ledger summary embedded in GET /system (ADR 0022)."""

    count: int = Field(default=0, description="Total entries in the chain.")
    last_entry_ts: str | None = Field(
        default=None, description="ISO timestamp of the most recent entry, if any."
    )
    egress_today: int = Field(
        default=0, description="Number of cloud-egress entries recorded since midnight UTC."
    )
    chain_verified: bool | None = Field(
        default=None,
        description="Result of the last verify_chain this session; null if not verified yet.",
    )
    chain_verified_at: str | None = Field(
        default=None,
        description="ISO timestamp of the last verification this session; null if never verified.",
    )


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
    embedding_model: str = Field(
        default="",
        description="Configured local embedding model used for semantic recall.",
    )
    embedding_available: bool = Field(
        default=False,
        description="True when the configured local embedding model is available.",
    )
    semantic_recall_status: str = Field(
        default="unknown",
        description="'ready' when vector recall can embed queries, 'degraded' when it falls back to keyword search, or 'unknown' when not checked.",
    )
    semantic_recall_detail: str = Field(
        default="",
        description="Human-readable explanation of semantic recall readiness.",
    )
    api_key_present: bool = Field(
        default=False,
        description="True when the active runtime has the credentials it needs. Only meaningful for the gemini preset.",
    )
    version: str = Field(
        default="unknown",
        description="Short git SHA (or JUNE_BUILD_SHA) identifying the running build.",
    )
    ledger_summary: LedgerSummary | None = Field(
        default=None,
        description="Trust Ledger summary (ADR 0022): entry count, last entry, egress today, last verification.",
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
# Trust Ledger receipts (GET /system/ledger, POST /system/ledger/verify)
# ---------------------------------------------------------------------------


class LedgerEntryView(BaseModel):
    """One tamper-evident ledger receipt (ADR 0022)."""

    seq: int = Field(..., description="Gap-free monotonic position in the chain.")
    id: str = Field(..., description="Entry uuid.")
    ts: str = Field(..., description="ISO-8601 UTC timestamp.")
    kind: str = Field(..., description="'egress', 'action', 'approval', or 'system'.")
    actor: str = Field(..., description="'june' or 'user'.")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Redacted summary of the event (never raw content)."
    )
    prev_hash: str = Field(..., description="entry_hash of the previous entry; 64 zeros at genesis.")
    entry_hash: str = Field(..., description="blake2b digest committing to this entry.")
    sig: str | None = Field(default=None, description="Hex Ed25519 signature, or null if unsigned.")


class LedgerPageResponse(BaseModel):
    """GET /system/ledger payload — newest-first page of receipts."""

    entries: list[LedgerEntryView] = Field(default_factory=list)
    count: int = Field(default=0, description="Number of entries in this page.")
    next_cursor: int | None = Field(
        default=None,
        description="Pass as ?cursor= to fetch the next (older) page; null when no more.",
    )


class LedgerVerifyResponse(BaseModel):
    """POST /system/ledger/verify payload — the chain integrity check result."""

    ok: bool = Field(..., description="True when the whole chain verifies.")
    first_broken_seq: int | None = Field(
        default=None, description="First seq where the chain breaks, or null when ok."
    )
    signed: bool = Field(
        default=False,
        description="True when the chain carries signatures that were verified with a device key.",
    )


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


class CapabilityProfileView(BaseModel):
    """GET /system/capability — cached local-model capability verdicts."""

    summarization: str = Field(
        ..., description="'good', 'weak', or 'poor' — summarization quality verdict."
    )
    structured_output: str = Field(
        ..., description="'good', 'weak', or 'poor' — structured-output (JSON) verdict."
    )
    long_context: str = Field(
        ..., description="'good', 'weak', or 'poor' — long-context recall verdict."
    )
    relevance_scoring: str = Field(
        ..., description="'good', 'weak', or 'poor' — relevance-scoring verdict."
    )
    checked_at: str = Field(
        default="",
        description="ISO timestamp of the last probe run; empty string if never probed.",
    )
