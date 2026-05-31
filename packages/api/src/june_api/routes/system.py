"""GET /system — honest runtime indicator for the shells, plus the activity log."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from june_brain.activity import ActivityLog
from june_brain.config import resolve_runtime_config
from june_brain.loop.trace import TraceStore
from june_brain.ollama_manager import is_model_available, is_ollama_running

from ..schemas import (
    ActivityEntryView,
    ActivityResponse,
    SystemStatus,
    TraceEventView,
    TraceListResponse,
    TraceSummary,
    TurnTraceView,
)

router = APIRouter(tags=["system"])


@router.get("/system", response_model=SystemStatus)
def get_system_status() -> SystemStatus:
    """Report which runtime is active and whether it's actually usable.

    Shells display a small runtime badge built from this payload, so it
    must be honest about misconfigurations (Ollama not running, Gemma
    tag not pulled, missing API key) instead of pretending everything
    is fine.
    """
    try:
        from june_brain.config_store import get_privacy_dial

        dial = get_privacy_dial().value
    except Exception:
        dial = "private_by_default"

    try:
        runtime = resolve_runtime_config()
    except ValueError:
        return SystemStatus(
            provider="gemini",
            label="Gemini (cloud)",
            model="",
            mode="api",
            privacy_label="api-assisted",
            privacy_dial=dial,
            base_url="",
            ollama_reachable=False,
            ollama_has_model=False,
            api_key_present=False,
        )

    ollama_reachable = False
    ollama_has_model = False
    if runtime.preset_key == "gemma":
        ollama_reachable = is_ollama_running(runtime.base_url)
        if ollama_reachable:
            ollama_has_model = is_model_available(runtime.model, runtime.base_url)

    return SystemStatus(
        provider=runtime.preset_key,
        label=runtime.label,
        model=runtime.model,
        mode=runtime.mode,
        privacy_label=runtime.privacy_label,
        privacy_dial=dial,
        base_url=runtime.base_url,
        ollama_reachable=ollama_reachable,
        ollama_has_model=ollama_has_model,
        api_key_present=bool(runtime.api_key) and runtime.api_key != "ollama",
    )


# ---------------------------------------------------------------------------
# Activity log (Batch 1 — trust primitive)
# ---------------------------------------------------------------------------


def _entry_to_view(entry):
    return ActivityEntryView(
        id=entry.id,
        timestamp=entry.timestamp,
        kind=entry.kind,
        label=entry.label,
        status=entry.status,
        latency_ms=entry.latency_ms,
        detail=entry.detail,
    )


@router.get("/system/activity", response_model=ActivityResponse)
def get_activity(limit: int = 100, kind: str | None = None) -> ActivityResponse:
    """Reverse-chronological activity log: every recorded API request and tool call."""
    entries = ActivityLog().list(kind=kind, limit=max(1, min(int(limit), 500)))
    views = [_entry_to_view(e) for e in entries]
    return ActivityResponse(entries=views, count=len(views))


@router.delete("/system/activity", response_model=ActivityResponse)
def clear_activity() -> ActivityResponse:
    """Clear the activity log. Returns an empty response shape."""
    ActivityLog().clear()
    return ActivityResponse(entries=[], count=0)


# ---------------------------------------------------------------------------
# Glass-box turn traces — reopen a past turn's full harness operation
# ---------------------------------------------------------------------------


@router.get("/system/traces", response_model=TraceListResponse)
def list_traces(limit: int = 50) -> TraceListResponse:
    """Recent persisted turn traces, newest first (summaries only, no bodies)."""
    recent = TraceStore().list_recent(limit=max(1, min(int(limit), 200)))
    traces = [
        TraceSummary(
            turn_id=str(r["turn_id"]),
            started_at=float(r["started_at"]),
            event_count=int(r["event_count"]),
        )
        for r in recent
    ]
    return TraceListResponse(traces=traces, count=len(traces))


@router.delete("/system/traces", response_model=TraceListResponse)
def clear_traces() -> TraceListResponse:
    """Delete all persisted turn traces. Returns the now-empty list."""
    TraceStore().clear()
    return TraceListResponse(traces=[], count=0)


@router.get("/system/traces/{turn_id}", response_model=TurnTraceView)
def get_trace(turn_id: str) -> TurnTraceView:
    """The full glass-box trace for one turn: prompt, iterations, tools, reasoning."""
    trace = TraceStore().read(turn_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return TurnTraceView(
        turn_id=trace.turn_id,
        user_id=trace.user_id,
        started_at=trace.started_at,
        events=[
            TraceEventView(
                seq=e.seq, ts=e.ts, kind=e.kind, summary=e.summary, detail=e.detail
            )
            for e in trace.events
        ],
    )
