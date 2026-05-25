"""GET /system — honest runtime indicator for the shells, plus the activity log."""

from __future__ import annotations

from fastapi import APIRouter
from june_brain.activity import ActivityLog
from june_brain.config import resolve_runtime_config
from june_brain.ollama_manager import is_model_available, is_ollama_running

from ..schemas import ActivityEntryView, ActivityResponse, SystemStatus

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
        runtime = resolve_runtime_config()
    except ValueError:
        return SystemStatus(
            provider="gemini",
            label="Gemini (cloud)",
            model="",
            mode="api",
            privacy_label="api-assisted",
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
