"""GET /system — honest runtime indicator for the shells, plus the activity log."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from june_brain.activity import ActivityLog
from june_brain.build_info import build_version
from june_brain.config import resolve_runtime_config
from june_brain.loop.trace import TraceStore
from june_brain.memory.embeddings import default_embed_model
from june_brain.ollama_manager import is_model_available, is_ollama_running

from ..schemas import (
    ActivityEntryView,
    ActivityResponse,
    CapabilityProfileView,
    EntitlementView,
    LedgerEntryView,
    LedgerPageResponse,
    LedgerSummary,
    LedgerVerifyResponse,
    SurfacingDecisionView,
    SurfacingDrainResponse,
    SurfacingFeedbackRequest,
    SurfacingFeedbackResponse,
    SurfacingPageResponse,
    SystemStatus,
    TraceEventView,
    TraceListResponse,
    TraceSummary,
    TurnTraceView,
)

router = APIRouter(tags=["system"])


def _ledger_summary() -> LedgerSummary | None:
    """Build the Trust Ledger summary for /system. Best-effort; never raises."""
    from datetime import UTC, datetime

    try:
        from june_brain.trust import get_reader, last_verification

        reader = get_reader()
        midnight = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        verified = last_verification()
        return LedgerSummary(
            count=reader.count(),
            last_entry_ts=reader.last_entry_ts(),
            egress_today=reader.egress_count_since(midnight.isoformat()),
            chain_verified=verified[1] if verified else None,
            chain_verified_at=verified[0] if verified else None,
        )
    except Exception:  # noqa: BLE001 - the summary must never break the status badge
        return None


def _entitlement_view() -> EntitlementView | None:
    """Build the EntitlementView for /system. Best-effort; never raises."""
    try:
        from june_brain.licensing import get_entitlement

        ent = get_entitlement()
        return EntitlementView(
            tier=ent.tier,
            status=ent.status,
            holder=ent.holder if ent.holder else None,
            valid_until=ent.valid_until,
            features=sorted(list(ent.features)),
        )
    except Exception:  # noqa: BLE001 - the view must never break the status badge
        return None


@router.get("/system/entitlement", response_model=EntitlementView)
def get_entitlement_view() -> EntitlementView:
    """Report the current license entitlement (free by default, no phone-home)."""
    view = _entitlement_view()
    return view if view is not None else EntitlementView(tier="free", status="free — no license installed")


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
            version=build_version(),
            ledger_summary=_ledger_summary(),
            entitlement=_entitlement_view(),
        )

    ollama_reachable = False
    ollama_has_model = False
    embedding_model = default_embed_model()
    embedding_available = False
    semantic_recall_status = "unknown"
    semantic_recall_detail = "Semantic recall readiness was not checked for this runtime."
    if runtime.preset_key == "gemma":
        ollama_reachable = is_ollama_running(runtime.base_url)
        if ollama_reachable:
            ollama_has_model = is_model_available(runtime.model, runtime.base_url)
            embedding_available = is_model_available(embedding_model, runtime.base_url)
            if embedding_available:
                semantic_recall_status = "ready"
                semantic_recall_detail = (
                    f"Embedding model {embedding_model!r} is available for semantic recall."
                )
            else:
                semantic_recall_status = "degraded"
                semantic_recall_detail = (
                    f"Embedding model {embedding_model!r} is not pulled; semantic recall "
                    "falls back to SQLite keyword search."
                )
        else:
            semantic_recall_status = "degraded"
            semantic_recall_detail = (
                "Ollama is not reachable; semantic recall falls back to SQLite keyword search."
            )

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
        embedding_model=embedding_model,
        embedding_available=embedding_available,
        semantic_recall_status=semantic_recall_status,
        semantic_recall_detail=semantic_recall_detail,
        api_key_present=bool(runtime.api_key) and runtime.api_key != "ollama",
        version=build_version(),
        ledger_summary=_ledger_summary(),
        entitlement=_entitlement_view(),
    )


@router.get("/system/capability", response_model=CapabilityProfileView)
def get_capability() -> CapabilityProfileView:
    """Return the cached local-model capability profile.

    Uses the cheap cached accessor only — never triggers the expensive probe
    from the request path. Returns optimistic defaults (all 'good', empty
    checked_at) when no probe has run yet.
    """
    from june_brain.capability.probe import get_capability_profile

    profile = get_capability_profile()
    return CapabilityProfileView(
        summarization=profile.summarization,
        structured_output=profile.structured_output,
        long_context=profile.long_context,
        relevance_scoring=profile.relevance_scoring,
        checked_at=profile.checked_at,
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
# Trust Ledger — tamper-evident provenance receipts (ADR 0022)
# ---------------------------------------------------------------------------


def _ledger_entry_to_view(entry) -> LedgerEntryView:
    return LedgerEntryView(
        seq=entry.seq,
        id=entry.id,
        ts=entry.ts,
        kind=entry.kind,
        actor=entry.actor,
        payload=entry.payload,
        prev_hash=entry.prev_hash,
        entry_hash=entry.entry_hash,
        sig=entry.sig,
    )


@router.get("/system/ledger", response_model=LedgerPageResponse)
def get_ledger(cursor: int | None = None, limit: int = 50) -> LedgerPageResponse:
    """Newest-first page of Trust Ledger receipts. ``cursor`` pages older entries."""
    from june_brain.trust import get_reader

    limit = max(1, min(int(limit), 200))
    entries = get_reader().page(cursor=cursor, limit=limit)
    views = [_ledger_entry_to_view(e) for e in entries]
    next_cursor = views[-1].seq if len(views) == limit and views else None
    return LedgerPageResponse(entries=views, count=len(views), next_cursor=next_cursor)


@router.post("/system/ledger/verify", response_model=LedgerVerifyResponse)
def verify_ledger() -> LedgerVerifyResponse:
    """Recompute the whole chain and report integrity (and signatures, if keyed)."""
    from june_brain.trust import get_reader

    result = get_reader().verify_chain()
    return LedgerVerifyResponse(
        ok=result.ok, first_broken_seq=result.first_broken_seq, signed=result.signed
    )


# ---------------------------------------------------------------------------
# Silence Model — the "why June stayed quiet" surface (ADR 0023)
# ---------------------------------------------------------------------------


def _surfacing_to_view(rec) -> SurfacingDecisionView:
    return SurfacingDecisionView(
        id=rec.id,
        candidate_id=rec.candidate_id,
        kind=rec.kind,
        action=rec.action,
        reason=rec.reason,
        features=rec.features,
        ts=rec.ts,
        outcome=rec.outcome,
        surfaced_at=rec.surfaced_at,
        verdict=rec.verdict,
    )


@router.get("/system/surfacing", response_model=SurfacingPageResponse)
def get_surfacing(cursor: str | None = None, limit: int = 50) -> SurfacingPageResponse:
    """Recent Silence Model decisions with their truthful reasons (newest-first)."""
    from june_brain.silence import get_store

    limit = max(1, min(int(limit), 200))
    records = get_store().page(cursor=cursor, limit=limit)
    views = [_surfacing_to_view(r) for r in records]
    next_cursor = views[-1].ts if len(views) == limit and views else None
    return SurfacingPageResponse(entries=views, count=len(views), next_cursor=next_cursor)


@router.post("/system/surfacing/drain", response_model=SurfacingDrainResponse)
def drain_surfacing() -> SurfacingDrainResponse:
    """Drain the held batch digest — the explicit user-pulled 'catch me up' action.

    Returns all batch decisions that were held since the last drain and marks
    them surfaced. A second call immediately after returns an empty list.
    This is the event-boundary drain (ADR 0023): it is wired to a user action,
    never to a timer or scheduler.
    """
    try:
        from june_brain.silence import get_store

        records = get_store().drain_digest()
        views = [_surfacing_to_view(r) for r in records]
        return SurfacingDrainResponse(drained=views, count=len(views))
    except Exception:  # noqa: BLE001
        return SurfacingDrainResponse(drained=[], count=0)


@router.post("/system/surfacing/{decision_id}/feedback", response_model=SurfacingFeedbackResponse)
def surfacing_feedback(
    decision_id: str, body: SurfacingFeedbackRequest
) -> SurfacingFeedbackResponse:
    """Record the user's judgment of a decision; it feeds v2 and the suppress signal."""
    from june_brain.silence import get_store

    updated = get_store().record_feedback(decision_id, body.verdict)
    if updated is None:
        raise HTTPException(status_code=404, detail="surfacing decision not found")
    return SurfacingFeedbackResponse(
        id=updated.id, verdict=body.verdict, outcome=updated.outcome
    )


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
