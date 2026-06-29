"""GET /home/{user_id}/holdings — the control-room "what June is holding" surface.

Read-only aggregation over three existing sources (handoff S5): the user's open
promises, the Silence Model's held digest (ADR 0023), and the Trust Ledger
summary (ADR 0022). It introduces no new state. The ledger/silence parts are
device-global; only the promises are scoped to ``user_id``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter
from june_brain.tasks.models import TaskStatus
from june_brain.tasks.store import TasksStore

from ..schemas import HomeHoldings, NextDeadlineView
from ..schemas.silence import SurfacingDecisionView

logger = logging.getLogger(__name__)
router = APIRouter(tags=["home"])


def _promise_rollup(user_id: str) -> tuple[int, int, int, NextDeadlineView | None]:
    """Counts + soonest deadline over the user's active promises."""
    try:
        active = TasksStore(user_id=user_id).active()
    except Exception:  # noqa: BLE001 - home must render even if the store hiccups
        logger.debug("home promise rollup failed", exc_info=True)
        return 0, 0, 0, None

    waiting = [t for t in active if t.status == TaskStatus.AWAITING_USER]
    blocked_local = [t for t in waiting if (t.blocked_kind or "") == "local_only"]

    dated = [t for t in active if (t.due_at or "").strip()]
    next_deadline: NextDeadlineView | None = None
    if dated:
        soonest = min(dated, key=lambda t: t.due_at or "")
        next_deadline = NextDeadlineView(
            task_id=soonest.id, goal=soonest.goal, due_at=soonest.due_at or ""
        )
    return len(active), len(waiting), len(blocked_local), next_deadline


def _held_digest() -> tuple[list[SurfacingDecisionView], int]:
    """The batched surfacings June is holding (peek; does not drain)."""
    try:
        from june_brain.silence import get_store

        store = get_store()
        held = store.held_digest(limit=50)
        views = [
            SurfacingDecisionView(
                id=r.id,
                candidate_id=r.candidate_id,
                kind=r.kind,
                action=r.action,
                reason=r.reason,
                features=r.features,
                ts=r.ts,
                outcome=r.outcome,
                surfaced_at=r.surfaced_at,
                verdict=r.verdict,
            )
            for r in held
        ]
        return views, store.held_count()
    except Exception:  # noqa: BLE001
        logger.debug("home held-digest read failed", exc_info=True)
        return [], 0


def _needs_now() -> list[SurfacingDecisionView]:
    """Unresolved 'now' surfacings (peek; does not consume or mark as surfaced)."""
    try:
        from june_brain.silence import get_store

        recs = get_store().open_now_decisions(limit=20)
        return [
            SurfacingDecisionView(
                id=r.id,
                candidate_id=r.candidate_id,
                kind=r.kind,
                action=r.action,
                reason=r.reason,
                features=r.features,
                ts=r.ts,
                outcome=r.outcome,
                surfaced_at=r.surfaced_at,
                verdict=r.verdict,
            )
            for r in recs
        ]
    except Exception:  # noqa: BLE001
        logger.debug("home needs-now read failed", exc_info=True)
        return []


def _ledger_rollup() -> tuple[int, bool | None]:
    """(egress_today, chain_verified) from the Trust Ledger."""
    try:
        from june_brain.trust import get_reader, last_verification

        midnight = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        egress_today = get_reader().egress_count_since(midnight.isoformat())
        verified = last_verification()
        return egress_today, (verified[1] if verified else None)
    except Exception:  # noqa: BLE001
        logger.debug("home ledger rollup failed", exc_info=True)
        return 0, None


@router.get("/home/{user_id}/holdings", response_model=HomeHoldings)
def get_holdings(user_id: str) -> HomeHoldings:
    """What June is holding for this user: promises, held digest, egress, chain status."""
    try:
        from june_brain.silence.producers import run_silence_producers

        run_silence_producers(user_id)
    except Exception:
        logger.debug("home silence producer failed", exc_info=True)
    open_promises, waiting, blocked_local, next_deadline = _promise_rollup(user_id)
    held_digest, held_count = _held_digest()
    needs_now = _needs_now()
    egress_today, chain_verified = _ledger_rollup()
    return HomeHoldings(
        open_promises=open_promises,
        waiting_on_user=waiting,
        blocked_by_local_only=blocked_local,
        next_deadline=next_deadline,
        held_digest=held_digest,
        held_count=held_count,
        needs_now=needs_now,
        egress_today=egress_today,
        chain_verified=chain_verified,
    )
