"""Read-time Silence Model producers (ADR 0023).

Wires the Silence Model so it produces real decisions at read time: when
``GET /home/{user_id}/holdings`` is served, ``run_silence_producers`` builds
candidates from live state, runs each through ``decide``, and persists results
idempotently via ``record_if_new``.

**Why write on a GET?** ADR 0016 forbids timer-driven proactivity. There is no
heartbeat, no background loop, no periodic job. Production is triggered by the
natural event boundary of the user consulting their Home view — the moment when
surfacing is actually meaningful. The write is idempotent (``record_if_new``
deduplicates unchanged decisions) so re-fetching the page does not grow rows.

The clock is read exactly once in ``run_silence_producers`` (if not injected).
``decide`` and ``derive_presence`` never read the clock; they receive ``now_iso``
as a parameter so decisions are deterministic and testable (ADR 0023 invariant).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from .policy import SurfacingCandidate, SurfacingContext, decide
from .presence import derive_presence
from .store import get_store

logger = logging.getLogger(__name__)

# Salience thresholds by time-to-deadline bucket (hours).
_SALIENCE_OVERDUE = 0.95
_SALIENCE_LE_3H = 0.90
_SALIENCE_LE_24H = 0.78
_SALIENCE_LE_72H = 0.62
_SALIENCE_DISTANT = 0.45

# Salience for a promise that is explicitly waiting on the user (AWAITING_USER,
# no deadline).  Set above HIGH_SALIENCE_THRESHOLD (0.7) so the policy CAN
# surface it when the user is idle and free, while dismissal/presence/mid-task
# rules still gate it.
_SALIENCE_AWAITING_USER = 0.8

_SUMMARY_MAX_CHARS = 120


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _deadline_hours(due_at: str, now_iso: str) -> float | None:
    """Hours from now until due_at (negative if overdue), or None on parse failure."""
    deadline = _parse_iso(due_at)
    now = _parse_iso(now_iso)
    if deadline is None or now is None:
        return None
    try:
        return (deadline - now).total_seconds() / 3600.0
    except (TypeError, ValueError, OverflowError):
        return None


def _salience_for_hours(hours: float) -> float:
    if hours < 0:
        return _SALIENCE_OVERDUE
    if hours <= 3.0:
        return _SALIENCE_LE_3H
    if hours <= 24.0:
        return _SALIENCE_LE_24H
    if hours <= 72.0:
        return _SALIENCE_LE_72H
    return _SALIENCE_DISTANT


def build_deadline_candidates(
    tasks: list,
    *,
    now_iso: str,
) -> list[SurfacingCandidate]:
    """Build SurfacingCandidates for active tasks that have a non-empty ``due_at``.

    Skips tasks whose ``due_at`` fails to parse. Salience is derived purely from
    time-to-deadline so a task approaching its deadline escalates automatically
    on the next Home fetch without any manual re-classification.
    """
    candidates: list[SurfacingCandidate] = []
    for task in tasks:
        due_at = (getattr(task, "due_at", None) or "").strip()
        if not due_at:
            continue
        hours = _deadline_hours(due_at, now_iso)
        if hours is None:
            continue
        salience = _salience_for_hours(hours)
        goal = str(getattr(task, "goal", "") or "")
        summary = goal[:_SUMMARY_MAX_CHARS]
        candidates.append(
            SurfacingCandidate(
                candidate_id=f"deadline:{task.id}",
                kind="deadline",
                summary=summary,
                salience=salience,
                deadline_at=due_at,
            )
        )
    return candidates


def build_promise_nudge_candidates(
    tasks: list,
    *,
    now_iso: str,  # kept for API symmetry with build_deadline_candidates
) -> list[SurfacingCandidate]:
    """Build SurfacingCandidates for AWAITING_USER tasks that have no deadline.

    Partition rule: tasks that carry a non-empty ``due_at`` are already owned by
    ``build_deadline_candidates``; surfacing them here too would duplicate the same
    promise.  So this producer only covers awaiting-user promises with NO deadline.
    """
    from june_brain.tasks.models import TaskStatus

    candidates: list[SurfacingCandidate] = []
    for task in tasks:
        # Only surface tasks explicitly waiting on the user.
        if getattr(task, "status", None) != TaskStatus.AWAITING_USER:
            continue
        # Partition: tasks with a deadline are handled by build_deadline_candidates.
        # Avoid double-surfacing the same promise through two producers.
        due_at = (getattr(task, "due_at", None) or "").strip()
        if due_at:
            continue
        goal = str(getattr(task, "goal", "") or "")
        summary = goal[:_SUMMARY_MAX_CHARS]
        # Optionally append next_action or blocked_reason as a short hint,
        # staying within the char cap.
        hint = str(getattr(task, "next_action", "") or "") or str(
            getattr(task, "blocked_reason", "") or ""
        )
        if hint:
            available = _SUMMARY_MAX_CHARS - len(summary)
            if available > 5:
                suffix = f" — {hint}"
                summary = (summary + suffix)[:_SUMMARY_MAX_CHARS]
        candidates.append(
            SurfacingCandidate(
                candidate_id=f"promise_nudge:{task.id}",
                kind="promise_nudge",
                summary=summary,
                salience=_SALIENCE_AWAITING_USER,
                deadline_at=None,
            )
        )
    return candidates


def run_silence_producers(user_id: str, *, now_iso: str | None = None) -> list:
    """Build and persist Silence Model decisions for the given user (best-effort).

    Any exception is caught and logged at DEBUG so the Home endpoint always
    renders. Returns a list of newly recorded SurfacingRecords (empty on error
    or when all decisions are unchanged since the last run).
    """
    if now_iso is None:
        now_iso = datetime.now(UTC).isoformat()

    results = []
    try:
        from june_brain.activity import ActivityLog
        from june_brain.tasks.store import TasksStore

        tasks = TasksStore(user_id=user_id).active()
        activity_entries = ActivityLog().list(limit=50)

        presence_state, active_thread_open = derive_presence(
            activity_entries, now_iso=now_iso
        )

        candidates = build_deadline_candidates(tasks, now_iso=now_iso) + build_promise_nudge_candidates(tasks, now_iso=now_iso)
        store = get_store()

        for candidate in candidates:
            try:
                dismissals = store.dismissals_for_similar(candidate.kind)
                ctx = SurfacingContext(
                    now=now_iso,
                    presence_state=presence_state,
                    active_thread_open=active_thread_open,
                    dismissals_for_similar=dismissals,
                )
                decision = decide(candidate, ctx)
                rec = store.record_if_new(candidate, decision, ts=now_iso)
                if rec is not None:
                    results.append(rec)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "silence producer failed for candidate %s",
                    candidate.candidate_id,
                    exc_info=True,
                )
    except Exception:  # noqa: BLE001
        logger.debug("silence producer outer failure", exc_info=True)

    return results
