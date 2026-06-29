"""Tests for Silence Model producers and presence derivation (ADR 0023)."""

from __future__ import annotations

from june_brain.activity import ActivityEntry
from june_brain.silence import (
    PRESENCE_ABSENT,
    PRESENCE_ACTIVE,
    PRESENCE_IDLE,
    SurfacingContext,
    decide,
    get_store,
)
from june_brain.silence.presence import derive_presence
from june_brain.silence.producers import build_deadline_candidates, run_silence_producers
from june_brain.tasks.store import TasksStore

# Fixed reference timestamp used across all tests that need a deterministic now.
_NOW = "2026-06-29T12:00:00+00:00"

# ---------------------------------------------------------------------------
# derive_presence
# ---------------------------------------------------------------------------


def _entry(timestamp: str, label: str = "POST /some/path") -> ActivityEntry:
    return ActivityEntry(timestamp=timestamp, label=label, kind="request")


def test_derive_presence_active_30s_ago() -> None:
    entry = _entry("2026-06-29T11:59:30+00:00")  # 30 s before _NOW
    state, thread_open = derive_presence([entry], now_iso=_NOW)
    assert state == PRESENCE_ACTIVE


def test_derive_presence_idle_10min_ago() -> None:
    entry = _entry("2026-06-29T11:50:00+00:00")  # 600 s -> idle
    state, thread_open = derive_presence([entry], now_iso=_NOW)
    assert state == PRESENCE_IDLE
    assert thread_open is False  # > 300 s window


def test_derive_presence_absent_2h_ago() -> None:
    entry = _entry("2026-06-29T10:00:00+00:00")  # 7200 s -> absent
    state, thread_open = derive_presence([entry], now_iso=_NOW)
    assert state == PRESENCE_ABSENT
    assert thread_open is False


def test_derive_presence_empty_entries() -> None:
    state, thread_open = derive_presence([], now_iso=_NOW)
    assert state == PRESENCE_ABSENT
    assert thread_open is False


def test_derive_presence_active_thread_open_on_post_chat() -> None:
    # Entry 30 s ago, label is "POST /chat" -> active + thread open.
    entry = _entry("2026-06-29T11:59:30+00:00", label="POST /chat")
    state, thread_open = derive_presence([entry], now_iso=_NOW)
    assert state == PRESENCE_ACTIVE
    assert thread_open is True


def test_derive_presence_thread_not_open_for_history() -> None:
    # "GET /chat/history" should NOT set active_thread_open.
    entry = _entry("2026-06-29T11:59:30+00:00", label="GET /chat/history")
    _, thread_open = derive_presence([entry], now_iso=_NOW)
    assert thread_open is False


def test_derive_presence_skips_unparseable_timestamps() -> None:
    bad = _entry("not-a-date")
    good = _entry("2026-06-29T11:59:30+00:00")
    state, _ = derive_presence([bad, good], now_iso=_NOW)
    assert state == PRESENCE_ACTIVE


# ---------------------------------------------------------------------------
# build_deadline_candidates + decide
# ---------------------------------------------------------------------------


def _task_obj(task_id: str, goal: str, due_at: str | None):
    """Minimal duck-type matching what TasksStore.create returns."""
    from types import SimpleNamespace

    return SimpleNamespace(id=task_id, goal=goal, due_at=due_at)


def test_due_2h_gives_salience_and_now() -> None:
    task = _task_obj("t1", "finish the report", "2026-06-29T14:00:00+00:00")
    candidates = build_deadline_candidates([task], now_iso=_NOW)
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.salience >= 0.9
    ctx = SurfacingContext(now=_NOW, presence_state=PRESENCE_ABSENT)
    d = decide(cand, ctx)
    assert d.action == "now"


def test_due_5_days_gives_batch() -> None:
    task = _task_obj("t2", "plan the quarter", "2026-07-04T12:00:00+00:00")
    candidates = build_deadline_candidates([task], now_iso=_NOW)
    assert len(candidates) == 1
    ctx = SurfacingContext(now=_NOW, presence_state=PRESENCE_ABSENT)
    d = decide(candidates[0], ctx)
    assert d.action == "batch"


def test_overdue_gives_now() -> None:
    task = _task_obj("t3", "submit expense report", "2026-06-29T11:00:00+00:00")
    candidates = build_deadline_candidates([task], now_iso=_NOW)
    assert len(candidates) == 1
    assert candidates[0].salience == 0.95
    ctx = SurfacingContext(now=_NOW, presence_state=PRESENCE_ABSENT)
    d = decide(candidates[0], ctx)
    assert d.action == "now"


def test_task_without_due_at_is_skipped() -> None:
    task = _task_obj("t4", "vague promise", None)
    candidates = build_deadline_candidates([task], now_iso=_NOW)
    assert candidates == []


def test_task_with_bad_due_at_is_skipped() -> None:
    task = _task_obj("t5", "another promise", "not-a-date")
    candidates = build_deadline_candidates([task], now_iso=_NOW)
    assert candidates == []


# ---------------------------------------------------------------------------
# run_silence_producers idempotency
# ---------------------------------------------------------------------------


def test_idempotent_same_action() -> None:
    """Calling the producer twice for the same unchanged state must not grow rows."""
    user_id = "idempotency_user"
    DUE_AT = "2026-07-04T12:00:00+00:00"  # 5 days away -> batch

    ts = TasksStore(user_id=user_id)
    ts.create(goal="batch promise", due_at=DUE_AT)

    run_silence_producers(user_id, now_iso=_NOW)
    run_silence_producers(user_id, now_iso=_NOW)

    store = get_store()
    rows = store.page(limit=200)
    # Exactly one row: second call was a no-op (same action).
    assert len(rows) == 1


def test_state_transition_adds_new_row() -> None:
    """A batch->now transition (deadline approaching) must record a second row."""
    user_id = "transition_user"
    DUE_AT = "2026-07-03T16:00:00+00:00"

    # First call: task is ~4 days away -> batch.
    NOW_FAR = "2026-06-29T12:00:00+00:00"
    # Second call: task is 2 h away -> now.
    NOW_NEAR = "2026-07-03T14:00:00+00:00"

    ts = TasksStore(user_id=user_id)
    ts.create(goal="approaching deadline", due_at=DUE_AT)

    run_silence_producers(user_id, now_iso=NOW_FAR)
    run_silence_producers(user_id, now_iso=NOW_NEAR)

    store = get_store()
    rows = store.page(limit=200)
    # Two rows: one batch (initial) and one now (transition).
    assert len(rows) == 2
    actions = {r.action for r in rows}
    assert "batch" in actions
    assert "now" in actions
