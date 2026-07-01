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
from june_brain.silence.producers import (
    build_deadline_candidates,
    build_promise_nudge_candidates,
    run_silence_producers,
)
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
# build_promise_nudge_candidates
# ---------------------------------------------------------------------------


def _promise_task(
    task_id: str,
    goal: str,
    status: str,
    *,
    due_at: str | None = None,
    next_action: str | None = None,
    blocked_reason: str | None = None,
):
    """Minimal duck-type for promise-nudge tests."""
    from types import SimpleNamespace

    from june_brain.tasks.models import TaskStatus

    return SimpleNamespace(
        id=task_id,
        goal=goal,
        status=TaskStatus(status) if status else status,
        due_at=due_at,
        next_action=next_action,
        blocked_reason=blocked_reason,
    )


def test_awaiting_user_no_due_at_gives_candidate() -> None:
    task = _promise_task("p1", "waiting on user to approve", "awaiting_user")
    candidates = build_promise_nudge_candidates([task], now_iso=_NOW)
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.candidate_id == "promise_nudge:p1"
    assert cand.kind == "promise_nudge"
    assert cand.salience == 0.8
    assert "waiting on user to approve" in cand.summary
    assert cand.deadline_at is None


def test_awaiting_user_with_due_at_is_skipped() -> None:
    """Partition: tasks with a deadline are owned by build_deadline_candidates."""
    task = _promise_task(
        "p2", "has a deadline", "awaiting_user", due_at="2026-07-05T12:00:00+00:00"
    )
    candidates = build_promise_nudge_candidates([task], now_iso=_NOW)
    assert candidates == []


def test_non_awaiting_user_status_is_skipped() -> None:
    running = _promise_task("p3", "still running", "running")
    paused = _promise_task("p4", "paused", "paused")
    planning = _promise_task("p5", "planning", "planning")
    candidates = build_promise_nudge_candidates([running, paused, planning], now_iso=_NOW)
    assert candidates == []


def test_empty_tasks_gives_empty_candidates() -> None:
    assert build_promise_nudge_candidates([], now_iso=_NOW) == []


def test_hint_appended_from_next_action() -> None:
    task = _promise_task(
        "p6",
        "deploy to staging",
        "awaiting_user",
        next_action="approve the deploy step",
    )
    candidates = build_promise_nudge_candidates([task], now_iso=_NOW)
    assert len(candidates) == 1
    assert "approve the deploy step" in candidates[0].summary


def test_hint_falls_back_to_blocked_reason_when_no_next_action() -> None:
    task = _promise_task(
        "p7",
        "send the invoice",
        "awaiting_user",
        blocked_reason="waiting for bank details",
    )
    candidates = build_promise_nudge_candidates([task], now_iso=_NOW)
    assert len(candidates) == 1
    assert "waiting for bank details" in candidates[0].summary


def test_summary_truncated_to_max_chars() -> None:
    long_goal = "x" * 200
    task = _promise_task("p8", long_goal, "awaiting_user")
    candidates = build_promise_nudge_candidates([task], now_iso=_NOW)
    assert len(candidates) == 1
    assert len(candidates[0].summary) <= 120


def test_promise_nudge_salience_above_high_threshold() -> None:
    """_SALIENCE_AWAITING_USER must exceed HIGH_SALIENCE_THRESHOLD (0.7)."""
    from june_brain.silence.policy import HIGH_SALIENCE_THRESHOLD

    task = _promise_task("p9", "some pending promise", "awaiting_user")
    candidates = build_promise_nudge_candidates([task], now_iso=_NOW)
    assert candidates[0].salience > HIGH_SALIENCE_THRESHOLD


def test_promise_nudge_surfaces_when_idle() -> None:
    """High-salience nudge surfaces 'now' when user is idle and free."""
    task = _promise_task("p10", "idle nudge test", "awaiting_user")
    candidates = build_promise_nudge_candidates([task], now_iso=_NOW)
    cand = candidates[0]
    ctx = SurfacingContext(
        now=_NOW,
        presence_state=PRESENCE_IDLE,
        active_thread_open=False,
        dismissals_for_similar=0,
    )
    d = decide(cand, ctx)
    assert d.action == "now"


def test_promise_nudge_batched_mid_task() -> None:
    """Promise-nudge is held when the user is mid-task (active_thread_open)."""
    task = _promise_task("p11", "mid-task nudge test", "awaiting_user")
    candidates = build_promise_nudge_candidates([task], now_iso=_NOW)
    cand = candidates[0]
    ctx = SurfacingContext(
        now=_NOW,
        presence_state=PRESENCE_ACTIVE,
        active_thread_open=True,
        dismissals_for_similar=0,
    )
    d = decide(cand, ctx)
    assert d.action == "batch"


# ---------------------------------------------------------------------------
# run_silence_producers — promise-nudge integration
# ---------------------------------------------------------------------------


def test_run_silence_producers_records_promise_nudge() -> None:
    """run_silence_producers processes promise-nudge candidates from TasksStore."""
    user_id = "nudge_integration_user"

    ts = TasksStore(user_id=user_id)
    task = ts.create(goal="approve my PR")
    ts.set_blocked(task.id, reason="waiting on you", next_action="approve the action")

    store = get_store()
    run_silence_producers(user_id, now_iso=_NOW)

    rows = store.page(limit=200)
    nudge_rows = [r for r in rows if r.candidate_id == f"promise_nudge:{task.id}"]
    assert len(nudge_rows) == 1
    assert nudge_rows[0].kind == "promise_nudge"


def test_run_silence_producers_deadline_and_nudge_coexist() -> None:
    """Deadline producer and promise-nudge producer run in the same pass."""
    user_id = "coexist_user"
    DUE_AT = "2026-07-04T12:00:00+00:00"

    ts = TasksStore(user_id=user_id)
    # One task with a deadline (deadline producer).
    ts.create(goal="deadline promise", due_at=DUE_AT)
    # One awaiting-user task without a deadline (nudge producer).
    blocker = ts.create(goal="waiting promise")
    ts.set_blocked(blocker.id, reason="need your input", next_action="reply to me")

    store = get_store()
    run_silence_producers(user_id, now_iso=_NOW)

    rows = store.page(limit=200)
    kinds = {r.kind for r in rows}
    assert "deadline" in kinds
    assert "promise_nudge" in kinds


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


def test_open_now_decisions_returns_only_unresolved_now() -> None:
    """open_now_decisions returns action='now' AND outcome IS NULL, newest-first."""
    from june_brain.silence import SurfacingCandidate, get_store
    from june_brain.silence.policy import SurfacingDecision

    store = get_store()
    ts_older = "2026-06-29T10:00:00+00:00"
    ts_newer = "2026-06-29T11:00:00+00:00"

    cand1 = SurfacingCandidate(candidate_id="now_older", kind="deadline", salience=0.95)
    cand2 = SurfacingCandidate(candidate_id="now_newer", kind="deadline", salience=0.95)
    cand_batch = SurfacingCandidate(candidate_id="batch_one", kind="promise_nudge", salience=0.3)

    store.record(cand1, SurfacingDecision(action="now", reason="overdue", features={}), ts=ts_older)
    store.record(cand2, SurfacingDecision(action="now", reason="overdue", features={}), ts=ts_newer)
    store.record(cand_batch, SurfacingDecision(action="batch", reason="not urgent", features={}), ts=ts_older)

    rows = store.open_now_decisions(limit=20)
    # Only the 'now' rows, no 'batch', newest first.
    assert len(rows) == 2
    assert all(r.action == "now" for r in rows)
    assert all(r.outcome is None for r in rows)
    assert rows[0].ts > rows[1].ts  # newest first

    # Mark one as engaged — it should disappear from open_now_decisions.
    store.set_outcome(rows[1].id, "engaged")
    rows_after = store.open_now_decisions(limit=20)
    assert len(rows_after) == 1
    assert rows_after[0].outcome is None


def test_drain_digest_marks_surfaced_and_second_call_empty() -> None:
    """drain_digest marks held batch items surfaced; a second call returns empty."""
    from june_brain.silence import SurfacingCandidate, get_store
    from june_brain.silence.policy import SurfacingDecision

    store = get_store()
    cand = SurfacingCandidate(candidate_id="drain_c1", kind="promise_nudge", salience=0.3)
    store.record(cand, SurfacingDecision(action="batch", reason="not urgent", features={}))

    # First drain returns the held item and marks it surfaced.
    drained = store.drain_digest()
    assert len(drained) == 1
    assert drained[0].candidate_id == "drain_c1"

    # Second drain returns empty because all items are now marked surfaced.
    drained_again = store.drain_digest()
    assert drained_again == []


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
