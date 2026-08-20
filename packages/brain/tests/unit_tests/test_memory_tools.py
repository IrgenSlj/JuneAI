"""The four model-callable memory tools (ADR 0032).

These replace the seven v1 domain writers. The behaviour worth pinning is not
that they write — it is what they do when they are *unsure*, because that is
where the inversions live: `forget` must not resolve a near-tie by ranking, and
`update_promise` must not be able to claim work the runtime did not do.
"""

from __future__ import annotations

import pytest
from june_brain.memory import Memory, MemoryManager
from june_brain.memory import vector as vector_module
from june_brain.tasks.models import TaskStatus
from june_brain.tasks.store import TasksStore
from june_brain.tools_memory import (
    _too_close_to_call,
    forget,
    list_promises,
    remember,
    update_promise,
)

from .test_vector_store import _HashEmbedder

USER = "tools_memory_user"
STATE = {"user_id": USER}


@pytest.fixture(autouse=True)
def _local_embedder(monkeypatch):
    """Make the real MemoryManager path work without Ollama."""
    vector_module.reset_singletons()
    monkeypatch.setattr(vector_module, "_default_embedder", _HashEmbedder())
    Memory(USER)  # create tables
    yield
    vector_module.reset_singletons()


# ---------------------------------------------------------------------------
# remember
# ---------------------------------------------------------------------------


def test_remember_stores_a_recallable_fact() -> None:
    result = remember.invoke({"text": "Her sister is called Mira.", "state": STATE})

    assert "Remembered" in result
    hits = MemoryManager(USER).recall("Her sister is called Mira.", k=5)
    assert any("Mira" in h.get("text", "") for h in hits)


def test_remember_rejects_empty_text_without_writing() -> None:
    assert "Nothing to remember" in remember.invoke({"text": "   ", "state": STATE})


def test_remember_reports_a_write_failure_instead_of_claiming_success(monkeypatch) -> None:
    """A Glass Box rule: a tool result may not assert something that did not happen."""

    class _Boom:
        def write(self, *_args, **_kwargs):
            raise RuntimeError("disk full")

    monkeypatch.setattr("june_brain.memory.MemoryManager", lambda *_a, **_k: _Boom())
    result = remember.invoke({"text": "something durable", "state": STATE})

    assert "Could not store" in result
    assert "Remembered" not in result


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


def test_forget_removes_a_single_clear_match() -> None:
    remember.invoke({"text": "The user is allergic to penicillin.", "state": STATE})

    result = forget.invoke({"description": "allergic to penicillin", "state": STATE})

    assert "Forgotten" in result
    assert "restored" in result  # reversibility is stated, not implied
    remaining = MemoryManager(USER).recall("allergic to penicillin", k=5)
    assert not any("penicillin" in h.get("text", "") for h in remaining)


def test_forget_says_so_when_nothing_matches() -> None:
    result = forget.invoke({"description": "a memory never stored", "state": STATE})
    assert "nothing was forgotten" in result


def test_forget_refuses_to_choose_between_near_ties(monkeypatch) -> None:
    """Inversion 1: the one place being confidently wrong destroys user data."""
    deleted: list[str] = []

    class _Ambiguous:
        def recall(self, _query, k=5):
            return [
                {"source": "vector", "ref": "a", "text": "Coffee order is a flat white.", "score": 0.40},
                {"source": "vector", "ref": "b", "text": "Coffee order is a cortado.", "score": 0.41},
            ]

        def forget(self, ref):
            deleted.append(ref)
            return True

    monkeypatch.setattr("june_brain.memory.MemoryManager", lambda *_a, **_k: _Ambiguous())
    result = forget.invoke({"description": "my coffee order", "state": STATE})

    assert deleted == [], "a near-tie was resolved by ranking instead of by asking"
    assert "nothing was forgotten" in result
    assert "flat white" in result and "cortado" in result


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"source": "vector", "score": None}, {"source": "vector", "score": 0.4}),
        ({"source": "vector", "score": 0.4}, {"source": "sqlite", "score": 0.4}),
        ({"source": "vector", "score": 0.0}, {"source": "vector", "score": 0.0}),
    ],
)
def test_ambiguity_check_fails_toward_asking(first, second) -> None:
    """Missing scores and cross-source comparisons are ambiguous, not decidable."""
    assert _too_close_to_call(first, second) is True


def test_a_clear_winner_is_not_ambiguous() -> None:
    assert _too_close_to_call(
        {"source": "vector", "score": 0.9}, {"source": "vector", "score": 0.2}
    ) is False


# ---------------------------------------------------------------------------
# promises
# ---------------------------------------------------------------------------


def test_list_promises_renders_blocked_state_explicitly() -> None:
    store = TasksStore(user_id=USER)
    task = store.create(goal="Book the flight to Lisbon")
    store.set_blocked(
        task.id,
        reason="Needs the travel dates.",
        next_action="Ask the user which week.",
    )

    result = list_promises.invoke({"state": STATE})

    assert "Book the flight to Lisbon" in result
    assert "Needs the travel dates." in result
    assert "Ask the user which week." in result


def test_list_promises_is_honest_about_having_none() -> None:
    assert list_promises.invoke({"state": STATE}) == "No open promises."


def test_update_promise_completes_by_short_id() -> None:
    store = TasksStore(user_id=USER)
    task = store.create(goal="Send the contract")

    result = update_promise.invoke(
        {"promise": task.id[:8], "status": "completed", "state": STATE}
    )

    assert "completed" in result
    assert store.get(task.id).status == TaskStatus.COMPLETED


def test_update_promise_cannot_set_running() -> None:
    """`running` means the runtime is executing it; a tool must not assert that."""
    store = TasksStore(user_id=USER)
    task = store.create(goal="Draft the announcement")

    result = update_promise.invoke(
        {"promise": task.id, "status": "running", "state": STATE}
    )

    assert "not a status June can set" in result
    assert store.get(task.id).status != TaskStatus.RUNNING


def test_update_promise_rejects_an_unknown_id() -> None:
    result = update_promise.invoke(
        {"promise": "deadbeef", "status": "completed", "state": STATE}
    )
    assert "No open promise matches" in result


def test_update_promise_records_a_next_action_without_a_status() -> None:
    store = TasksStore(user_id=USER)
    task = store.create(goal="Renew the domain")

    result = update_promise.invoke(
        {"promise": task.id, "next_action": "Confirm the registrar login.", "state": STATE}
    )

    assert "Confirm the registrar login." in result
    assert store.get(task.id).next_action == "Confirm the registrar login."


def test_update_promise_needs_something_to_do() -> None:
    store = TasksStore(user_id=USER)
    task = store.create(goal="Nothing to change")
    assert "Nothing to update" in update_promise.invoke(
        {"promise": task.id, "state": STATE}
    )


# ---------------------------------------------------------------------------
# scheduler tools — identity, not a default
# ---------------------------------------------------------------------------


def test_a_schedule_cannot_be_deleted_across_users() -> None:
    """`ScheduleStore.delete` is not user-scoped; the caller must check.

    The /schedules route always did. The tool did not, and it never read the
    user at all, so a schedule id was enough to delete someone else's job.
    """
    from june_brain.memory.sqlite import _get_connection, db_path
    from june_brain.scheduler.models import _SCHEDULES_TABLE_SQL
    from june_brain.scheduler.models import Schedule as _Schedule
    from june_brain.scheduler.store import ScheduleStore
    from june_brain.tools import delete_schedule

    conn = _get_connection(db_path())
    conn.executescript(_SCHEDULES_TABLE_SQL)
    conn.commit()
    store = ScheduleStore(conn)
    theirs = store.create(_Schedule(user_id="someone_else", name="Their job"))

    result = delete_schedule.invoke(
        {"schedule_id": theirs.id, "state": {"user_id": "attacker"}}
    )

    assert "not found" in result
    assert store.get(theirs.id) is not None, "another user's schedule was deleted"


def test_a_scheduler_tool_without_state_raises_rather_than_guessing() -> None:
    """A missing identity is a bug to surface, not a partition to guess."""
    from june_brain.tools import list_schedules

    with pytest.raises(ValueError, match="injected agent state"):
        list_schedules.func(state=None)


def test_update_promise_matches_the_words_the_user_uses() -> None:
    """Requiring an id measured 0/12 on the local model (D.5d).

    The id exists only inside a `list_promises` result, so the tool could not
    be reached without chaining two calls, and a 2B model does not chain
    reliably. A user also says a promise is done in different words than the
    promise was written in — "the passport renewal" for "Renew the passport" —
    so the match is on content words, prefix-tolerant.
    """
    store = TasksStore(user_id=USER)
    store.create(goal="Renew the passport")
    store.create(goal="File the tax return")

    result = update_promise.invoke(
        {"promise": "the passport renewal", "status": "completed", "state": STATE}
    )

    assert "Renew the passport" in result
    done = [t for t in store.list() if t.goal == "Renew the passport"][0]
    assert done.status == TaskStatus.COMPLETED


def test_update_promise_refuses_to_choose_between_matches() -> None:
    """Same refusal as `forget`: a tie is asked about, not ranked."""
    store = TasksStore(user_id=USER)
    store.create(goal="Book the flight to Lisbon")
    store.create(goal="Book a dentist appointment")

    result = update_promise.invoke(
        {"promise": "the booking", "status": "cancelled", "state": STATE}
    )

    assert "nothing was changed" in result
    assert "Lisbon" in result and "dentist" in result
    assert all(t.status != TaskStatus.CANCELLED for t in store.list())


def test_update_promise_ignores_a_promise_that_is_not_open() -> None:
    """Matching runs over active promises, so a finished one is not re-opened."""
    store = TasksStore(user_id=USER)
    finished = store.create(goal="Post the letter")
    store.set_status(finished.id, TaskStatus.COMPLETED)

    result = update_promise.invoke(
        {"promise": "the letter", "status": "cancelled", "state": STATE}
    )

    assert "No open promise matches" in result


def test_a_schedule_with_no_time_is_refused_not_confirmed() -> None:
    """A tool result may not assert something that did not happen.

    Neither a cron nor an interval is not a schedule: `compute_next_run` treats
    it as a one-shot due immediately and finished, so the row existed, never
    usefully ran, and the tool answered "Scheduled 'X' with cron ''." — a
    confirmation of a recurring job that was never created. The same shape as
    the no-op UI tools D.5a deleted, where every layer is honest about a lie it
    was handed.
    """
    from june_brain.tools import create_schedule, list_schedules

    result = create_schedule.invoke({"name": "Water the plants", "state": STATE})

    assert "Cannot schedule" in result
    assert "Scheduled" not in result
    assert list_schedules.invoke({"state": STATE}) == "No schedules."


def test_a_schedule_with_a_time_still_works() -> None:
    from june_brain.tools import create_schedule, list_schedules

    by_cron = create_schedule.invoke(
        {"name": "Morning brief", "cron_expression": "0 8 * * *", "state": STATE}
    )
    by_interval = create_schedule.invoke(
        {"name": "Ping", "interval_seconds": 3600, "state": STATE}
    )

    assert "Morning brief" in by_cron and "0 8 * * *" in by_cron
    assert "every 3600s" in by_interval
    listed = list_schedules.invoke({"state": STATE})
    assert "Morning brief" in listed and "Ping" in listed
