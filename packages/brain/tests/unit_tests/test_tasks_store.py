"""Tests for the tasks store (ADR 0010, Sprint 1.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from june_brain.tasks import TasksStore, TaskStatus, TaskStep, TaskStepStatus


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TasksStore:
    """A clean tasks store backed by a fresh per-test june.db."""
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    # Patch both the package attribute and the sqlite module's lazy reader so
    # _current_memory_dir() picks up tmp_path for this test only.
    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    return TasksStore(user_id="alice")


def test_create_returns_a_task_with_id_and_default_status(store: TasksStore) -> None:
    task = store.create(goal="Plan a weekend trip")
    assert task.id
    assert task.goal == "Plan a weekend trip"
    assert task.status == TaskStatus.PLANNING
    assert task.plan == []
    assert task.user_id == "alice"


def test_create_persists_through_get(store: TasksStore) -> None:
    created = store.create(goal="Draft a reply", owner_skill="gmail")
    fetched = store.get(created.id)
    assert fetched is not None
    assert fetched.goal == "Draft a reply"
    assert fetched.owner_skill == "gmail"
    assert fetched.status == TaskStatus.PLANNING


def test_get_returns_none_for_unknown_id(store: TasksStore) -> None:
    assert store.get("does-not-exist") is None


def test_list_returns_newest_first(store: TasksStore) -> None:
    t1 = store.create(goal="first")
    t2 = store.create(goal="second")
    t3 = store.create(goal="third")
    rows = store.list()
    assert [t.id for t in rows][:3] == [t3.id, t2.id, t1.id]


def test_list_filters_by_status(store: TasksStore) -> None:
    a = store.create(goal="planning task")
    b = store.create(goal="running task")
    store.set_status(b.id, TaskStatus.RUNNING)
    running = store.list(status=TaskStatus.RUNNING)
    assert [t.id for t in running] == [b.id]
    assert all(t.status == TaskStatus.RUNNING for t in running)
    planning = store.list(status=TaskStatus.PLANNING)
    assert {t.id for t in planning} == {a.id}


def test_active_includes_planning_running_paused_awaiting_user(store: TasksStore) -> None:
    planning = store.create(goal="planning")
    running = store.create(goal="running")
    paused = store.create(goal="paused")
    awaiting = store.create(goal="awaiting")
    done = store.create(goal="done")
    cancelled = store.create(goal="cancelled")

    store.set_status(running.id, TaskStatus.RUNNING)
    store.set_status(paused.id, TaskStatus.PAUSED)
    store.set_status(awaiting.id, TaskStatus.AWAITING_USER)
    store.set_status(done.id, TaskStatus.COMPLETED)
    store.set_status(cancelled.id, TaskStatus.CANCELLED)

    ids = {t.id for t in store.active()}
    assert ids == {planning.id, running.id, paused.id, awaiting.id}


def test_set_status_running_stamps_started_at(store: TasksStore) -> None:
    t = store.create(goal="x")
    assert t.started_at is None
    after = store.set_status(t.id, TaskStatus.RUNNING)
    assert after is not None
    assert after.started_at is not None
    assert after.status == TaskStatus.RUNNING


def test_set_status_completed_stamps_finished_at(store: TasksStore) -> None:
    t = store.create(goal="x")
    store.set_status(t.id, TaskStatus.RUNNING)
    after = store.set_status(t.id, TaskStatus.COMPLETED)
    assert after is not None
    assert after.finished_at is not None
    assert after.status == TaskStatus.COMPLETED


def test_set_status_failed_records_error(store: TasksStore) -> None:
    t = store.create(goal="x")
    after = store.set_status(t.id, TaskStatus.FAILED, error="tool refused")
    assert after is not None
    assert after.status == TaskStatus.FAILED
    assert after.error == "tool refused"
    assert after.finished_at is not None


def test_set_plan_replaces_plan_steps(store: TasksStore) -> None:
    t = store.create(goal="x")
    steps = [
        TaskStep(index=0, description="open browser"),
        TaskStep(index=1, description="navigate"),
    ]
    after = store.set_plan(t.id, steps)
    assert after is not None
    assert [s.description for s in after.plan] == ["open browser", "navigate"]


def test_append_step_assigns_index_and_persists(store: TasksStore) -> None:
    t = store.create(goal="x")
    store.append_step(t.id, TaskStep(description="first"))
    store.append_step(t.id, TaskStep(description="second"))
    fetched = store.get(t.id)
    assert fetched is not None
    assert [s.index for s in fetched.plan] == [0, 1]
    assert [s.description for s in fetched.plan] == ["first", "second"]


def test_update_step_marks_running_and_completed(store: TasksStore) -> None:
    t = store.create(goal="x")
    store.append_step(t.id, TaskStep(id="s1", description="step"))
    store.update_step(t.id, "s1", status=TaskStepStatus.RUNNING)
    t1 = store.get(t.id)
    assert t1 is not None
    assert t1.plan[0].status == TaskStepStatus.RUNNING
    assert t1.plan[0].started_at is not None

    store.update_step(
        t.id,
        "s1",
        status=TaskStepStatus.COMPLETED,
        tool_result={"ok": True},
        finished=True,
        model_provenance={"provider": "gemma", "tier": "local"},
    )
    t2 = store.get(t.id)
    assert t2 is not None
    assert t2.plan[0].status == TaskStepStatus.COMPLETED
    assert t2.plan[0].tool_result == {"ok": True}
    assert t2.plan[0].finished_at is not None
    assert t2.plan[0].model_provenance == {"provider": "gemma", "tier": "local"}


def test_update_step_can_clear_tool_result_to_none(store: TasksStore) -> None:
    """tool_result=None should be a real update, not 'leave unchanged'."""
    t = store.create(goal="x")
    store.append_step(t.id, TaskStep(id="s1", description="step"))
    store.update_step(t.id, "s1", tool_result={"first": "value"})
    store.update_step(t.id, "s1", tool_result=None)
    fetched = store.get(t.id)
    assert fetched is not None
    assert fetched.plan[0].tool_result is None


def test_delete_returns_true_and_removes(store: TasksStore) -> None:
    t = store.create(goal="x")
    assert store.delete(t.id) is True
    assert store.get(t.id) is None
    assert store.delete(t.id) is False  # second delete: nothing to remove


def test_user_scoping_isolates_two_users(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    alice = TasksStore(user_id="alice")
    bob = TasksStore(user_id="bob")

    a = alice.create(goal="alice task")
    b = bob.create(goal="bob task")

    assert {t.id for t in alice.list()} == {a.id}
    assert {t.id for t in bob.list()} == {b.id}
    assert alice.get(b.id) is None
    assert bob.get(a.id) is None


def test_task_round_trips_plan_through_json(store: TasksStore) -> None:
    t = store.create(goal="x", plan=[TaskStep(description="alpha")])
    store.append_step(t.id, TaskStep(description="beta", tool_args={"k": "v"}))
    fetched = store.get(t.id)
    assert fetched is not None
    assert [s.description for s in fetched.plan] == ["alpha", "beta"]
    assert fetched.plan[1].tool_args == {"k": "v"}
