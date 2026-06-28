"""Tests for the TaskRuntime — streams a goal through the hand-written loop and
records the trace (ADR 0018: one engine)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from june_brain.loop.interface import StreamEvent
from june_brain.tasks import (
    TaskRuntime,
    TasksStore,
    TaskStatus,
    TaskStepStatus,
)


class _StubLoop:
    """Replays a scripted sequence of StreamEvents via stream_turn."""

    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events
        self.sessions: list[Any] = []

    async def stream_turn(self, session: Any, user_msg: Any) -> Any:
        self.sessions.append(session)
        for ev in self._events:
            yield ev


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TasksStore:
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    return TasksStore(user_id="alice")


def _runtime_with_events(store: TasksStore, events: list[StreamEvent]) -> TaskRuntime:
    loop = _StubLoop(events)
    return TaskRuntime(store, loop_factory=lambda: loop)


# ---------------------------------------------------------------------------


def test_executing_a_goal_only_task_records_the_final_response(store: TasksStore) -> None:
    task = store.create(goal="Say hello")
    events = [StreamEvent(type="token", content="hello there")]
    runtime = _runtime_with_events(store, events)

    result = asyncio.run(runtime.execute(task.id))

    assert result.status == TaskStatus.COMPLETED
    assert result.final_deliverable == "hello there"
    assert len(result.plan) == 1
    assert result.plan[0].description == "Final response"
    assert result.plan[0].tool_result == "hello there"


def test_executing_records_tool_calls_with_args_and_results(store: TasksStore) -> None:
    task = store.create(goal="Check the weather")
    events = [
        StreamEvent(
            type="tool_call",
            tool_name="get_weather",
            tool_args={"city": "Amsterdam"},
        ),
        StreamEvent(
            type="tool_result", tool_name="get_weather", tool_result="sunny, 18C"
        ),
        StreamEvent(type="token", content="It's sunny and 18C in Amsterdam."),
    ]
    runtime = _runtime_with_events(store, events)

    result = asyncio.run(runtime.execute(task.id))

    assert result.status == TaskStatus.COMPLETED
    assert [step.status for step in result.plan] == [
        TaskStepStatus.COMPLETED,
        TaskStepStatus.COMPLETED,
    ]
    weather_step = result.plan[0]
    assert weather_step.tool_name == "get_weather"
    assert weather_step.tool_args == {"city": "Amsterdam"}
    assert weather_step.tool_result == "sunny, 18C"

    final_step = result.plan[-1]
    assert final_step.description == "Final response"
    assert final_step.tool_result == "It's sunny and 18C in Amsterdam."


def test_proposed_tool_with_no_result_is_recorded_as_skipped(store: TasksStore) -> None:
    task = store.create(goal="Try something")
    events = [
        StreamEvent(type="tool_call", tool_name="missing_tool", tool_args={"a": 1}),
        # Note: no tool_result for the call.
        StreamEvent(type="token", content="I tried but couldn't."),
    ]
    runtime = _runtime_with_events(store, events)

    result = asyncio.run(runtime.execute(task.id))

    skipped = next(s for s in result.plan if s.tool_name == "missing_tool")
    assert skipped.status == TaskStepStatus.SKIPPED
    assert skipped.tool_args == {"a": 1}


def test_tool_blocked_records_waiting_step_and_awaits_user(
    store: TasksStore,
) -> None:
    task = store.create(goal="Search for the latest weather")
    events = [
        StreamEvent(
            type="tool_blocked",
            tool_name="web_search",
            tool_args={"query": "Amsterdam weather"},
            detail='{"query": "Amsterdam weather"}',
        ),
        StreamEvent(type="token", content="I need approval before searching the web."),
    ]
    runtime = _runtime_with_events(store, events)

    result = asyncio.run(runtime.execute(task.id))

    assert result.status == TaskStatus.AWAITING_USER
    assert result.finished_at is None
    assert result.blocked_reason == "web_search is blocked by local-only mode."
    assert result.next_action == (
        "Switch to Private-by-default if this networked tool is acceptable, "
        "then retry the promise."
    )
    assert result.final_deliverable == "I need approval before searching the web."
    blocked_step = result.plan[0]
    assert blocked_step.status == TaskStepStatus.PENDING
    assert blocked_step.description == (
        "Waiting for user approval: web_search is blocked by local-only mode"
    )
    assert blocked_step.tool_name == "web_search"
    assert blocked_step.tool_args == {"query": "Amsterdam weather"}
    assert blocked_step.tool_result == '{"query": "Amsterdam weather"}'
    assert result.plan[-1].description == "Final response"


def test_approval_block_records_distinct_reason_and_next_action(
    store: TasksStore,
) -> None:
    """A guard approval block (needs_approval) must not be labelled local-only;
    its next action is to approve the action, not change the privacy dial."""
    task = store.create(goal="Text my friend the plan")
    events = [
        StreamEvent(
            type="tool_blocked",
            tool_name="send_telegram_message",
            tool_args={"text": "hi"},
            detail="Sends data off your device",
            needs_approval=True,
            action_class="write_network",
        ),
        StreamEvent(type="token", content="I need your OK before sending that."),
    ]
    runtime = _runtime_with_events(store, events)

    result = asyncio.run(runtime.execute(task.id))

    assert result.status == TaskStatus.AWAITING_USER
    assert result.finished_at is None
    assert result.blocked_reason == (
        "send_telegram_message needs your approval before it can run "
        "(Sends data off your device)."
    )
    assert result.next_action == "Approve this action, then retry the promise."
    blocked_step = result.plan[0]
    assert blocked_step.status == TaskStepStatus.PENDING
    assert blocked_step.description == "Waiting for your approval: send_telegram_message"
    assert "local-only" not in blocked_step.description


def test_failed_loop_marks_task_failed_with_error(store: TasksStore) -> None:
    task = store.create(goal="boom")

    class _BadLoop:
        async def stream_turn(self, session: Any, user_msg: Any) -> Any:
            raise RuntimeError("model unreachable")
            yield  # pragma: no cover - marks this as an async generator

    runtime = TaskRuntime(store, loop_factory=_BadLoop)

    with pytest.raises(RuntimeError):
        asyncio.run(runtime.execute(task.id))

    refreshed = store.get(task.id)
    assert refreshed is not None
    assert refreshed.status == TaskStatus.FAILED
    assert refreshed.error == "model unreachable"
    assert refreshed.finished_at is not None


def test_already_completed_task_is_a_noop(store: TasksStore) -> None:
    task = store.create(goal="x")
    store.set_status(task.id, TaskStatus.COMPLETED)

    class _ExplodeLoop:
        async def stream_turn(self, session: Any, user_msg: Any) -> Any:
            raise AssertionError("should not be called for completed task")
            yield  # pragma: no cover - marks this as an async generator

    runtime = TaskRuntime(store, loop_factory=_ExplodeLoop)

    result = asyncio.run(runtime.execute(task.id))
    assert result.status == TaskStatus.COMPLETED


def test_missing_task_raises(store: TasksStore) -> None:
    runtime = _runtime_with_events(store, [])
    with pytest.raises(KeyError):
        asyncio.run(runtime.execute("does-not-exist"))


def test_cancel_during_invoke_preserves_cancelled_state(store: TasksStore) -> None:
    """If the user PATCHes status=cancelled mid-run, the runtime must not
    overwrite the row with COMPLETED when the loop eventually returns.

    Simulates the race by flipping the row to CANCELLED from inside the loop's
    stream_turn; the runtime's trace step still runs, but the final status set
    should be skipped.
    """
    task = store.create(goal="cancel me")
    store_ref = store

    class _CancelMidwayLoop:
        async def stream_turn(self, session: Any, user_msg: Any) -> Any:
            store_ref.set_status(task.id, TaskStatus.CANCELLED)
            yield StreamEvent(type="token", content="too late")

    runtime = TaskRuntime(store, loop_factory=_CancelMidwayLoop)

    result = asyncio.run(runtime.execute(task.id))

    assert result.status == TaskStatus.CANCELLED
    # Trace still landed so the user can see what happened.
    assert any(step.description == "Final response" for step in result.plan)


def test_cancel_during_invoke_error_still_preserves_cancelled(store: TasksStore) -> None:
    """Cancel + loop crash should leave the task cancelled, not failed."""
    task = store.create(goal="cancel and crash")
    store_ref = store

    class _CancelThenCrashLoop:
        async def stream_turn(self, session: Any, user_msg: Any) -> Any:
            store_ref.set_status(task.id, TaskStatus.CANCELLED)
            raise RuntimeError("model died after cancel")
            yield  # pragma: no cover - marks this as an async generator

    runtime = TaskRuntime(store, loop_factory=_CancelThenCrashLoop)

    # No raise: the cancel-preservation branch swallows the loop error.
    result = asyncio.run(runtime.execute(task.id))
    assert result.status == TaskStatus.CANCELLED


def test_runtime_caps_steps_at_max(store: TasksStore) -> None:
    """A runaway loop must stop at _MAX_STEPS."""
    from june_brain.tasks import runtime as runtime_module

    task = store.create(goal="loop")
    # 60 alternating call+result pairs.
    events: list[StreamEvent] = []
    for i in range(60):
        events.append(
            StreamEvent(type="tool_call", tool_name="noop", tool_args={"i": i})
        )
        events.append(StreamEvent(type="tool_result", tool_name="noop", tool_result=str(i)))

    runtime = _runtime_with_events(store, events)
    result = asyncio.run(runtime.execute(task.id))

    tool_steps = [s for s in result.plan if s.tool_name == "noop"]
    assert len(tool_steps) <= runtime_module._MAX_STEPS
