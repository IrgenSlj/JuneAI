"""Tests for the TaskRuntime — feeds a goal to the chat agent and records the trace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from june_brain.tasks import (
    TaskRuntime,
    TasksStore,
    TaskStatus,
    TaskStepStatus,
)


@dataclass
class _FakeHumanMessage:
    content: str
    type: str = "human"


@dataclass
class _FakeAIMessage:
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    type: str = "ai"


@dataclass
class _FakeToolMessage:
    content: str
    tool_call_id: str
    name: str = ""
    type: str = "tool"


class _StubAgent:
    """Records the invoked state and returns a canned message trace."""

    def __init__(self, trace: list[Any]) -> None:
        self.trace = trace
        self.invocations: list[dict[str, Any]] = []

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.invocations.append(state)
        return {"messages": [*state["messages"], *self.trace]}


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TasksStore:
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    return TasksStore(user_id="alice")


def _runtime_with_trace(store: TasksStore, trace: list[Any]) -> TaskRuntime:
    agent = _StubAgent(trace)
    return TaskRuntime(
        store,
        agent_factory=lambda: agent,
        message_factory=_FakeHumanMessage,
    )


# ---------------------------------------------------------------------------


def test_executing_a_goal_only_task_records_the_final_response(store: TasksStore) -> None:
    task = store.create(goal="Say hello")
    trace = [_FakeAIMessage(content="hello there")]
    runtime = _runtime_with_trace(store, trace)

    import asyncio
    result = asyncio.run(runtime.execute(task.id))

    assert result.status == TaskStatus.COMPLETED
    assert len(result.plan) == 1
    assert result.plan[0].description == "Final response"
    assert result.plan[0].tool_result == "hello there"


def test_executing_records_tool_calls_with_args_and_results(store: TasksStore) -> None:
    task = store.create(goal="Check the weather")
    trace = [
        _FakeAIMessage(
            content="",
            tool_calls=[
                {"id": "call_1", "name": "get_weather", "args": {"city": "Amsterdam"}}
            ],
        ),
        _FakeToolMessage(content="sunny, 18C", tool_call_id="call_1", name="get_weather"),
        _FakeAIMessage(content="It's sunny and 18C in Amsterdam."),
    ]
    runtime = _runtime_with_trace(store, trace)

    import asyncio
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
    trace = [
        _FakeAIMessage(
            content="",
            tool_calls=[{"id": "call_x", "name": "missing_tool", "args": {"a": 1}}],
        ),
        # Note: no ToolMessage for call_x.
        _FakeAIMessage(content="I tried but couldn't."),
    ]
    runtime = _runtime_with_trace(store, trace)

    import asyncio
    result = asyncio.run(runtime.execute(task.id))

    # First step: the proposed-but-unanswered call, marked skipped.
    skipped = next(s for s in result.plan if s.tool_name == "missing_tool")
    assert skipped.status == TaskStepStatus.SKIPPED
    assert skipped.tool_args == {"a": 1}


def test_failed_agent_marks_task_failed_with_error(store: TasksStore) -> None:
    task = store.create(goal="boom")

    class _BadAgent:
        async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("model unreachable")

    runtime = TaskRuntime(
        store,
        agent_factory=_BadAgent,
        message_factory=_FakeHumanMessage,
    )

    import asyncio
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

    class _ExplodeAgent:
        async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("should not be called for completed task")

    runtime = TaskRuntime(
        store,
        agent_factory=_ExplodeAgent,
        message_factory=_FakeHumanMessage,
    )

    import asyncio
    result = asyncio.run(runtime.execute(task.id))
    assert result.status == TaskStatus.COMPLETED


def test_missing_task_raises(store: TasksStore) -> None:
    runtime = _runtime_with_trace(store, [])
    import asyncio
    with pytest.raises(KeyError):
        asyncio.run(runtime.execute("does-not-exist"))


def test_cancel_during_invoke_preserves_cancelled_state(store: TasksStore) -> None:
    """If the user PATCHes status=cancelled mid-invoke, the runtime must not
    overwrite the row with COMPLETED when the agent eventually returns.

    Simulates the race by flipping the row to CANCELLED from inside the
    agent's ainvoke; the runtime's trace step still runs, but the final
    status set should be skipped.
    """
    task = store.create(goal="cancel me")
    store_ref = store

    class _CancelMidwayAgent:
        async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
            # The PATCH endpoint would do this from another request; we
            # simulate the effect by writing the row directly.
            store_ref.set_status(task.id, TaskStatus.CANCELLED)
            return {"messages": [*state["messages"], _FakeAIMessage(content="too late")]}

    runtime = TaskRuntime(
        store,
        agent_factory=_CancelMidwayAgent,
        message_factory=_FakeHumanMessage,
    )

    import asyncio
    result = asyncio.run(runtime.execute(task.id))

    assert result.status == TaskStatus.CANCELLED
    # Trace still landed so the user can see what happened.
    assert any(step.description == "Final response" for step in result.plan)


def test_cancel_during_invoke_error_still_preserves_cancelled(store: TasksStore) -> None:
    """Cancel + agent crash should leave the task cancelled, not failed."""
    task = store.create(goal="cancel and crash")
    store_ref = store

    class _CancelThenCrashAgent:
        async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
            store_ref.set_status(task.id, TaskStatus.CANCELLED)
            raise RuntimeError("model died after cancel")

    runtime = TaskRuntime(
        store,
        agent_factory=_CancelThenCrashAgent,
        message_factory=_FakeHumanMessage,
    )

    import asyncio
    # No raise: the cancel-preservation branch swallows the agent error.
    result = asyncio.run(runtime.execute(task.id))
    assert result.status == TaskStatus.CANCELLED


def test_runtime_caps_steps_at_max(store: TasksStore) -> None:
    """A runaway agent must stop at _MAX_STEPS."""
    from june_brain.tasks import runtime as runtime_module

    task = store.create(goal="loop")
    # 60 alternating call+result pairs.
    trace: list[Any] = []
    for i in range(60):
        cid = f"call_{i}"
        trace.append(
            _FakeAIMessage(
                content="",
                tool_calls=[{"id": cid, "name": "noop", "args": {"i": i}}],
            )
        )
        trace.append(_FakeToolMessage(content=str(i), tool_call_id=cid))

    runtime = _runtime_with_trace(store, trace)
    import asyncio
    result = asyncio.run(runtime.execute(task.id))

    tool_steps = [s for s in result.plan if s.tool_name == "noop"]
    assert len(tool_steps) <= runtime_module._MAX_STEPS
