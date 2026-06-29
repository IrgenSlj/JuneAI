"""HTTP routes for the tasks primitive (ADR 0010, Sprint 1.2).

Thin facade over ``june_brain.tasks.TasksStore``. The task runtime that
actually executes plans lives elsewhere and is wired in a later slice; this
module covers CRUD plus a poll-based SSE live-trace endpoint.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from june_brain.activity import ActivityLog
from june_brain.tasks import Task, TasksStore, TaskStatus, execute_task_in_background

from ..schemas import (
    TaskApproveRequest,
    TaskCreateRequest,
    TaskDeleteResponse,
    TaskEventFrame,
    TaskListResponse,
    TaskPatchRequest,
    TaskStepView,
    TaskView,
)

MAX_POLL_SECONDS = 1800  # 30 minutes hard ceiling
_TERMINAL = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}

logger = logging.getLogger(__name__)
router = APIRouter(tags=["tasks"])


def _store_for(user_id: str) -> TasksStore:
    return TasksStore(user_id=user_id)


def _record_promise_approval(task_id: str, goal: str, tool: str) -> None:
    """Log the user's decision to approve a gated tool for a promise.

    Makes the grant a visible record in Trust alongside the request. Best-effort:
    a logging failure must never block the approve-and-retry path.
    """
    try:
        ActivityLog().record(
            kind="approval",
            label=f"approved · {tool}",
            detail={
                "tool": tool,
                "task_id": task_id,
                "goal": goal,
                "surface": "promise",
                "decision": "approved",
            },
        )
    except Exception:  # noqa: BLE001
        logger.debug("promise-approval activity log failed", exc_info=True)
    # Record the user's grant in the tamper-evident Trust Ledger (ADR 0022):
    # actor="user" because this entry is the human's decision, not June's.
    try:
        from june_brain.trust import get_writer

        get_writer().append(
            kind="approval",
            actor="user",
            payload={
                "tool": tool,
                "task_id": task_id,
                "goal": goal,
                "surface": "promise",
                "decision": "approved",
            },
        )
    except Exception:  # noqa: BLE001
        logger.debug("promise-approval ledger append failed", exc_info=True)


def _to_view(task: Task) -> TaskView:
    return TaskView(
        id=task.id,
        user_id=task.user_id,
        goal=task.goal,
        status=task.status.value,
        plan=[
            TaskStepView(
                id=step.id,
                index=step.index,
                description=step.description,
                tool_name=step.tool_name,
                tool_args=step.tool_args,
                tool_result=step.tool_result,
                status=step.status.value,
                model_provenance=step.model_provenance,
                started_at=step.started_at,
                finished_at=step.finished_at,
                error=step.error,
            )
            for step in task.plan
        ],
        owner_skill=task.owner_skill,
        schedule=task.schedule,
        due_at=task.due_at,
        error=task.error,
        blocked_reason=task.blocked_reason,
        next_action=task.next_action,
        blocked_kind=task.blocked_kind,
        final_deliverable=task.final_deliverable,
        approved_tools=list(task.approved_tools),
        created_at=task.created_at,
        updated_at=task.updated_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
    )


def _frame(event: TaskEventFrame) -> str:
    """Serialise a TaskEventFrame as one SSE data frame."""
    return f"data: {event.model_dump_json(exclude_none=True)}\n\n"


async def _poll_task_events(store: TasksStore, task_id: str) -> AsyncIterator[str]:
    """Async generator that polls the store and yields SSE frames.

    Emits:
    - ``{"type":"step", "step": ...}``   for each new step that appears.
    - ``{"type":"status", "status": ...}`` when the task status changes.
    - ``{"type":"done"}``                 when a terminal status is reached.

    Terminates automatically after MAX_POLL_SECONDS regardless of task state.
    """
    seen_step_ids: set[str] = set()
    last_status: str | None = None
    elapsed = 0.0

    while elapsed < MAX_POLL_SECONDS:
        task = store.get(task_id)
        if task is None:
            # Task was deleted while we were watching — just close.
            yield _frame(TaskEventFrame(type="done"))
            return

        # Emit status-change frame.
        if task.status.value != last_status:
            last_status = task.status.value
            yield _frame(TaskEventFrame(type="status", status=last_status))

        # Emit one frame per new step.
        for step in task.plan:
            if step.id not in seen_step_ids:
                seen_step_ids.add(step.id)
                step_view = TaskStepView(
                    id=step.id,
                    index=step.index,
                    description=step.description,
                    tool_name=step.tool_name,
                    tool_args=step.tool_args,
                    tool_result=step.tool_result,
                    status=step.status.value,
                    model_provenance=step.model_provenance,
                    started_at=step.started_at,
                    finished_at=step.finished_at,
                    error=step.error,
                )
                yield _frame(TaskEventFrame(type="step", step=step_view))

        if task.status in _TERMINAL:
            yield _frame(TaskEventFrame(type="done"))
            return

        await asyncio.sleep(0.5)
        elapsed += 0.5

    # Timeout — emit done so the client knows to stop reconnecting.
    yield _frame(TaskEventFrame(type="done"))


@router.get(
    "/tasks/{user_id}/{task_id}/events",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "SSE stream of task step and status events. See TaskEventFrame.",
        }
    },
)
async def task_events(user_id: str, task_id: str) -> StreamingResponse:
    """Subscribe to live step-trace events for a running task.

    Streams TaskEventFrame JSON objects as SSE data frames at ~500 ms
    resolution. Terminates when the task reaches a terminal status or after
    30 minutes of inactivity to prevent resource leaks.
    """
    store = _store_for(user_id)
    task = store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return StreamingResponse(
        _poll_task_events(store, task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/tasks/{user_id}", response_model=TaskListResponse)
def list_tasks(user_id: str, status: str | None = None, limit: int = 50) -> TaskListResponse:
    store = _store_for(user_id)
    status_enum: TaskStatus | None = None
    if status:
        try:
            status_enum = TaskStatus(status.strip().lower())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"unknown status: {status}") from exc
    tasks = store.list(status=status_enum, limit=max(1, min(int(limit), 200)))
    views = [_to_view(t) for t in tasks]
    return TaskListResponse(tasks=views, count=len(views))


@router.get("/tasks/{user_id}/{task_id}", response_model=TaskView)
def get_task(user_id: str, task_id: str) -> TaskView:
    task = _store_for(user_id).get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return _to_view(task)


@router.post("/tasks/{user_id}", response_model=TaskView, status_code=201)
def create_task(user_id: str, payload: TaskCreateRequest) -> TaskView:
    task = _store_for(user_id).create(
        goal=payload.goal,
        owner_skill=payload.owner_skill,
        schedule=payload.schedule,
        due_at=payload.due_at,
    )
    return _to_view(task)


@router.patch("/tasks/{user_id}/{task_id}", response_model=TaskView)
def patch_task(
    user_id: str,
    task_id: str,
    payload: TaskPatchRequest,
    background_tasks: BackgroundTasks,
) -> TaskView:
    store = _store_for(user_id)
    task = store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    # due_at: None leaves it unchanged; "" clears it; a value sets it.
    if payload.due_at is not None:
        task = store.set_due(task_id, payload.due_at)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
    if payload.status is not None:
        try:
            status_enum = TaskStatus(payload.status.strip().lower())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"unknown status: {payload.status}") from exc
        # Idempotent on start: if the task is already running, surface that
        # rather than queueing a second runtime against the same row.
        if status_enum == TaskStatus.RUNNING and task.status == TaskStatus.RUNNING:
            raise HTTPException(status_code=409, detail="task is already running")
        task = store.set_status(task_id, status_enum, error=payload.error)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        # When the user starts a task, kick the runtime in the background so the
        # request returns immediately. The runtime advances the status as it
        # progresses; the UI polls /tasks to see the trace land.
        if status_enum == TaskStatus.RUNNING:
            background_tasks.add_task(execute_task_in_background, store, task_id)
    return _to_view(task)


@router.post("/tasks/{user_id}/{task_id}/run", response_model=TaskView)
def run_task(
    user_id: str,
    task_id: str,
    background_tasks: BackgroundTasks,
) -> TaskView:
    """Explicitly kick the runtime for a task. Idempotent for finished or already-running tasks."""
    store = _store_for(user_id)
    task = store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    if task.status == TaskStatus.RUNNING:
        # Already in flight — return the current view rather than starting a
        # second runtime against the same row.
        return _to_view(task)
    if task.status not in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
        task = store.set_status(task_id, TaskStatus.RUNNING)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        background_tasks.add_task(execute_task_in_background, store, task_id)
    return _to_view(task)


@router.post("/tasks/{user_id}/{task_id}/approve", response_model=TaskView)
def approve_task_tool(
    user_id: str,
    task_id: str,
    payload: TaskApproveRequest,
    background_tasks: BackgroundTasks,
) -> TaskView:
    """Approve one consequential tool for a blocked promise, then re-run it.

    The promise's runtime carries this allow-list into its session, so the
    guard waives the approved action on the retry. Taint-flagged network
    actions still always ask. No-op on an already-running promise.
    """
    store = _store_for(user_id)
    task = store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    task = store.approve_tool(task_id, payload.tool)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    _record_promise_approval(task_id, task.goal, payload.tool)
    if task.status == TaskStatus.RUNNING:
        # Already in flight — record the approval but don't start a second run.
        return _to_view(task)
    task = store.set_status(task_id, TaskStatus.RUNNING)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    background_tasks.add_task(execute_task_in_background, store, task_id)
    return _to_view(task)


@router.delete("/tasks/{user_id}/{task_id}", response_model=TaskDeleteResponse)
def delete_task(user_id: str, task_id: str) -> TaskDeleteResponse:
    deleted = _store_for(user_id).delete(task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="task not found")
    return TaskDeleteResponse(deleted=True, task_id=task_id)
