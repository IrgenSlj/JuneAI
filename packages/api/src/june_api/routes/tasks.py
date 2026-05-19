"""HTTP routes for the tasks primitive (ADR 0010, Sprint 1.2).

Thin facade over ``june_brain.tasks.TasksStore``. The task runtime that
actually executes plans lives elsewhere and is wired in a later slice; this
module is CRUD only. SSE live-trace is deferred until the runtime exists.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from june_brain.tasks import Task, TaskStatus, TasksStore, execute_task_in_background

from ..schemas import (
    TaskCreateRequest,
    TaskDeleteResponse,
    TaskListResponse,
    TaskPatchRequest,
    TaskStepView,
    TaskView,
)

router = APIRouter(tags=["tasks"])


def _store_for(user_id: str) -> TasksStore:
    return TasksStore(user_id=user_id)


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
        error=task.error,
        created_at=task.created_at,
        updated_at=task.updated_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
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


@router.delete("/tasks/{user_id}/{task_id}", response_model=TaskDeleteResponse)
def delete_task(user_id: str, task_id: str) -> TaskDeleteResponse:
    deleted = _store_for(user_id).delete(task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="task not found")
    return TaskDeleteResponse(deleted=True, task_id=task_id)
