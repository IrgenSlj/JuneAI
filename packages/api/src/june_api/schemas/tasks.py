"""Pydantic schemas for the tasks API (ADR 0010, Sprint 1.2)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TaskStepView(BaseModel):
    """One step inside a task's plan, as seen by the UI."""

    id: str
    index: int
    description: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any = None
    status: str
    model_provenance: dict[str, Any] | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


class TaskView(BaseModel):
    """A task as returned to the UI."""

    id: str
    user_id: str
    goal: str
    status: str
    plan: list[TaskStepView] = Field(default_factory=list)
    owner_skill: str | None = None
    schedule: str | None = None
    error: str | None = None
    blocked_reason: str | None = Field(
        default=None,
        description="Why this promise is waiting on the user, if blocked.",
    )
    next_action: str | None = Field(
        default=None,
        description="Plain-language action the user can take to unblock the promise.",
    )
    final_deliverable: str | None = Field(
        default=None,
        description="Final assistant-facing deliverable captured when the promise finishes.",
    )
    blocked_kind: str | None = Field(
        default=None,
        description=(
            "How the promise is blocked while awaiting the user: 'approval' (a "
            "consequential action the guard gated) or 'local_only' (a networked "
            "tool the privacy dial forbids). Lets the UI pick the right unblock "
            "control without parsing the reason text."
        ),
    )
    approved_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Tools the user has approved for this promise. A retry runs these "
            "without re-asking; taint-flagged network actions still always ask."
        ),
    )
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None


class TaskListResponse(BaseModel):
    tasks: list[TaskView] = Field(default_factory=list)
    count: int = 0


class TaskCreateRequest(BaseModel):
    goal: str = Field(..., min_length=1, description="Free-form natural language goal.")
    owner_skill: str | None = Field(
        default=None,
        description="Optional skill key that should own execution of this task.",
    )
    schedule: str | None = Field(
        default=None,
        description="Optional schedule descriptor (cron-like) for recurring tasks.",
    )


class TaskPatchRequest(BaseModel):
    status: str | None = Field(
        default=None,
        description=(
            "New task status. One of planning, running, paused, awaiting_user, "
            "completed, failed, cancelled."
        ),
    )
    error: str | None = Field(default=None, description="Optional error message for failed status.")


class TaskApproveRequest(BaseModel):
    tool: str = Field(
        ...,
        min_length=1,
        description=(
            "Name of the tool to approve for this promise. The promise is then "
            "re-run with this tool on its allow-list."
        ),
    )


class TaskDeleteResponse(BaseModel):
    deleted: bool
    task_id: str


class TaskEventFrame(BaseModel):
    """One SSE frame emitted by GET /tasks/{user_id}/{task_id}/events.

    ``type`` is one of:
    - ``"step"``   — a new step appeared in the plan; ``step`` is populated.
    - ``"status"`` — the task's top-level status changed; ``status`` is populated.
    - ``"done"``   — the task reached a terminal state; the stream will close.
    """

    type: str
    step: TaskStepView | None = None
    status: str | None = None
