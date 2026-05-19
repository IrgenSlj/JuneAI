"""Pydantic schemas for the tasks API (ADR 0010, Sprint 1.2)."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskStepView(BaseModel):
    """One step inside a task's plan, as seen by the UI."""

    id: str
    index: int
    description: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    tool_result: Any = None
    status: str
    model_provenance: Optional[dict[str, Any]] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None


class TaskView(BaseModel):
    """A task as returned to the UI."""

    id: str
    user_id: str
    goal: str
    status: str
    plan: list[TaskStepView] = Field(default_factory=list)
    owner_skill: Optional[str] = None
    schedule: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class TaskListResponse(BaseModel):
    tasks: list[TaskView] = Field(default_factory=list)
    count: int = 0


class TaskCreateRequest(BaseModel):
    goal: str = Field(..., min_length=1, description="Free-form natural language goal.")
    owner_skill: Optional[str] = Field(
        default=None,
        description="Optional skill key that should own execution of this task.",
    )
    schedule: Optional[str] = Field(
        default=None,
        description="Optional schedule descriptor (cron-like) for recurring tasks.",
    )


class TaskPatchRequest(BaseModel):
    status: Optional[str] = Field(
        default=None,
        description=(
            "New task status. One of planning, running, paused, awaiting_user, "
            "completed, failed, cancelled."
        ),
    )
    error: Optional[str] = Field(default=None, description="Optional error message for failed status.")


class TaskDeleteResponse(BaseModel):
    deleted: bool
    task_id: str
