"""Data model for tasks and the steps that compose them."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


class TaskStatus(str, Enum):
    """Lifecycle states for a task. See ADR 0010."""

    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    AWAITING_USER = "awaiting_user"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStepStatus(str, Enum):
    """Lifecycle states for one step inside a task's trace."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskStep:
    """One unit of work within a task's plan.

    ``model_provenance`` is a free-form dict so this module does not import
    the routing module; the runtime that records steps owns the serialisation
    of ``ModelProvenance`` into this field.
    """

    id: str = field(default_factory=_new_id)
    index: int = 0
    description: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    tool_result: Any = None
    status: TaskStepStatus = TaskStepStatus.PENDING
    model_provenance: Optional[dict[str, Any]] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "index": self.index,
            "description": self.description,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result,
            "status": self.status.value,
            "model_provenance": self.model_provenance,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TaskStep":
        return cls(
            id=str(raw.get("id") or _new_id()),
            index=int(raw.get("index", 0)),
            description=str(raw.get("description") or ""),
            tool_name=raw.get("tool_name"),
            tool_args=raw.get("tool_args"),
            tool_result=raw.get("tool_result"),
            status=_status_or(TaskStepStatus, raw.get("status"), TaskStepStatus.PENDING),
            model_provenance=raw.get("model_provenance"),
            started_at=raw.get("started_at"),
            finished_at=raw.get("finished_at"),
            error=raw.get("error"),
        )


@dataclass
class Task:
    """A persistable unit of agentic work."""

    id: str = field(default_factory=_new_id)
    user_id: str = "default"
    goal: str = ""
    status: TaskStatus = TaskStatus.PLANNING
    plan: list[TaskStep] = field(default_factory=list)
    owner_skill: Optional[str] = None
    schedule: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "goal": self.goal,
            "status": self.status.value,
            "plan": [step.to_dict() for step in self.plan],
            "owner_skill": self.owner_skill,
            "schedule": self.schedule,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Task":
        return cls(
            id=str(raw.get("id") or _new_id()),
            user_id=str(raw.get("user_id") or "default"),
            goal=str(raw.get("goal") or ""),
            status=_status_or(TaskStatus, raw.get("status"), TaskStatus.PLANNING),
            plan=[TaskStep.from_dict(step) for step in (raw.get("plan") or [])],
            owner_skill=raw.get("owner_skill"),
            schedule=raw.get("schedule"),
            error=raw.get("error"),
            created_at=str(raw.get("created_at") or _now()),
            updated_at=str(raw.get("updated_at") or _now()),
            started_at=raw.get("started_at"),
            finished_at=raw.get("finished_at"),
        )


def _status_or(enum_cls, value, fallback):  # type: ignore[no-untyped-def]
    if value is None:
        return fallback
    try:
        return enum_cls(value)
    except ValueError:
        return fallback
