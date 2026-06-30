"""Data model for tasks and the steps that compose them."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


MAX_TASK_ATTEMPTS = 5
"""A single promise is retired as terminal FAILED after this many failed/blocked-retryable runs."""


class TaskStatus(StrEnum):
    """Lifecycle states for a task. See ADR 0010."""

    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    AWAITING_USER = "awaiting_user"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStepStatus(StrEnum):
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
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any = None
    status: TaskStepStatus = TaskStepStatus.PENDING
    model_provenance: dict[str, Any] | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

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
    def from_dict(cls, raw: dict[str, Any]) -> TaskStep:
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
    owner_skill: str | None = None
    schedule: str | None = None
    # Explicit deadline (ISO 8601). Due state is derived at read time, never by a
    # background timer (ADR 0016: no heartbeat). None means no deadline.
    due_at: str | None = None
    error: str | None = None
    blocked_reason: str | None = None
    next_action: str | None = None
    final_deliverable: str | None = None
    # How this promise is blocked, so the UI shows the right unblock control
    # without parsing reason text: "approval" (approve one action) or
    # "local_only" (change the privacy dial). None when not blocked.
    blocked_kind: str | None = None
    # Tools the user approved for this promise (the guard's per-promise
    # allow-list, ADR 0021 S6.2). Lets a retry run a previously approval-gated
    # action without re-blocking; taint-flagged network actions still ask.
    approved_tools: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    is_recurring: bool = False
    recurrence_rule: str = ""
    parent_task_id: str | None = None
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "goal": self.goal,
            "status": self.status.value,
            "plan": [step.to_dict() for step in self.plan],
            "owner_skill": self.owner_skill,
            "schedule": self.schedule,
            "due_at": self.due_at,
            "error": self.error,
            "blocked_reason": self.blocked_reason,
            "next_action": self.next_action,
            "final_deliverable": self.final_deliverable,
            "blocked_kind": self.blocked_kind,
            "approved_tools": list(self.approved_tools),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "is_recurring": self.is_recurring,
            "recurrence_rule": self.recurrence_rule,
            "parent_task_id": self.parent_task_id,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Task:
        return cls(
            id=str(raw.get("id") or _new_id()),
            user_id=str(raw.get("user_id") or "default"),
            goal=str(raw.get("goal") or ""),
            status=_status_or(TaskStatus, raw.get("status"), TaskStatus.PLANNING),
            plan=[TaskStep.from_dict(step) for step in (raw.get("plan") or [])],
            owner_skill=raw.get("owner_skill"),
            schedule=raw.get("schedule"),
            due_at=raw.get("due_at"),
            error=raw.get("error"),
            blocked_reason=raw.get("blocked_reason"),
            next_action=raw.get("next_action"),
            final_deliverable=raw.get("final_deliverable"),
            blocked_kind=raw.get("blocked_kind"),
            approved_tools=list(raw.get("approved_tools") or []),
            created_at=str(raw.get("created_at") or _now()),
            updated_at=str(raw.get("updated_at") or _now()),
            started_at=raw.get("started_at"),
            finished_at=raw.get("finished_at"),
            is_recurring=bool(raw.get("is_recurring", False)),
            recurrence_rule=str(raw.get("recurrence_rule", "")),
            parent_task_id=raw.get("parent_task_id"),
            attempts=int(raw.get("attempts") or 0),
        )


def _status_or(enum_cls, value, fallback):  # type: ignore[no-untyped-def]
    if value is None:
        return fallback
    try:
        return enum_cls(value)
    except ValueError:
        return fallback
