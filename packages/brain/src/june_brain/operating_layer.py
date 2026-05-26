"""Shared models for June's personal operating layer.

These types are intentionally side-effect free. They define the vocabulary for
v0.1.1 work: capture, classify, propose, approve, commit, and record.
Database stores, API schemas, and UI types should map to these concepts rather
than inventing parallel names.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def now_iso() -> str:
    """Return a timezone-aware ISO timestamp."""
    return datetime.now(UTC).isoformat()


def new_id() -> str:
    """Return a compact random identifier."""
    return uuid.uuid4().hex


class CaptureKind(StrEnum):
    """Classification labels for user input."""

    TASK = "task"
    EVENT = "event"
    MEMORY = "memory"
    DECISION = "decision"
    PROMISE = "promise"
    FEELING = "feeling"
    IDEA = "idea"
    QUESTION = "question"
    NOTE = "note"


class ActionKind(StrEnum):
    """Types of actions June can propose after understanding a capture."""

    SAVE_MEMORY = "save_memory"
    CREATE_TASK = "create_task"
    CREATE_SCHEDULE = "create_schedule"
    CREATE_CALENDAR_EVENT = "create_calendar_event"
    SEND_NOTIFICATION = "send_notification"
    SEND_MESSAGE = "send_message"
    DELETE_DATA = "delete_data"
    CALL_TOOL = "call_tool"
    CLOUD_CALL = "cloud_call"


class ActionRisk(StrEnum):
    """Risk level for an action intent."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTERNAL = "external"


class ApprovalStatus(StrEnum):
    """Lifecycle state for an action intent that may require consent."""

    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class EventKind(StrEnum):
    """Durable ledger event names."""

    CAPTURE_RECEIVED = "capture_received"
    CAPTURE_CLASSIFIED = "capture_classified"
    ACTION_INTENT_CREATED = "action_intent_created"
    APPROVAL_REQUESTED = "approval_requested"
    ACTION_APPROVED = "action_approved"
    ACTION_REJECTED = "action_rejected"
    ACTION_COMMITTED = "action_committed"
    MEMORY_WRITTEN = "memory_written"
    MEMORY_EDITED = "memory_edited"
    MEMORY_DELETED = "memory_deleted"
    TASK_CREATED = "task_created"
    TASK_COMPLETED = "task_completed"
    SCHEDULE_CREATED = "schedule_created"
    SCHEDULE_FIRED = "schedule_fired"
    NOTIFICATION_SENT = "notification_sent"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"


_ALWAYS_APPROVE_ACTIONS = {
    ActionKind.CREATE_CALENDAR_EVENT,
    ActionKind.SEND_MESSAGE,
    ActionKind.DELETE_DATA,
    ActionKind.CLOUD_CALL,
}


@dataclass
class CaptureItem:
    """Raw user input plus its structured classification labels."""

    text: str
    user_id: str = "default"
    source: str = "chat"
    id: str = field(default_factory=new_id)
    kinds: tuple[CaptureKind, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "source": self.source,
            "text": self.text,
            "kinds": [kind.value for kind in self.kinds],
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CaptureItem:
        return cls(
            id=str(raw.get("id") or new_id()),
            user_id=str(raw.get("user_id") or "default"),
            source=str(raw.get("source") or "chat"),
            text=str(raw.get("text") or ""),
            kinds=tuple(_capture_kind_or(kind, CaptureKind.NOTE) for kind in raw.get("kinds") or ()),
            metadata=dict(raw.get("metadata") or {}),
            created_at=str(raw.get("created_at") or now_iso()),
        )


@dataclass
class ActionIntent:
    """A proposed write or side effect that June may commit."""

    kind: ActionKind
    title: str
    summary: str
    user_id: str = "default"
    risk: ActionRisk = ActionRisk.LOW
    id: str = field(default_factory=new_id)
    source_capture_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    approval_status: ApprovalStatus = ApprovalStatus.NOT_REQUIRED
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def __post_init__(self) -> None:
        if self.requires_approval and self.approval_status == ApprovalStatus.NOT_REQUIRED:
            self.approval_status = ApprovalStatus.PENDING

    @property
    def requires_approval(self) -> bool:
        if self.kind in _ALWAYS_APPROVE_ACTIONS:
            return True
        return self.risk in {ActionRisk.MEDIUM, ActionRisk.HIGH, ActionRisk.EXTERNAL}

    @property
    def can_commit(self) -> bool:
        if not self.requires_approval:
            return True
        return self.approval_status == ApprovalStatus.APPROVED

    def approve(self) -> None:
        self.approval_status = ApprovalStatus.APPROVED
        self.updated_at = now_iso()

    def reject(self) -> None:
        self.approval_status = ApprovalStatus.REJECTED
        self.updated_at = now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "kind": self.kind.value,
            "title": self.title,
            "summary": self.summary,
            "risk": self.risk.value,
            "source_capture_id": self.source_capture_id,
            "payload": self.payload,
            "requires_approval": self.requires_approval,
            "approval_status": self.approval_status.value,
            "can_commit": self.can_commit,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ActionIntent:
        return cls(
            id=str(raw.get("id") or new_id()),
            user_id=str(raw.get("user_id") or "default"),
            kind=_action_kind_or(raw.get("kind"), ActionKind.SAVE_MEMORY),
            title=str(raw.get("title") or ""),
            summary=str(raw.get("summary") or ""),
            risk=_action_risk_or(raw.get("risk"), ActionRisk.LOW),
            source_capture_id=raw.get("source_capture_id"),
            payload=dict(raw.get("payload") or {}),
            approval_status=_approval_status_or(
                raw.get("approval_status"),
                ApprovalStatus.NOT_REQUIRED,
            ),
            created_at=str(raw.get("created_at") or now_iso()),
            updated_at=str(raw.get("updated_at") or now_iso()),
        )


@dataclass
class LedgerEvent:
    """Append-only event for June's durable product record."""

    kind: EventKind
    user_id: str = "default"
    source: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "kind": self.kind.value,
            "source": self.source,
            "payload": self.payload,
            "created_at": self.created_at,
        }


def _capture_kind_or(value: Any, fallback: CaptureKind) -> CaptureKind:
    try:
        return CaptureKind(str(value))
    except ValueError:
        return fallback


def _action_kind_or(value: Any, fallback: ActionKind) -> ActionKind:
    try:
        return ActionKind(str(value))
    except ValueError:
        return fallback


def _action_risk_or(value: Any, fallback: ActionRisk) -> ActionRisk:
    try:
        return ActionRisk(str(value))
    except ValueError:
        return fallback


def _approval_status_or(value: Any, fallback: ApprovalStatus) -> ApprovalStatus:
    try:
        return ApprovalStatus(str(value))
    except ValueError:
        return fallback
