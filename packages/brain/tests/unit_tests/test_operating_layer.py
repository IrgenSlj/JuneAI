from __future__ import annotations

from june_brain.operating_layer import (
    ActionIntent,
    ActionKind,
    ActionRisk,
    ApprovalStatus,
    CaptureItem,
    CaptureKind,
    EventKind,
    LedgerEvent,
)


def test_capture_item_round_trips_kinds() -> None:
    capture = CaptureItem(
        text="Tomorrow I need to call Sam and finish the deck.",
        user_id="local",
        source="quick_capture",
        kinds=(CaptureKind.TASK, CaptureKind.EVENT),
    )

    restored = CaptureItem.from_dict(capture.to_dict())

    assert restored.text == capture.text
    assert restored.user_id == "local"
    assert restored.source == "quick_capture"
    assert restored.kinds == (CaptureKind.TASK, CaptureKind.EVENT)


def test_low_risk_local_action_can_commit_without_approval() -> None:
    intent = ActionIntent(
        kind=ActionKind.CREATE_TASK,
        title="Call Sam",
        summary="Create a local task from quick capture.",
        risk=ActionRisk.LOW,
    )

    assert intent.requires_approval is False
    assert intent.approval_status == ApprovalStatus.NOT_REQUIRED
    assert intent.can_commit is True


def test_medium_risk_action_defaults_to_pending_approval() -> None:
    intent = ActionIntent(
        kind=ActionKind.SEND_NOTIFICATION,
        title="Reminder",
        summary="Interrupt the user later with a reminder.",
        risk=ActionRisk.MEDIUM,
    )

    assert intent.requires_approval is True
    assert intent.approval_status == ApprovalStatus.PENDING
    assert intent.can_commit is False

    intent.approve()

    assert intent.approval_status == ApprovalStatus.APPROVED
    assert intent.can_commit is True


def test_external_message_always_requires_approval() -> None:
    intent = ActionIntent(
        kind=ActionKind.SEND_MESSAGE,
        title="Send Telegram message",
        summary="Send an outbound message.",
        risk=ActionRisk.LOW,
    )

    assert intent.requires_approval is True
    assert intent.approval_status == ApprovalStatus.PENDING
    assert intent.can_commit is False


def test_rejected_intent_cannot_commit() -> None:
    intent = ActionIntent(
        kind=ActionKind.DELETE_DATA,
        title="Delete memory",
        summary="Delete user data.",
        risk=ActionRisk.HIGH,
    )

    intent.reject()

    assert intent.approval_status == ApprovalStatus.REJECTED
    assert intent.can_commit is False


def test_ledger_event_serializes_event_kind() -> None:
    event = LedgerEvent(
        kind=EventKind.CAPTURE_RECEIVED,
        user_id="local",
        source="quick_capture",
        payload={"text": "remember this"},
    )

    raw = event.to_dict()

    assert raw["kind"] == "capture_received"
    assert raw["user_id"] == "local"
    assert raw["payload"] == {"text": "remember this"}
