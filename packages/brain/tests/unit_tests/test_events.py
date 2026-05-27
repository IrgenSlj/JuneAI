"""Tests for the event ledger (ADR 0014) — store + migration + export."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    from june_brain.events import EventLedger

    return EventLedger()


def test_events_preserve_append_order_newest_first(ledger) -> None:
    from june_brain.operating_layer import EventKind, LedgerEvent, now_iso

    # Identical timestamps would tie on created_at — rowid must still order them.
    ts = now_iso()
    for kind in (EventKind.CAPTURE_RECEIVED, EventKind.CAPTURE_CLASSIFIED, EventKind.MEMORY_WRITTEN):
        ledger.append(LedgerEvent(kind=kind, user_id="u", created_at=ts))

    events = ledger.list_events("u")
    assert [e.kind for e in events] == [
        EventKind.MEMORY_WRITTEN,
        EventKind.CAPTURE_CLASSIFIED,
        EventKind.CAPTURE_RECEIVED,
    ]


def test_events_are_scoped_per_user(ledger) -> None:
    from june_brain.operating_layer import EventKind, LedgerEvent

    ledger.append(LedgerEvent(kind=EventKind.MEMORY_WRITTEN, user_id="alice"))
    ledger.append(LedgerEvent(kind=EventKind.MEMORY_WRITTEN, user_id="bob"))
    assert len(ledger.list_events("alice")) == 1
    assert len(ledger.list_events("bob")) == 1


def test_capture_roundtrips_with_kinds_and_metadata(ledger) -> None:
    from june_brain.operating_layer import CaptureItem, CaptureKind

    item = CaptureItem(
        text="call Sam tomorrow",
        user_id="u",
        kinds=(CaptureKind.TASK, CaptureKind.EVENT),
        metadata={"due": "tomorrow"},
    )
    ledger.save_capture(item)

    (loaded,) = ledger.recent_captures("u")
    assert loaded.text == "call Sam tomorrow"
    assert loaded.kinds == (CaptureKind.TASK, CaptureKind.EVENT)
    assert loaded.metadata == {"due": "tomorrow"}


def test_intent_roundtrips_and_pending_flag_persists(ledger) -> None:
    from june_brain.operating_layer import ActionIntent, ActionKind, ApprovalStatus

    # SEND_MESSAGE always requires approval → __post_init__ sets it PENDING.
    intent = ActionIntent(
        kind=ActionKind.SEND_MESSAGE, title="Text Lisa", summary="…", user_id="u"
    )
    ledger.save_intent(intent)

    loaded = ledger.get_intent(intent.id)
    assert loaded is not None
    assert loaded.kind == ActionKind.SEND_MESSAGE
    assert loaded.approval_status == ApprovalStatus.PENDING
    assert loaded.can_commit is False


def test_migration_v4_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    from june_brain.memory.migration import ensure_schema
    from june_brain.memory.sqlite import _get_connection, db_path

    conn = _get_connection(db_path())
    ensure_schema(conn)
    ensure_schema(conn)  # second run must not raise
    applied = {r[0] for r in conn.execute("SELECT version FROM _schema_migrations")}
    assert 4 in applied


def test_export_includes_ledger_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    from june_brain.events import EventLedger
    from june_brain.memory.export import export_memory
    from june_brain.operating_layer import EventKind, LedgerEvent

    EventLedger().append(LedgerEvent(kind=EventKind.MEMORY_WRITTEN, user_id="u", source="test"))

    dump = export_memory("u")
    events = dump["stores"]["sqlite"].get("events", [])
    assert any(e["source"] == "test" for e in events)
