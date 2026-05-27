"""Durable event ledger and capture/intent persistence (ADR 0014).

The ledger is the spine of the personal operating layer: every capture,
classification, proposed action, and commit is recorded so memory, tasks,
reviews, and debugging all read from one durable record. Events are
append-only; insertion order is preserved by SQLite's rowid.

Persists the side-effect-free models from ``operating_layer`` — this module
owns the SQLite mapping, those own the vocabulary.
"""

from __future__ import annotations

import json
import sqlite3

from .memory.migration import ensure_schema
from .memory.sqlite import _get_connection, db_path
from .operating_layer import ActionIntent, CaptureItem, EventKind, LedgerEvent


def _connect() -> sqlite3.Connection:
    conn = _get_connection(db_path())
    ensure_schema(conn)
    return conn


def _row_to_event(row: dict) -> LedgerEvent:
    return LedgerEvent(
        id=row["id"],
        user_id=row["user_id"],
        kind=EventKind(row["kind"]),
        source=row.get("source", ""),
        payload=json.loads(row.get("payload") or "{}"),
        created_at=row.get("created_at", ""),
    )


def _row_to_capture(row: dict) -> CaptureItem:
    return CaptureItem.from_dict(
        {
            "id": row["id"],
            "user_id": row["user_id"],
            "source": row.get("source", "chat"),
            "text": row.get("text", ""),
            "kinds": json.loads(row.get("kinds") or "[]"),
            "metadata": json.loads(row.get("metadata") or "{}"),
            "created_at": row.get("created_at", ""),
        }
    )


def _row_to_intent(row: dict) -> ActionIntent:
    return ActionIntent.from_dict(
        {
            "id": row["id"],
            "user_id": row["user_id"],
            "kind": row["kind"],
            "title": row.get("title", ""),
            "summary": row.get("summary", ""),
            "risk": row.get("risk", "low"),
            "source_capture_id": row.get("source_capture_id"),
            "payload": json.loads(row.get("payload") or "{}"),
            "approval_status": row.get("approval_status", "not_required"),
            "created_at": row.get("created_at", ""),
            "updated_at": row.get("updated_at", ""),
        }
    )


class EventLedger:
    """CRUD over the events / capture_items / action_intents tables."""

    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self._conn = conn or _connect()

    # -- events (append-only) ------------------------------------------------
    def append(self, event: LedgerEvent) -> LedgerEvent:
        self._conn.execute(
            "INSERT INTO events (id, user_id, kind, source, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                event.id,
                event.user_id,
                event.kind.value,
                event.source,
                json.dumps(event.payload),
                event.created_at,
            ),
        )
        self._conn.commit()
        return event

    def list_events(self, user_id: str, limit: int = 100) -> list[LedgerEvent]:
        """Return this user's events, newest first (stable via rowid)."""
        rows = self._conn.execute(
            "SELECT * FROM events WHERE user_id = ? ORDER BY rowid DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [_row_to_event(dict(r)) for r in rows]

    # -- captures ------------------------------------------------------------
    def save_capture(self, item: CaptureItem) -> CaptureItem:
        self._conn.execute(
            "INSERT OR REPLACE INTO capture_items "
            "(id, user_id, source, text, kinds, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                item.id,
                item.user_id,
                item.source,
                item.text,
                json.dumps([k.value for k in item.kinds]),
                json.dumps(item.metadata),
                item.created_at,
            ),
        )
        self._conn.commit()
        return item

    def recent_captures(self, user_id: str, limit: int = 20) -> list[CaptureItem]:
        rows = self._conn.execute(
            "SELECT * FROM capture_items WHERE user_id = ? ORDER BY rowid DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [_row_to_capture(dict(r)) for r in rows]

    # -- action intents ------------------------------------------------------
    def save_intent(self, intent: ActionIntent) -> ActionIntent:
        self._conn.execute(
            "INSERT OR REPLACE INTO action_intents "
            "(id, user_id, kind, title, summary, risk, source_capture_id, "
            " payload, approval_status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                intent.id,
                intent.user_id,
                intent.kind.value,
                intent.title,
                intent.summary,
                intent.risk.value,
                intent.source_capture_id,
                json.dumps(intent.payload),
                intent.approval_status.value,
                intent.created_at,
                intent.updated_at,
            ),
        )
        self._conn.commit()
        return intent

    def get_intent(self, intent_id: str) -> ActionIntent | None:
        row = self._conn.execute(
            "SELECT * FROM action_intents WHERE id = ?", (intent_id,)
        ).fetchone()
        return _row_to_intent(dict(row)) if row else None
