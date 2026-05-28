"""Activity log — the trust primitive.

A mainstream user accepts agency from June only if they can see what June
just did. The activity log is the audit trail: every API request and every
tool invocation is recorded with timestamp, latency, status, and a short
label. The /system/activity endpoint serves it back; the /system page renders
it as a reverse-chronological list.

Stored in the same ``june.db`` as memory and tasks via a rolling table that
caps at ``_MAX_ROWS`` entries. Oldest rows are pruned on each write so the
table never grows unbounded.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .memory.sqlite import _get_connection, db_path

_MAX_ROWS = 1000
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    status INTEGER,
    latency_ms INTEGER,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_activity_time ON activity_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_activity_kind ON activity_log(kind, timestamp DESC);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ActivityEntry:
    """One row in the activity log."""

    id: int = 0
    timestamp: str = field(default_factory=_now)
    kind: str = "request"  # "request" | "tool" | "task" | "skill"
    label: str = ""
    status: int | None = None
    latency_ms: int | None = None
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "detail": self.detail,
        }


class ActivityLog:
    """Singleton write/read interface to the activity_log table."""

    _instance: ActivityLog | None = None
    _lock = threading.Lock()

    def __new__(cls) -> ActivityLog:
        # One instance per process. The connection pool below is shared with
        # memory, so this is just a tidy entry point.
        #
        # All initialization happens inside __new__ under the lock so that
        # concurrent threads never race on the schema DDL. __init__ is a
        # no-op because everything is already set up by the time the lock
        # is released.
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._db_path = db_path()
                conn = _get_connection(instance._db_path)
                conn.executescript(_SCHEMA_SQL)
                conn.commit()
                instance._initialised = True
                cls._instance = instance
            return cls._instance

    def __init__(self) -> None:
        pass

    @property
    def _conn(self):  # type: ignore[no-untyped-def]
        return _get_connection(self._db_path)

    def record(
        self,
        *,
        kind: str,
        label: str,
        status: int | None = None,
        latency_ms: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> ActivityEntry:
        timestamp = _now()
        detail_json = json.dumps(detail) if detail else None
        cur = self._conn.execute(
            "INSERT INTO activity_log (timestamp, kind, label, status, latency_ms, detail) "
            "VALUES (?,?,?,?,?,?)",
            (timestamp, kind, label, status, latency_ms, detail_json),
        )
        self._conn.commit()
        # Trim oldest if we exceeded the cap. Cheap on a 1000-row table.
        self._conn.execute(
            "DELETE FROM activity_log WHERE id NOT IN ("
            "SELECT id FROM activity_log ORDER BY id DESC LIMIT ?)",
            (_MAX_ROWS,),
        )
        self._conn.commit()
        return ActivityEntry(
            id=int(cur.lastrowid or 0),
            timestamp=timestamp,
            kind=kind,
            label=label,
            status=status,
            latency_ms=latency_ms,
            detail=detail,
        )

    def list(
        self,
        *,
        kind: str | None = None,
        limit: int = 100,
    ) -> list[ActivityEntry]:
        limit = max(1, min(int(limit), _MAX_ROWS))
        if kind:
            rows = self._conn.execute(
                "SELECT * FROM activity_log WHERE kind=? ORDER BY id DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM activity_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_entry(row) for row in rows]

    def clear(self) -> int:
        cur = self._conn.execute("DELETE FROM activity_log")
        self._conn.commit()
        return cur.rowcount


def _row_to_entry(row) -> ActivityEntry:  # type: ignore[no-untyped-def]
    detail_raw = row["detail"]
    detail = None
    if detail_raw:
        try:
            detail = json.loads(detail_raw)
        except (TypeError, ValueError):
            detail = None
    return ActivityEntry(
        id=int(row["id"]),
        timestamp=row["timestamp"],
        kind=row["kind"],
        label=row["label"],
        status=row["status"],
        latency_ms=row["latency_ms"],
        detail=detail,
    )


def reset_for_tests() -> None:
    """Drop the singleton so test fixtures that swap MEMORY_DIR get a fresh DB path."""
    with ActivityLog._lock:
        ActivityLog._instance = None
