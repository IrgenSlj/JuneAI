"""SQLite CRUD for :class:`Schedule` records."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from .models import Schedule

_SCHEDULE_COLS = (
    "id", "user_id", "name", "description", "cron_expression",
    "interval_seconds", "scheduled_at", "last_run_at", "action_type",
    "action_config", "max_runs", "run_count", "enabled",
    "created_at", "updated_at",
)


def _row_to_schedule(row: sqlite3.Row) -> Schedule:
    d = dict(row)
    action_config_raw = d.pop("action_config", "{}")
    return Schedule(
        id=d["id"],
        user_id=d["user_id"],
        name=d["name"],
        description=d.get("description", ""),
        cron_expression=d.get("cron_expression", ""),
        interval_seconds=d.get("interval_seconds", 0),
        scheduled_at=d.get("scheduled_at", ""),
        last_run_at=d.get("last_run_at", ""),
        action_type=d.get("action_type", "agent_invoke"),
        action_config=json.loads(action_config_raw) if isinstance(action_config_raw, str) else {},
        max_runs=d.get("max_runs", 0),
        run_count=d.get("run_count", 0),
        enabled=bool(d.get("enabled", 1)),
        created_at=d.get("created_at", ""),
        updated_at=d.get("updated_at", ""),
    )


class ScheduleStore:
    """CRUD operations on the ``schedules`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create(self, schedule: Schedule) -> Schedule:
        """Insert a new schedule (auto-generates id and timestamps if missing)."""
        if not schedule.id:
            schedule.id = uuid.uuid4().hex[:12]
        now = self._now()
        if not schedule.created_at:
            schedule.created_at = now
        schedule.updated_at = now
        if not schedule.scheduled_at:
            schedule.scheduled_at = now

        self._conn.execute(
            f"INSERT INTO schedules ({', '.join(_SCHEDULE_COLS)}) "
            f"VALUES ({', '.join('?' for _ in _SCHEDULE_COLS)})",
            (
                schedule.id, schedule.user_id, schedule.name, schedule.description,
                schedule.cron_expression, schedule.interval_seconds,
                schedule.scheduled_at, schedule.last_run_at,
                schedule.action_type, json.dumps(schedule.action_config),
                schedule.max_runs, schedule.run_count, int(schedule.enabled),
                schedule.created_at, schedule.updated_at,
            ),
        )
        self._conn.commit()
        return schedule

    def get(self, schedule_id: str) -> Schedule | None:
        row = self._conn.execute(
            "SELECT * FROM schedules WHERE id = ?", (schedule_id,)
        ).fetchone()
        return _row_to_schedule(row) if row else None

    def list(self, user_id: str | None = None) -> list[Schedule]:
        if user_id:
            rows = self._conn.execute(
                "SELECT * FROM schedules WHERE user_id = ? ORDER BY scheduled_at",
                (user_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM schedules ORDER BY scheduled_at"
            ).fetchall()
        return [_row_to_schedule(r) for r in rows]

    def list_due(self, user_id: str | None = None) -> list[Schedule]:
        """Return schedules that are enabled and past their ``scheduled_at``."""
        from .models import Schedule as _S

        all_schedules = self.list(user_id)
        return [s for s in all_schedules if s.is_due()]

    def update(self, schedule: Schedule) -> Schedule:
        schedule.updated_at = self._now()
        self._conn.execute(
            """UPDATE schedules SET
                name=?, description=?, cron_expression=?, interval_seconds=?,
                scheduled_at=?, last_run_at=?, action_type=?, action_config=?,
                max_runs=?, run_count=?, enabled=?, updated_at=?
             WHERE id = ?""",
            (
                schedule.name, schedule.description, schedule.cron_expression,
                schedule.interval_seconds, schedule.scheduled_at,
                schedule.last_run_at, schedule.action_type,
                json.dumps(schedule.action_config), schedule.max_runs,
                schedule.run_count, int(schedule.enabled), schedule.updated_at,
                schedule.id,
            ),
        )
        self._conn.commit()
        return schedule

    def delete(self, schedule_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def mark_run(self, schedule: Schedule) -> Schedule:
        """Increment run_count and update scheduled_at for next run."""
        from .service import compute_next_run

        now = self._now()
        schedule.last_run_at = now
        schedule.run_count += 1
        schedule.scheduled_at = compute_next_run(schedule) or ""
        schedule.updated_at = now
        return self.update(schedule)
