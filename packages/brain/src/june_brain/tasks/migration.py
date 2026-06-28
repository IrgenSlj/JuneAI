"""Embedded migration for the tasks table.

The tasks table is managed by ``TasksStore`` (not the main ``Memory``
schema), so additive task columns need their own migration path.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

_TASK_COLS = {
    "is_recurring": "INTEGER DEFAULT 0",
    "recurrence_rule": "TEXT DEFAULT ''",
    "parent_task_id": "TEXT",
    "blocked_reason": "TEXT",
    "next_action": "TEXT",
    "final_deliverable": "TEXT",
    "approved_tools": "TEXT DEFAULT '[]'",
}


def ensure_tasks_migration(conn: sqlite3.Connection) -> None:
    """Add task columns if they don't exist (idempotent)."""
    existing = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(tasks)").fetchall()
    }
    for col, col_type in _TASK_COLS.items():
        if col not in existing:
            logger.info("Migrating tasks table: adding column %s", col)
            conn.execute(f"ALTER TABLE tasks ADD COLUMN {col} {col_type}")
    conn.commit()
