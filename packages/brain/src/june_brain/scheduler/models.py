"""Data model for scheduled actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class Schedule:
    # Default "" so callers can omit it; ScheduleStore.create() backfills a
    # generated id on insert. _row_to_schedule always passes id explicitly.
    id: str = ""
    user_id: str = ""
    name: str = ""
    description: str = ""
    cron_expression: str = ""
    interval_seconds: int = 0
    scheduled_at: str = ""  # ISO 8601
    last_run_at: str = ""
    action_type: str = "agent_invoke"  # agent_invoke | notification | webhook
    action_config: dict[str, Any] = field(default_factory=dict)
    max_runs: int = 0
    run_count: int = 0
    enabled: bool = True
    created_at: str = ""
    updated_at: str = ""

    def is_due(self, now: datetime | None = None) -> bool:
        """Check if this schedule is due to run."""
        if not self.enabled:
            return False
        if self.max_runs > 0 and self.run_count >= self.max_runs:
            return False
        if not self.scheduled_at:
            return False
        reference = now or datetime.now(UTC)
        scheduled = datetime.fromisoformat(self.scheduled_at)
        return scheduled <= reference


_SCHEDULES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schedules (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    cron_expression TEXT DEFAULT '',
    interval_seconds INTEGER DEFAULT 0,
    scheduled_at TEXT NOT NULL,
    last_run_at TEXT,
    action_type TEXT NOT NULL DEFAULT 'agent_invoke',
    action_config TEXT NOT NULL DEFAULT '{}',
    max_runs INTEGER DEFAULT 0,
    run_count INTEGER DEFAULT 0,
    enabled INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""
