"""Export June's full memory to a portable JSON archive.

Usage::

    python -m june_brain.memory.export --user <user_id> --output path/to/export.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .sqlite import _get_connection, db_path

# Tables whose rows are tied to a user_id and should be filtered on export.
_USER_TABLES = frozenset({
    "chat_messages", "moods", "journal", "relationships", "goals",
    "open_loops", "preferences", "favorites", "calendar_items",
    "gym_plans", "food_programs", "workout_sessions", "body_metrics",
    "habits", "habit_completions", "nutrition_logs", "water_logs",
    "telemetry", "app_state", "memory_feedback", "semantic_facts",
    "graph_nodes", "graph_edges", "schedules", "skill_inbound_events",
})

_SKIP_TABLES = frozenset({
    "chat_messages",  # transient
    "_schema_migrations",
})

# Device-global audit tables have no user_id and are exported in full (not
# user-filtered): notably ``trust_ledger`` (ADR 0022), so the tamper-evident
# chain is verifiable offline from the export. They fall through to the
# unfiltered ``SELECT *`` branch below; do not add them to ``_SKIP_TABLES``.


def _export_sqlite(db_path: str, user_id: str) -> dict[str, list[dict[str, Any]]]:
    """Export per-user tables, filtered by user_id where possible."""
    conn = _get_connection(db_path)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    out: dict[str, list[dict[str, Any]]] = {}
    for (table_name,) in tables:
        if table_name in _SKIP_TABLES:
            continue
        if table_name in _USER_TABLES:
            try:
                rows = conn.execute(
                    f"SELECT * FROM [{table_name}] WHERE user_id = ?", (user_id,)
                ).fetchall()
            except sqlite3.OperationalError:
                rows = conn.execute(f"SELECT * FROM [{table_name}]").fetchall()
        else:
            rows = conn.execute(f"SELECT * FROM [{table_name}]").fetchall()
        out[table_name] = [dict(r) for r in rows]
    return out


def export_memory(user_id: str, output_path: str | Path | None = None) -> dict[str, Any]:
    """Export all memory stores to a portable dict.

    If ``output_path`` is provided, writes JSON and returns the dict.
    Otherwise returns the dict in memory.
    """
    sqlite_path = db_path()

    archive: dict[str, Any] = {
        "version": 1,
        "exported_at": datetime.now(UTC).isoformat(),
        "user_id": user_id,
        "stores": {
            "sqlite": _export_sqlite(sqlite_path, user_id),
        },
    }

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(archive, indent=2, default=str), encoding="utf-8")

    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export June memory to JSON")
    parser.add_argument("--user", required=True, help="User ID to export")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()
    export_memory(args.user, args.output)
