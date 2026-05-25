"""Import a previously exported JSON archive back into June's memory.

Usage::

    python -m june_brain.memory.import_ --user <user_id> --input path/to/export.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from .export import _USER_TABLES
from .sqlite import _current_memory_dir, _get_connection

_IMPORT_SKIP_TABLES = frozenset({
    "chat_messages", "_schema_migrations",
})


def import_memory(input_path: str | Path, user_id: str) -> dict[str, int]:
    """Import a JSON archive into the user's database.

    Returns ``{table_name: row_count}`` for every imported table.
    Existing rows are **not** overwritten — inserts use INSERT OR IGNORE
    so primary-key collisions preserve existing data.
    User-owned rows are rewritten to the target user_id on import.
    """
    archive = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if archive.get("version") != 1:
        raise ValueError(f"Unsupported archive version: {archive.get('version')}")

    db_dir = Path(_current_memory_dir())
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(db_dir / "june.db")
    conn = _get_connection(db_path)

    # Ensure all tables exist before importing
    from .sqlite import _SCHEMA_SQL as schema_sql

    for stmt in schema_sql.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()

    sqlite_data: dict[str, list[dict[str, Any]]] = archive.get("stores", {}).get("sqlite", {})
    counts: dict[str, int] = {}

    for table_name, rows in sqlite_data.items():
        if table_name in _IMPORT_SKIP_TABLES:
            continue
        if not rows:
            counts[table_name] = 0
            continue

        is_user_table = table_name in _USER_TABLES

        columns = list(rows[0].keys())
        placeholders = ", ".join("?" for _ in columns)
        col_names = ", ".join(f"[{c}]" for c in columns)

        imported = 0
        for row in rows:
            # Reassign user-owned rows to the target user
            if is_user_table and "user_id" in row:
                row = dict(row)
                row["user_id"] = user_id
            values = [row.get(c) for c in columns]
            try:
                conn.execute(
                    f"INSERT OR IGNORE INTO [{table_name}] ({col_names}) VALUES ({placeholders})",
                    values,
                )
                if conn.total_changes > 0:
                    imported += 1
            except sqlite3.OperationalError:
                pass  # skip rows that don't fit (schema mismatch)

        conn.commit()
        counts[table_name] = imported

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import June memory from JSON")
    parser.add_argument("--user", required=True, help="User ID to import into")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    args = parser.parse_args()
    counts = import_memory(args.input, args.user)
    total = sum(counts.values())
    print(f"Imported {total} rows across {len(counts)} tables: {counts}")
