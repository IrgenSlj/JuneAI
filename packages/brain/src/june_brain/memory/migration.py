"""Embedded schema migration system for June's SQLite store.

Each migration is a function that receives a ``sqlite3.Connection`` and
returns nothing (or raises on failure). Migrations are applied in version
order. A ``_schema_migrations`` table tracks which versions have run.

Usage::

    from .migration import MIGRATIONS, ensure_schema

    conn = _get_connection(db_path)
    MIGRATIONS.ensure(conn)  # runs any pending migrations
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS _schema_migrations (
    version   INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class MigrationRegistry:
    """Ordered collection of schema migrations."""

    def __init__(self) -> None:
        self._migrations: dict[int, tuple[str, Any]] = {}

    def register(self, version: int, description: str) -> Any:
        """Decorator to register a migration function."""

        def _wrap(fn: Any) -> Any:
            self._migrations[version] = (description, fn)
            return fn

        return _wrap

    @property
    def latest_version(self) -> int:
        return max(self._migrations, default=0)

    def ensure(self, conn: Any) -> None:
        """Run all pending migrations against the given connection."""
        conn.executescript(_MIGRATIONS_TABLE_SQL)

        applied = {
            row["version"]
            for row in conn.execute("SELECT version FROM _schema_migrations").fetchall()
        }

        for version in sorted(self._migrations):
            if version in applied:
                continue
            description, fn = self._migrations[version]
            logger.info("Schema migration %d: %s …", version, description)
            try:
                fn(conn)
                conn.execute(
                    "INSERT OR IGNORE INTO _schema_migrations (version) VALUES (?)",
                    (version,),
                )
                conn.commit()
                logger.info("Schema migration %d applied.", version)
            except Exception:
                logger.exception("Schema migration %d failed — database may be inconsistent", version)
                raise


MIGRATIONS = MigrationRegistry()


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


@MIGRATIONS.register(1, "Initial schema — all existing tables")
def _migration_001(conn: Any) -> None:
    """Create all domain tables. Idempotent via IF NOT EXISTS."""
    from . import sqlite as _sqlite  # lazy import to break circular dependency

    conn.executescript(_sqlite._SCHEMA_SQL)


@MIGRATIONS.register(2, "Add schedules + skill_inbound_events tables")
def _migration_002(conn: Any) -> None:
    """Create tables for personal assistant framework."""
    conn.executescript("""
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
        CREATE TABLE IF NOT EXISTS skill_inbound_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            skill_key TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload TEXT NOT NULL,
            received_at TEXT NOT NULL,
            processed INTEGER DEFAULT 0,
            agent_invoked INTEGER DEFAULT 0
        );
    """)


@MIGRATIONS.register(3, "Add shopping + chores tables")
def _migration_003(conn: Any) -> None:
    """Create tables for shopping and chores domain memory."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            preferred_price REAL,
            preferred_store TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            url TEXT DEFAULT '',
            date_added TEXT NOT NULL,
            active INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS purchase_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            product_id INTEGER NOT NULL REFERENCES products(id),
            price REAL,
            store TEXT DEFAULT '',
            date TEXT NOT NULL,
            notes TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS price_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            product_id INTEGER NOT NULL REFERENCES products(id),
            target_price REAL NOT NULL,
            active INTEGER DEFAULT 1,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            interval_days INTEGER NOT NULL DEFAULT 7,
            last_done TEXT,
            next_due TEXT,
            notes TEXT DEFAULT '',
            estimated_minutes INTEGER DEFAULT 0,
            active INTEGER DEFAULT 1,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chore_completions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            chore_id INTEGER NOT NULL REFERENCES chores(id),
            completed_at TEXT NOT NULL,
            note TEXT DEFAULT '',
            skipped INTEGER DEFAULT 0
        );
    """)


def ensure_schema(conn: Any) -> None:
    """Idempotent: create tables + apply pending migrations."""
    MIGRATIONS.ensure(conn)
