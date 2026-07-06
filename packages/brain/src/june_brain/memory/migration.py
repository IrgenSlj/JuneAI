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
import sqlite3
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
    # Tables removed in migration 6; see ADR 0016.
    pass


@MIGRATIONS.register(4, "Add operating-layer ledger: events, capture_items, action_intents")
def _migration_004(conn: Any) -> None:
    # Tables removed in migration 6; see ADR 0016.
    pass


@MIGRATIONS.register(5, "Add salience bookkeeping columns to semantic_facts")
def _migration_005(conn: Any) -> None:
    """Add access_count and last_accessed to semantic_facts.

    Idempotent for the already-applied case only: a "duplicate column name"
    error means the column exists (this migration ran before), which is safe to
    swallow. Any OTHER error (missing table, disk full, ...) is a genuine
    failure and MUST propagate — otherwise the migration is marked applied,
    skipped forever, and callers hit an unhelpful missing-column error at query
    time instead of a clean retry on the next startup.
    """
    for stmt in (
        "ALTER TABLE semantic_facts ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE semantic_facts ADD COLUMN last_accessed TEXT NOT NULL DEFAULT ''",
    ):
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise
    conn.commit()


@MIGRATIONS.register(6, "Drop abandoned shopping/chores + operating-layer ledger tables")
def _migration_006(conn: Any) -> None:
    """Remove tables introduced in migrations 003/004 and reversed by ADR 0016.

    DROP TABLE IF EXISTS is safe to run twice (idempotent).
    Children are dropped before parents to respect foreign-key constraints.
    """
    conn.executescript("""
        DROP TABLE IF EXISTS chore_completions;
        DROP TABLE IF EXISTS chores;
        DROP TABLE IF EXISTS price_alerts;
        DROP TABLE IF EXISTS purchase_history;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS action_intents;
        DROP TABLE IF EXISTS capture_items;
        DROP TABLE IF EXISTS events;
    """)


@MIGRATIONS.register(7, "Retrieval v2: bi-temporal columns + FTS5 over fact text")
def _migration_007(conn: Any) -> None:
    """Add bi-temporal validity columns to semantic_facts + a synced FTS5 index.

    ADR 0024. Columns follow the migration-5 idempotent pattern: a "duplicate
    column" OperationalError means the column already exists (this migration
    ran before) and is safe to swallow; any other error is a genuine failure
    and MUST propagate. ``observed_at`` is added plain (no DEFAULT) because
    SQLite forbids a non-constant default such as ``datetime('now')`` in
    ``ALTER TABLE ... ADD COLUMN``, then backfilled from ``created_at``.

    The FTS5 lexical channel is best-effort: if the SQLite build lacks the
    fts5 module, we log a warning and return without creating the FTS table
    or triggers — the columns and index above are still committed. This is
    the graceful degradation ADR 0024 requires (never a hard failure).
    """
    for stmt in (
        "ALTER TABLE semantic_facts ADD COLUMN valid_from TEXT",
        "ALTER TABLE semantic_facts ADD COLUMN valid_to TEXT",
        "ALTER TABLE semantic_facts ADD COLUMN observed_at TEXT",
        "ALTER TABLE semantic_facts ADD COLUMN superseded_by TEXT",
    ):
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise

    conn.execute(
        "UPDATE semantic_facts SET observed_at = created_at WHERE observed_at IS NULL"
    )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_semantic_facts_validity "
        "ON semantic_facts(user_id, valid_to, valid_from)"
    )

    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS semantic_facts_fts USING fts5("
            "fact_id UNINDEXED, user_id UNINDEXED, text, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
    except sqlite3.OperationalError as exc:
        msg = str(exc).lower()
        if "no such module" in msg or "fts5" in msg:
            logger.warning(
                "Schema migration 7: FTS5 module unavailable (%s) — lexical "
                "recall channel disabled; retrieval falls back to "
                "vec+graph+structured signals only.",
                exc,
            )
            return
        raise

    (fts_count,) = conn.execute("SELECT count(*) FROM semantic_facts_fts").fetchone()
    if fts_count == 0:
        conn.execute(
            "INSERT INTO semantic_facts_fts(fact_id, user_id, text) "
            "SELECT fact_id, user_id, text FROM semantic_facts"
        )

    conn.executescript("""
        CREATE TRIGGER IF NOT EXISTS trg_semantic_facts_fts_insert
        AFTER INSERT ON semantic_facts
        BEGIN
            INSERT INTO semantic_facts_fts(fact_id, user_id, text)
            VALUES (new.fact_id, new.user_id, new.text);
        END;

        CREATE TRIGGER IF NOT EXISTS trg_semantic_facts_fts_delete
        AFTER DELETE ON semantic_facts
        BEGIN
            DELETE FROM semantic_facts_fts
            WHERE fact_id = old.fact_id AND user_id = old.user_id;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_semantic_facts_fts_update
        AFTER UPDATE ON semantic_facts
        BEGIN
            DELETE FROM semantic_facts_fts
            WHERE fact_id = old.fact_id AND user_id = old.user_id;
            INSERT INTO semantic_facts_fts(fact_id, user_id, text)
            VALUES (new.fact_id, new.user_id, new.text);
        END;
    """)


def ensure_schema(conn: Any) -> None:
    """Idempotent: create tables + apply pending migrations."""
    MIGRATIONS.ensure(conn)
