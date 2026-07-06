"""Tests for schema migration 7: bi-temporal columns + FTS5 (ADR 0024, W2 S2.1)."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest


def _fts5_available() -> bool:
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        conn.close()
        return True
    except sqlite3.OperationalError:
        return False


_FTS5_AVAILABLE = _fts5_available()


@pytest.fixture
def memory_dir(tmp_path):
    """Patch the memory directory for each test."""
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        yield tmp_path


# ---------------------------------------------------------------------------
# (a) Fresh DB: columns, index, FTS table + triggers all present
# ---------------------------------------------------------------------------


def test_migration_007_adds_bitemporal_columns_and_index(memory_dir):
    from june_brain.memory.sqlite import Memory, _get_connection, db_path

    Memory("alex")  # __init__ runs ensure_schema -> all migrations, including 7
    conn = _get_connection(db_path())

    cols = {row[1] for row in conn.execute("PRAGMA table_info(semantic_facts)")}
    assert {"valid_from", "valid_to", "observed_at", "superseded_by"} <= cols

    indexes = {row[1] for row in conn.execute("PRAGMA index_list(semantic_facts)")}
    assert "idx_semantic_facts_validity" in indexes

    if _FTS5_AVAILABLE:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "semantic_facts_fts" in tables

        triggers = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
        }
        assert {
            "trg_semantic_facts_fts_insert",
            "trg_semantic_facts_fts_delete",
            "trg_semantic_facts_fts_update",
        } <= triggers


# ---------------------------------------------------------------------------
# (b) Backfill: observed_at populated from created_at for pre-v7 rows
# ---------------------------------------------------------------------------


def test_migration_007_backfills_observed_at_from_created_at():
    from june_brain.memory import sqlite as sqlite_mod
    from june_brain.memory.migration import MIGRATIONS

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(sqlite_mod._SCHEMA_SQL)

    # Simulate migration 5 having already run.
    conn.execute(
        "ALTER TABLE semantic_facts ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0"
    )
    conn.execute(
        "ALTER TABLE semantic_facts ADD COLUMN last_accessed TEXT NOT NULL DEFAULT ''"
    )

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS _schema_migrations (
            version   INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    for version in range(1, 7):
        conn.execute(
            "INSERT INTO _schema_migrations (version) VALUES (?)", (version,)
        )

    conn.execute(
        "INSERT INTO semantic_facts (user_id, fact_id, text, created_at) "
        "VALUES (?, ?, ?, ?)",
        ("alex", "fact-1", "loves coffee", "2026-01-01T00:00:00"),
    )
    conn.commit()

    MIGRATIONS.ensure(conn)  # only migration 7 is pending

    row = conn.execute(
        "SELECT observed_at, created_at FROM semantic_facts WHERE fact_id = ?",
        ("fact-1",),
    ).fetchone()
    assert row["observed_at"] == row["created_at"] == "2026-01-01T00:00:00"


# ---------------------------------------------------------------------------
# (c) Trigger sync: FTS mirrors insert/update/delete on semantic_facts
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _FTS5_AVAILABLE, reason="fts5 module unavailable in this sqlite build")
def test_migration_007_fts_trigger_sync(memory_dir):
    from june_brain.memory.sqlite import Memory, _get_connection, db_path

    Memory("alex")
    conn = _get_connection(db_path())

    conn.execute(
        "INSERT INTO semantic_facts (user_id, fact_id, text, created_at) "
        "VALUES (?, ?, ?, ?)",
        ("alex", "fact-1", "loves black coffee", "2026-01-01T00:00:00"),
    )
    conn.commit()

    hits = conn.execute(
        "SELECT fact_id FROM semantic_facts_fts WHERE semantic_facts_fts MATCH ?",
        ("coffee",),
    ).fetchall()
    assert {row["fact_id"] for row in hits} == {"fact-1"}

    conn.execute(
        "UPDATE semantic_facts SET text = ? WHERE user_id = ? AND fact_id = ?",
        ("prefers herbal tea", "alex", "fact-1"),
    )
    conn.commit()

    hits_new = conn.execute(
        "SELECT fact_id FROM semantic_facts_fts WHERE semantic_facts_fts MATCH ?",
        ("tea",),
    ).fetchall()
    assert {row["fact_id"] for row in hits_new} == {"fact-1"}

    hits_old = conn.execute(
        "SELECT fact_id FROM semantic_facts_fts WHERE semantic_facts_fts MATCH ?",
        ("coffee",),
    ).fetchall()
    assert hits_old == []

    conn.execute(
        "DELETE FROM semantic_facts WHERE user_id = ? AND fact_id = ?",
        ("alex", "fact-1"),
    )
    conn.commit()

    hits_deleted = conn.execute(
        "SELECT fact_id FROM semantic_facts_fts WHERE semantic_facts_fts MATCH ?",
        ("tea",),
    ).fetchall()
    assert hits_deleted == []


# ---------------------------------------------------------------------------
# (d) Idempotency + empty DB
# ---------------------------------------------------------------------------


def test_migration_007_idempotent_and_fresh_empty_db():
    from june_brain.memory.migration import MIGRATIONS

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    MIGRATIONS.ensure(conn)  # fresh empty DB: runs 1..7, must not raise
    MIGRATIONS.ensure(conn)  # second run: no-op, must not raise

    cols = {row[1] for row in conn.execute("PRAGMA table_info(semantic_facts)")}
    assert {"valid_from", "valid_to", "observed_at", "superseded_by"} <= cols
