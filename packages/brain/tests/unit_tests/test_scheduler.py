"""Tests for the scheduler models and store."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from june_brain.scheduler.models import Schedule
from june_brain.scheduler.service import compute_next_run


def test_is_due():
    past = datetime.now(timezone.utc) - timedelta(minutes=5)
    s = Schedule(id="d1", user_id="u", name="Past", scheduled_at=past.isoformat(), enabled=True)
    assert s.is_due()

    future = datetime.now(timezone.utc) + timedelta(hours=1)
    s2 = Schedule(id="d2", user_id="u", name="Future", scheduled_at=future.isoformat(), enabled=True)
    assert not s2.is_due()


def test_disabled_not_due():
    past = datetime.now(timezone.utc) - timedelta(minutes=5)
    s = Schedule(id="d3", user_id="u", name="Disabled", scheduled_at=past.isoformat(), enabled=False)
    assert not s.is_due()


def test_exhausted_not_due():
    past = datetime.now(timezone.utc) - timedelta(minutes=5)
    s = Schedule(id="d4", user_id="u", name="Exhausted", scheduled_at=past.isoformat(),
                 enabled=True, max_runs=1, run_count=1)
    assert not s.is_due()


def test_store_crud():
    conn = _make_conn()
    from june_brain.scheduler.store import ScheduleStore

    store = ScheduleStore(conn)
    s = Schedule(id="cr1", user_id="u", name="CRUD test", scheduled_at="2026-01-01T00:00:00")
    store.create(s)
    assert store.get("cr1") is not None
    assert len(store.list("u")) == 1
    s.name = "Updated"
    store.update(s)
    assert store.get("cr1").name == "Updated"
    store.delete("cr1")
    assert store.get("cr1") is None


def test_compute_next_run_interval():
    s = Schedule(id="int", user_id="u", name="Int", interval_seconds=3600, scheduled_at="2000-01-01T00:00:00")
    next_run = compute_next_run(s)
    assert next_run is not None


def test_compute_next_run_exhausted():
    s = Schedule(id="exh", user_id="u", name="Exhausted", max_runs=1, run_count=1, scheduled_at="2000-01-01T00:00:00")
    assert compute_next_run(s) is None


@pytest.fixture
def store():
    conn = _make_conn()
    from june_brain.scheduler.store import ScheduleStore
    return ScheduleStore(conn)


def test_list_due(store):
    past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    store.create(Schedule(id="a", user_id="u", name="A", scheduled_at=past))
    store.create(Schedule(id="b", user_id="u", name="B", scheduled_at=future))
    due = store.list_due("u")
    assert len(due) == 1
    assert due[0].id == "a"


def test_mark_run(store):
    past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    s = store.create(Schedule(id="mr", user_id="u", name="MR", scheduled_at=past, interval_seconds=300))
    assert s.run_count == 0
    store.mark_run(s)
    updated = store.get("mr")
    assert updated is not None
    assert updated.run_count == 1
    assert updated.last_run_at is not None and len(updated.last_run_at) > 0


def _make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS schedules (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL,
            description TEXT DEFAULT '', cron_expression TEXT DEFAULT '',
            interval_seconds INTEGER DEFAULT 0, scheduled_at TEXT NOT NULL,
            last_run_at TEXT, action_type TEXT NOT NULL DEFAULT 'agent_invoke',
            action_config TEXT NOT NULL DEFAULT '{}', max_runs INTEGER DEFAULT 0,
            run_count INTEGER DEFAULT 0, enabled INTEGER DEFAULT 1,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )
    """)
    return conn
