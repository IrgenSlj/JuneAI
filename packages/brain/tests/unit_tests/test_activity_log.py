"""Tests for the activity log."""

from __future__ import annotations

from pathlib import Path

import pytest

from june_brain.activity import ActivityLog, reset_for_tests


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    reset_for_tests()
    yield
    reset_for_tests()


def test_record_and_list_round_trip() -> None:
    log = ActivityLog()
    entry = log.record(kind="request", label="GET /skills", status=200, latency_ms=12)
    assert entry.id > 0

    listed = log.list(limit=10)
    assert len(listed) == 1
    assert listed[0].label == "GET /skills"
    assert listed[0].status == 200
    assert listed[0].latency_ms == 12


def test_list_returns_newest_first() -> None:
    log = ActivityLog()
    log.record(kind="request", label="first")
    log.record(kind="request", label="second")
    log.record(kind="tool", label="third")
    entries = log.list(limit=10)
    assert [e.label for e in entries] == ["third", "second", "first"]


def test_list_filters_by_kind() -> None:
    log = ActivityLog()
    log.record(kind="request", label="a")
    log.record(kind="tool", label="b")
    log.record(kind="request", label="c")
    requests = log.list(kind="request", limit=10)
    assert [e.label for e in requests] == ["c", "a"]
    tools = log.list(kind="tool", limit=10)
    assert [e.label for e in tools] == ["b"]


def test_record_persists_detail_as_dict() -> None:
    log = ActivityLog()
    log.record(
        kind="tool",
        label="files.list_directory",
        latency_ms=8,
        detail={"path": "~/Documents", "show_hidden": False},
    )
    entry = log.list(limit=1)[0]
    assert entry.detail == {"path": "~/Documents", "show_hidden": False}


def test_log_caps_at_max_rows() -> None:
    from june_brain import activity as activity_mod

    log = ActivityLog()
    cap = activity_mod._MAX_ROWS
    for i in range(cap + 10):
        log.record(kind="request", label=f"r{i}")
    entries = log.list(limit=cap + 50)
    assert len(entries) == cap
    # Newest preserved, oldest evicted.
    assert entries[0].label == f"r{cap + 9}"
    assert entries[-1].label == "r10"


def test_clear_removes_all_rows() -> None:
    log = ActivityLog()
    log.record(kind="request", label="x")
    log.record(kind="request", label="y")
    removed = log.clear()
    assert removed == 2
    assert log.list(limit=10) == []
