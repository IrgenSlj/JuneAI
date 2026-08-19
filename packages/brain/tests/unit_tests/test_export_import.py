"""Tests for data portability — export and import."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from june_brain.memory import Memory
from june_brain.memory.export import export_memory
from june_brain.memory.import_ import import_memory


@pytest.fixture
def memory_dir(tmp_path):
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        yield tmp_path


@pytest.fixture
def mem(memory_dir):
    return Memory("test_user")


def _seed_v1_mood(mem: Memory, mood: str = "happy", note: str = "") -> None:
    """Insert a `moods` row the way an upgraded install already has one.

    The v1 writer is gone (D.5b) but the table stays, which is the whole point
    of the "leave the tables, remove the code" decision: a user who used June
    before the cleanup keeps their rows, and export has to keep carrying them.
    Writing the row directly is not a shortcut here — it is the exact state
    these tests exist to cover.
    """
    conn = mem._conn
    conn.execute(
        "INSERT INTO moods (user_id, mood, note, timestamp) VALUES (?,?,?,?)",
        (mem.user_id, mood, note, "2026-01-15T09:00:00"),
    )
    conn.commit()


def test_export_roundtrip(mem, tmp_path):
    """Export memory, verify JSON structure, re-import."""
    mem.save_message("user", "hello")
    mem.save_message("assistant", "hi there")
    _seed_v1_mood(mem, "happy", "great day")

    export_path = tmp_path / "export.json"
    archive = export_memory("test_user", output_path=str(export_path))

    assert archive["version"] == 1
    assert archive["user_id"] == "test_user"
    assert "sqlite" in archive["stores"]

    sqlite_data = archive["stores"]["sqlite"]
    assert "chat_messages" not in sqlite_data  # skipped in export
    assert "moods" in sqlite_data
    assert len(sqlite_data["moods"]) >= 1


def test_export_no_output_path(mem):
    """Export without file path returns dict in memory."""
    archive = export_memory("test_user")
    assert archive["version"] == 1
    assert "sqlite" in archive["stores"]


def test_import_into_empty_db(memory_dir, tmp_path):
    """Import data into a fresh database."""
    # First create some data in one DB
    mem1 = Memory("user_a")
    _seed_v1_mood(mem1)
    mem1.save_preference("test", "value")

    export_path = tmp_path / "export.json"
    export_memory("user_a", output_path=str(export_path))

    # Now import into a different user's DB
    counts = import_memory(export_path, "user_b")
    assert len(counts) > 0

    mem2 = Memory("user_b")
    prefs = mem2.get_preferences()
    assert len(prefs) >= 1
