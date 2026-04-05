"""Tests for memory migration: JSON → SQLite and app-state persistence."""

from datetime import date
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.memory import Memory


@pytest.fixture
def memory_dir(tmp_path):
    """Patch the memory directory for each test."""
    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        yield tmp_path


@pytest.fixture
def mem(memory_dir):
    """Memory instance backed by a temporary directory."""
    return Memory("test_user")


def test_app_state_roundtrips_correctly(mem):
    """set_app_state_value / get_app_state should roundtrip all JSON-safe types."""
    today = date.today().isoformat()
    mem.set_app_state_value("last_daily_checkin_date", today)
    mem.set_app_state_value("selected_chapter", "body")

    state = mem.get_app_state()
    assert state["last_daily_checkin_date"] == today
    assert state["selected_chapter"] == "body"


def test_daily_checkin_persists_across_instances(memory_dir):
    """should_send_daily_checkin uses SQLite, so state is shared across instances."""
    m1 = Memory("test_user")
    assert m1.should_send_daily_checkin() is True
    m1.mark_daily_checkin_sent()

    m2 = Memory("test_user")
    assert m2.should_send_daily_checkin() is False


def test_json_migration_imports_chat_history(memory_dir):
    """Legacy JSON chat file is imported into SQLite on first Memory init."""
    uid = "test_user"
    chat_data = [
        {"role": "user", "content": "hello from JSON", "timestamp": "2026-01-01T10:00:00"},
        {"role": "assistant", "content": "hi back", "timestamp": "2026-01-01T10:00:01"},
    ]
    (memory_dir / f"{uid}_chat.json").write_text(json.dumps(chat_data), encoding="utf-8")

    mem = Memory(uid)
    history = mem.load_chat()

    assert len(history) == 2
    assert history[0]["content"] == "hello from JSON"
    assert history[1]["role"] == "assistant"
    # JSON file should be renamed, not left around
    assert not (memory_dir / f"{uid}_chat.json").exists()
    assert (memory_dir / f"{uid}_chat.json.migrated").exists()


def test_json_migration_imports_app_state(memory_dir):
    """Legacy flat app_state JSON is imported with all keys preserved."""
    uid = "test_user"
    today = date.today().isoformat()
    legacy = {
        "schema_version": 1,
        "data": {
            "last_daily_checkin_date": today,
            "runtime_preset": "local_gemma_4",
        },
    }
    (memory_dir / f"{uid}_app_state.json").write_text(json.dumps(legacy), encoding="utf-8")

    mem = Memory(uid)
    state = mem.get_app_state()

    assert state["last_daily_checkin_date"] == today
    assert state["runtime_preset"] == "local_gemma_4"
    assert mem.should_send_daily_checkin() is False
