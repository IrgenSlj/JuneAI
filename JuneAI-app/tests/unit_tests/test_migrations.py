"""Tests for JSON store versioning and migration upgrades."""

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
        yield


@pytest.fixture
def mem(memory_dir):
    """Memory instance backed by a temporary directory."""
    return Memory("test_user")


def test_app_state_upgrades_legacy_flat_payload(mem):
    today = date.today().isoformat()
    legacy_state = {
        "last_daily_checkin_date": today,
        "selected_chapter": "body",
    }
    Path(mem.app_state_file).write_text(json.dumps(legacy_state), encoding="utf-8")

    state = mem.get_app_state()
    persisted = json.loads(Path(mem.app_state_file).read_text(encoding="utf-8"))

    assert state == legacy_state
    assert persisted["schema_version"] == 1
    assert persisted["data"] == legacy_state


def test_app_state_current_writes_use_wrapped_schema(mem):
    today = date.today().isoformat()

    state = mem.set_app_state_value("last_daily_checkin_date", today)
    persisted = json.loads(Path(mem.app_state_file).read_text(encoding="utf-8"))

    assert state["last_daily_checkin_date"] == today
    assert persisted["schema_version"] == 1
    assert persisted["data"]["last_daily_checkin_date"] == today
    assert mem.should_send_daily_checkin() is False
