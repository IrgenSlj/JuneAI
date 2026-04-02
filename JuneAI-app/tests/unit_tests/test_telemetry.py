"""Tests for persistent telemetry event logging."""

from datetime import date
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.memory import Memory
from agent.telemetry import (
    append_event,
    get_recent_events,
    record_route_selection,
    record_save_event,
    record_tool_call,
)


@pytest.fixture
def memory_dir(tmp_path):
    """Patch the memory directory for each test."""
    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        yield


@pytest.fixture
def mem(memory_dir):
    """Memory instance backed by a temporary directory."""
    return Memory("test_user")


def test_telemetry_helpers_persist_versioned_events(mem):
    today = date.today()
    tool_event = record_tool_call(
        mem,
        "save_calendar_item",
        route="planner",
        payload={"issued_at": today, "candidates": {"planner", "calendar"}},
    )
    route_event = record_route_selection(
        mem,
        "planner",
        payload={"reason": "calendar item detected"},
    )
    save_event = record_save_event(
        mem,
        "calendar",
        "Dentist",
        payload={"date": today.isoformat()},
    )

    events = get_recent_events(mem, limit=10)
    persisted = json.loads(Path(mem.telemetry_file).read_text(encoding="utf-8"))

    assert persisted["schema_version"] == 1
    assert len(persisted["events"]) == 3
    assert [event["event_type"] for event in events] == [
        "tool_call",
        "route_selection",
        "save_event",
    ]
    assert all(event["schema_version"] == 1 for event in events)
    assert tool_event["event_id"]
    assert tool_event["payload"]["issued_at"] == today.isoformat()
    assert tool_event["payload"]["candidates"] == ["calendar", "planner"]
    assert route_event["name"] == "planner"
    assert save_event["payload"]["kind"] == "calendar"


def test_telemetry_rehydrates_legacy_lists_and_filters(mem):
    legacy_events = [
        {
            "event_type": "tool_call",
            "name": "old_tool",
            "timestamp": "2026-03-27T10:00:00",
        }
    ]
    Path(mem.telemetry_file).write_text(json.dumps(legacy_events), encoding="utf-8")

    recent = get_recent_events(mem, event_type="tool_call")
    app_event = append_event(
        mem,
        "route_selection",
        name="wellness",
        status="selected",
        source="graph",
        route="wellness",
        payload={"reason": "body metrics were mentioned"},
    )
    persisted = json.loads(Path(mem.telemetry_file).read_text(encoding="utf-8"))

    assert len(recent) == 1
    assert recent[0]["name"] == "old_tool"
    assert app_event["schema_version"] == 1
    assert persisted["schema_version"] == 1
    assert len(persisted["events"]) == 2
    assert persisted["events"][0]["schema_version"] == 1
    assert persisted["events"][1]["event_type"] == "route_selection"
