"""Tests for persistent telemetry event logging."""

from datetime import date
from unittest.mock import patch

import pytest
from june_brain.memory import Memory
from june_brain.telemetry import (
    append_event,
    get_recent_events,
    record_route_selection,
    record_save_event,
    record_tool_call,
)


@pytest.fixture
def memory_dir(tmp_path):
    """Patch the memory directory for each test."""
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
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

    assert len(events) == 3
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


def test_telemetry_filter_by_event_type(mem):
    """get_recent_events(event_type=...) only returns matching events."""
    append_event(mem, "tool_call", name="old_tool")
    append_event(mem, "route_selection", name="wellness", status="selected",
                 source="graph", route="wellness",
                 payload={"reason": "body metrics were mentioned"})

    tool_events = get_recent_events(mem, event_type="tool_call")
    route_events = get_recent_events(mem, event_type="route_selection")

    assert len(tool_events) == 1
    assert tool_events[0]["name"] == "old_tool"
    assert len(route_events) == 1
    assert route_events[0]["event_type"] == "route_selection"
    assert route_events[0]["schema_version"] == 1
