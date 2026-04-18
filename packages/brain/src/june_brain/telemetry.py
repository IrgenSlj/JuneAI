"""Telemetry helpers for persistent June event logging."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .memory import Memory

EVENT_SCHEMA_VERSION = 1
EVENT_STORE_SCHEMA_VERSION = 1

JsonDict = dict[str, Any]


def append_event(
    memory: Memory,
    event_type: str,
    *,
    name: str = "",
    status: str = "ok",
    source: str = "memory",
    route: str = "",
    payload: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Append a versioned telemetry event to a user's persistent log."""
    return memory.append_event(
        event_type=event_type,
        name=name,
        status=status,
        source=source,
        route=route,
        payload=payload,
    )


def record_tool_call(
    memory: Memory,
    tool_name: str,
    *,
    status: str = "started",
    source: str = "graph",
    route: str = "",
    payload: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Record a tool call event."""
    return memory.record_tool_call(
        tool_name=tool_name,
        status=status,
        source=source,
        route=route,
        payload=payload,
    )


def record_route_selection(
    memory: Memory,
    route: str,
    *,
    source: str = "graph",
    payload: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Record a route selection event."""
    return memory.record_route_selection(route=route, source=source, payload=payload)


def record_save_event(
    memory: Memory,
    kind: str,
    name: str,
    *,
    status: str = "saved",
    source: str = "memory",
    route: str = "",
    payload: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Record a persistence event for a saved artifact."""
    return memory.record_save_event(
        kind=kind,
        name=name,
        status=status,
        source=source,
        route=route,
        payload=payload,
    )


def get_recent_events(
    memory: Memory,
    limit: int = 20,
    event_type: str = "",
) -> list[JsonDict]:
    """Read recent telemetry events from persistent storage."""
    return memory.get_recent_events(limit=limit, event_type=event_type)
