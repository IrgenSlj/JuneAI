"""Minimal provenance hook for cloud model calls (C.1-scoped).

Full SSE TurnProvenance is C.6. This module provides the invariant:
"no cloud call without a provenance event" — every GeminiProvider call
records start/end events through a registered callback.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal


@dataclass
class CloudCallEvent:
    model_id: str
    phase: Literal["start", "end"]
    payload_summary: str
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


_recorder: Callable[[CloudCallEvent], None] | None = None


def record_cloud_call(event: CloudCallEvent) -> None:
    """Dispatch a cloud-call event to the registered recorder (default: log)."""
    if _recorder is not None:
        _recorder(event)
    else:
        logging.info("cloud-call %s %s: %s", event.phase, event.model_id, event.payload_summary)


def set_cloud_call_recorder(fn: Callable[[CloudCallEvent], None]) -> None:
    """Register a callback that receives every CloudCallEvent."""
    global _recorder
    _recorder = fn


def reset_cloud_call_recorder() -> None:
    """Remove the registered callback, reverting to the default logging sink."""
    global _recorder
    _recorder = None
