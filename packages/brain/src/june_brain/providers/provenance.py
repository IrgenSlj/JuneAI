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


def _privacy_dial_value() -> str:
    """Best-effort read of the active privacy dial, for the ledger payload."""
    try:
        from june_brain.config_store import get_privacy_dial

        return get_privacy_dial().value
    except Exception:  # noqa: BLE001 - never let a dial read break a model call
        return "unknown"


def _record_egress_to_ledger(event: CloudCallEvent) -> None:
    """Append one tamper-evident egress entry per cloud call (ADR 0022).

    Written here, on the ``start`` phase, because every ``GeminiProvider`` call
    routes through ``record_cloud_call`` — so a skill cannot perform cloud egress
    and skip the ledger. The payload is a field summary (never raw content) and is
    redacted again inside the writer. Best-effort: a ledger failure must never
    break the model call, and recording only on ``start`` yields exactly one entry
    per call (not one per start/end pair).
    """
    if event.phase != "start":
        return
    try:
        from june_brain.trust import get_writer

        get_writer().append(
            kind="egress",
            actor="june",
            payload={
                "model_id": event.model_id,
                "summary": event.payload_summary,
                "privacy_dial": _privacy_dial_value(),
                "at": event.at,
            },
        )
    except Exception:  # noqa: BLE001 - the ledger is best-effort at the call site
        logging.debug("trust-ledger egress append failed", exc_info=True)


def record_cloud_call(event: CloudCallEvent) -> None:
    """Dispatch a cloud-call event to the registered recorder (default: log).

    Always also records a Trust Ledger egress entry (ADR 0022), independent of
    whether a recorder callback is registered, so egress is provable after the
    fact and the record cannot be bypassed.
    """
    _record_egress_to_ledger(event)
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
