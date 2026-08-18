"""Cloud call provenance and enforcement gate (ADR 0022 + egress gate).

Full SSE TurnProvenance is C.6. This module provides two invariants:

1. "No cloud call without a provenance event" — every GeminiProvider call
   records start/end events through a registered callback.

2. "No cloud call when local-only" — when the privacy dial is LOCAL_ONLY,
   cloud calls are blocked before they leave the process. The block is
   enforced here, at the single chokepoint every cloud-routed call passes
   through, so a skill cannot perform cloud egress and skip the gate.
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


class CloudEgressBlockedError(Exception):
    """Raised when a cloud call is blocked by the LOCAL_ONLY privacy dial."""


def _privacy_dial_value() -> str:
    """The active privacy dial, for the ledger payload.

    Records "unknown" rather than guessing — see ``june_brain.privacy``.
    """
    from june_brain.privacy import dial_value

    return dial_value()


def _is_local_only() -> bool:
    """Return True if the privacy dial is set to LOCAL_ONLY, or unreadable.

    Delegates to ``june_brain.privacy``, which owns the predicate and fails
    closed: an unreadable dial blocks the call rather than waving it through.
    """
    from june_brain.privacy import local_only

    return local_only()


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

    Enforces the LOCAL_ONLY privacy dial: when active, cloud calls on the
    ``start`` phase are blocked with CloudEgressBlockedError before they
    leave the process. The block is here, at the single chokepoint every
    cloud-routed call passes through, so a skill cannot perform cloud
    egress and skip the gate.

    Always records a Trust Ledger egress entry (ADR 0022) for permitted
    calls, so egress is provable after the fact.
    """
    if event.phase == "start" and _is_local_only():
        logging.warning(
            "cloud egress blocked (LOCAL_ONLY dial): model=%s summary=%s",
            event.model_id,
            event.payload_summary,
        )
        raise CloudEgressBlockedError(
            f"Cloud call to {event.model_id} blocked: privacy dial is LOCAL_ONLY. "
            "Switch to private_by_default or cloud_first to allow cloud calls."
        )

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
