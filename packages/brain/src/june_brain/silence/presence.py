"""Locally derived presence signals for the Silence Model (ADR 0023).

Pure functions — no I/O, no clock read. The caller injects ``now_iso``
so these are deterministic and testable.
"""

from __future__ import annotations

from datetime import datetime

from .policy import PRESENCE_ABSENT, PRESENCE_ACTIVE, PRESENCE_IDLE

# Tunables (seconds). Module-level constants so they read declaratively.
ACTIVE_WINDOW_SECONDS = 120  # <= 2 min  -> present-active
IDLE_WINDOW_SECONDS = 1200  # <= 20 min -> present-idle
_CHAT_TURN_WINDOW_SECONDS = 300  # window within which a chat turn counts as open


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def derive_presence(
    activity_entries: list,
    *,
    now_iso: str,
) -> tuple[str, bool]:
    """Derive (presence_state, active_thread_open) from recent activity entries.

    ``activity_entries`` is a list of objects with ``.timestamp`` (ISO-8601 str)
    and ``.label`` (str). Parse failures are silently skipped (conservative).

    Returns
    -------
    A 2-tuple (presence_state, active_thread_open) where presence_state is one
    of the PRESENCE_* constants and active_thread_open is True only when the
    most recent entry within _CHAT_TURN_WINDOW_SECONDS is a ``POST /chat`` turn.

    ``active_thread_open`` defaults to False when it cannot be reliably
    determined. At the Home seam the user is not mid-chat, so False is safe.
    Activity labels are recorded as "{METHOD} {path}" by the API middleware
    (june_api/app.py); a real chat turn is ``POST /chat`` and a history fetch
    is ``GET /chat/history`` — they are distinguishable by label equality.
    """
    now = _parse_iso(now_iso)
    if now is None:
        return PRESENCE_ABSENT, False

    # Find the most recent parseable entry and its label.
    most_recent_dt: datetime | None = None
    most_recent_label: str = ""

    for entry in activity_entries:
        ts_str = str(getattr(entry, "timestamp", None) or "")
        dt = _parse_iso(ts_str)
        if dt is None:
            continue
        if most_recent_dt is None or dt > most_recent_dt:
            most_recent_dt = dt
            most_recent_label = str(getattr(entry, "label", "") or "")

    if most_recent_dt is None:
        return PRESENCE_ABSENT, False

    try:
        delta_seconds = (now - most_recent_dt).total_seconds()
    except (TypeError, ValueError, OverflowError):
        return PRESENCE_ABSENT, False

    # Presence state — buckets by recency.
    if delta_seconds <= ACTIVE_WINDOW_SECONDS:
        presence = PRESENCE_ACTIVE
    elif delta_seconds <= IDLE_WINDOW_SECONDS:
        presence = PRESENCE_IDLE
    else:
        presence = PRESENCE_ABSENT

    # active_thread_open: True only when the most recent entry is within the
    # chat-turn window AND its label matches the POST /chat endpoint exactly.
    # Conservative: any ambiguity yields False.
    active_thread_open = (
        delta_seconds <= _CHAT_TURN_WINDOW_SECONDS
        and most_recent_label == "POST /chat"
    )

    return presence, active_thread_open
