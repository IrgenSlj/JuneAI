"""Notification bus — lightweight pub/sub for personal assistant notifications.

Usage::

    from .notification import bus, Notification

    # Register a channel (e.g. desktop, Telegram)
    bus.register("telegram", my_handler)

    # Dispatch a notification to all channels
    bus.dispatch(Notification(title="Reminder", body="Drink water", priority="medium", source="chores"))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Notification:
    title: str
    body: str
    priority: str = "medium"  # low | medium | high | urgent
    channel_hint: str | None = None  # None = deliver to all registered channels
    source: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


NotificationHandler = Callable[[Notification], bool]


class NotificationBus:
    """In-memory pub/sub bus with pluggable channel handlers."""

    def __init__(self) -> None:
        self._channels: dict[str, NotificationHandler] = {}
        self.register("log", _log_channel)

    def register(self, name: str, handler: NotificationHandler) -> None:
        """Register a channel handler by name."""
        self._channels[name] = handler
        logger.debug("Notification channel registered: %s", name)

    def unregister(self, name: str) -> None:
        """Remove a channel handler."""
        self._channels.pop(name, None)
        logger.debug("Notification channel unregistered: %s", name)

    def dispatch(self, notification: Notification) -> list[tuple[str, bool]]:
        """Deliver a notification to all matching channels.

        Returns list of ``(channel_name, success)`` tuples.
        """
        results: list[tuple[str, bool]] = []
        for name, handler in self._channels.items():
            if notification.channel_hint and name != notification.channel_hint:
                continue
            try:
                ok = handler(notification)
            except Exception:  # noqa: BLE001
                logger.exception("Notification channel %s failed", name)
                ok = False
            results.append((name, ok))
        return results


# ---------------------------------------------------------------------------
# Built-in: log channel (always on, for audit trail)
# ---------------------------------------------------------------------------

def _log_channel(notification: Notification) -> bool:
    logger.info(
        "NOTIFICATION [%s] %s: %s",
        notification.priority.upper(),
        notification.title,
        notification.body[:200],
    )
    return True


# Module-level singleton
bus = NotificationBus()
bus.register("log", _log_channel)
