"""Notification bus — lightweight pub/sub for personal assistant notifications.

Usage::

    from .notification import bus, Notification

    # Register a channel (e.g. desktop, Telegram)
    bus.register("telegram", my_handler)

    # Dispatch a notification to all channels
    bus.dispatch(Notification(title="Reminder", body="Standup at 10:00", priority="medium", source="calendar"))
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Notification:
    title: str
    body: str
    priority: str = "medium"  # low | medium | high | urgent
    channel_hint: str | None = None  # None = deliver to all registered channels
    source: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
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


# ---------------------------------------------------------------------------
# Telegram channel — writes to skill_inbound_events for daemon skill pickup
# ---------------------------------------------------------------------------

def _default_data_dir() -> Path:
    data_dir = Path.home() / "Library" / "Application Support" / "June"
    env = __import__("os").environ.get("JUNE_DATA_DIR")
    if env:
        data_dir = Path(env)
    return data_dir


def telegram_channel(notification: Notification) -> bool:
    """Forward a notification to the Telegram daemon skill.

    Writes to ``skill_inbound_events``, where the Telegram skill picks it
    up on its next poll and sends it to the configured chat.
    """
    chat_id = notification.metadata.get("chat_id", notification.metadata.get("telegram_chat_id"))
    if not chat_id:
        logger.debug("Telegram channel: no chat_id in metadata, skipping")
        return False
    db_path = _default_data_dir() / "june.db"
    if not db_path.exists():
        logger.warning("Telegram channel: june.db not found at %s", db_path)
        return False
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO skill_inbound_events (skill_key, event_type, payload, user_id) "
        "VALUES (?, ?, ?, ?)",
        (
            "telegram",
            "send_notification",
            json.dumps({
                "chat_id": chat_id,
                "text": f"*{notification.title}*\n{notification.body}",
                "priority": notification.priority,
            }),
            str(chat_id),
        ),
    )
    conn.commit()
    conn.close()
    return True


# Module-level singleton
bus = NotificationBus()
bus.register("log", _log_channel)
