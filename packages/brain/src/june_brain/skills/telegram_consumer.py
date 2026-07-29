"""Consumer for inbound Telegram events.

Polls ``skill_inbound_events`` for unprocessed Telegram messages, routes each
through the harness loop as a user turn, then sends the response back via the
Telegram ``send_telegram_message`` MCP tool.

Started once at application startup (or lazily on first demand). Runs a
lightweight polling loop that checks the table every few seconds.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import UTC, datetime
from typing import Any

from ..loop.interface import SessionState
from ..memory.sqlite import _get_connection
from ..memory.sqlite import db_path as memory_db_path
from ..providers.base import Message

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 3.0
_MAX_EVENTS_PER_TICK = 5
_TELEGRAM_SKILL_KEY = "telegram"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TelegramInboundConsumer:
    """Poll ``skill_inbound_events`` and route Telegram messages through the loop.

    Usage::

        consumer = TelegramInboundConsumer(user_id="default")
        await consumer.start()   # runs forever in a background task
        # ... or:
        consumer.poll_once()     # one tick, for testing
    """

    def __init__(
        self,
        user_id: str = "default",
        *,
        loop_factory: Any | None = None,
    ) -> None:
        self.user_id = user_id
        self._loop_factory = loop_factory
        self._running = False
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the polling loop as an asyncio background task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("TelegramInboundConsumer started (poll interval: %.1fs)", _POLL_INTERVAL)

    async def stop(self) -> None:
        """Stop the polling loop gracefully."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("TelegramInboundConsumer stopped")

    async def poll_once(self) -> int:
        """Run one polling tick. Returns the number of events processed."""
        return await self._poll_events()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    def status(self) -> dict[str, Any]:
        return {
            "active": self._running,
            "user_id": self.user_id,
            "poll_interval_s": _POLL_INTERVAL,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Continuously poll for new events while running."""
        while self._running:
            try:
                count = await self._poll_events()
                if count:
                    logger.debug("TelegramInboundConsumer processed %d event(s)", count)
            except Exception:  # noqa: BLE001
                logger.exception("TelegramInboundConsumer poll tick failed")
            await asyncio.sleep(_POLL_INTERVAL)

    async def _poll_events(self) -> int:
        """Query ``skill_inbound_events`` for unprocessed Telegram messages.

        Returns the count of events processed.
        """
        try:
            conn = _get_connection(memory_db_path())
            rows = conn.execute(
                """SELECT id, payload, user_id FROM skill_inbound_events
                   WHERE skill_key=? AND processed=0
                   ORDER BY id ASC LIMIT ?""",
                (_TELEGRAM_SKILL_KEY, _MAX_EVENTS_PER_TICK),
            ).fetchall()
        except sqlite3.OperationalError:
            # Table may not exist yet on a fresh install.
            return 0

        processed = 0
        for row in rows:
            event_id: int = int(row["id"])
            payload_raw: str = str(row["payload"] or "{}")
            event_user_id: str = str(row["user_id"] or self.user_id)

            try:
                payload = json.loads(payload_raw)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Telegram event %d has invalid payload; marking processed", event_id)
                self._mark_processed(conn, event_id)
                processed += 1
                continue

            chat_id = payload.get("chat_id")
            text = payload.get("text", "").strip()
            if not chat_id or not text:
                logger.debug("Telegram event %d missing chat_id or text; skipping", event_id)
                self._mark_processed(conn, event_id)
                processed += 1
                continue

            await self._handle_message(event_id, event_user_id, chat_id, text)
            processed += 1

        return processed

    async def _handle_message(
        self,
        event_id: int,
        event_user_id: str,
        chat_id: int | str,
        text: str,
    ) -> None:
        """Route one Telegram message through the harness loop and send the reply."""
        conn = _get_connection(memory_db_path())

        try:
            loop = self._resolve_loop()
            session = SessionState(user_id=event_user_id, messages=[])
            user_msg = Message(role="user", content=text)

            # Stream the turn — collect all tokens into the reply.
            reply_parts: list[str] = []
            async for ev in loop.stream_turn(session, user_msg):
                if ev.type == "token":
                    reply_parts.append(ev.content)

            reply = "".join(reply_parts).strip()
            if not reply:
                reply = "I received your message but don't have a reply right now."
        except Exception:  # noqa: BLE001
            logger.exception("TelegramInboundConsumer: loop turn failed for event %d", event_id)
            reply = "Sorry, I encountered an error processing your message."

        # Send the reply via the Telegram skill's MCP tool.
        try:
            await self._send_telegram_reply(conn, chat_id, reply)
        except Exception:  # noqa: BLE001
            logger.exception(
                "TelegramInboundConsumer: failed to send reply for event %d", event_id,
            )

        self._mark_processed(conn, event_id, agent_invoked=True)

    async def _send_telegram_reply(
        self,
        conn: sqlite3.Connection,
        chat_id: int | str,
        text: str,
    ) -> None:
        """Send a reply to a Telegram chat via the skill's outbound path.

        Posts to the Telegram Bot API directly, matching the pattern used by
        the Telegram skill's ``send_telegram_message`` tool.
        """
        import urllib.request

        token = _get_telegram_bot_token()
        if not token:
            logger.warning("JUNE_TELEGRAM_BOT_TOKEN not set; cannot send reply")
            return

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=15)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace") if exc.fp else ""
            logger.error("Telegram sendMessage failed (%s): %s", exc.code, body)

    def _mark_processed(
        self,
        conn: sqlite3.Connection,
        event_id: int,
        *,
        agent_invoked: bool = False,
    ) -> None:
        """Mark an inbound event as processed so we don't replay it."""
        conn.execute(
            "UPDATE skill_inbound_events SET processed=1, agent_invoked=? WHERE id=?",
            (int(agent_invoked), event_id),
        )
        conn.commit()

    def _resolve_loop(self) -> Any:
        if self._loop_factory is not None:
            return self._loop_factory()
        from ..loop.engine import get_loop

        return get_loop()


def _get_telegram_bot_token() -> str:
    """Return the Telegram bot token from the environment or secret store."""
    import os

    token = os.environ.get("JUNE_TELEGRAM_BOT_TOKEN", "")
    if token:
        return token
    try:
        from ..secret_store import SecretStore

        store = SecretStore()
        return store.get("JUNE_TELEGRAM_BOT_TOKEN") or ""
    except Exception:  # noqa: BLE001
        return ""
