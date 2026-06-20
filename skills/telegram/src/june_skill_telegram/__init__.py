"""June Telegram skill -- bridge between Telegram and JuneAI.

Architecture
------------
This is a **daemon MCP skill** (manifest ``daemon = true``). It runs as a
subprocess managed by the June supervisor. On startup it:

1. Registers ``send_telegram_message`` and ``get_telegram_bot_status`` tools.
2. Starts a background thread that long-polls Telegram for new messages.
3. Writes incoming messages to the ``skill_inbound_events`` table in June's
   SQLite database, where the scheduler's event poller picks them up.

Configuration
-------------
Set ``JUNE_TELEGRAM_BOT_TOKEN`` to your Bot Token (from `@BotFather`).
The skill reads ``JUNE_DATA_DIR`` (default ``~/Library/Application Support/June``)
to find the shared database for inbound events.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from june_brain.skills.server import MCPStdioServer

logger = logging.getLogger(__name__)

server = MCPStdioServer(name="june-telegram", version="0.1.0")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOT_TOKEN: str | None = None
_BOT_USERNAME: str | None = None
_POLLING = False


def _bot_token() -> str:
    global _BOT_TOKEN
    if _BOT_TOKEN is None:
        tok = os.environ.get("JUNE_TELEGRAM_BOT_TOKEN", "")
        if not tok:
            raise RuntimeError(
                "JUNE_TELEGRAM_BOT_TOKEN not set. "
                "Create a bot via @BotFather and set this env var."
            )
        _BOT_TOKEN = tok
    return _BOT_TOKEN


def _db_path() -> str:
    data_dir = Path(os.environ.get("JUNE_DATA_DIR", ""))
    if not data_dir.exists():
        data_dir = Path.home() / "Library" / "Application Support" / "June"
    return str(data_dir / "june.db")


def _ensure_inbound_table() -> None:
    conn = sqlite3.connect(_db_path())
    conn.execute(
        """CREATE TABLE IF NOT EXISTS skill_inbound_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            skill_key TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload TEXT NOT NULL DEFAULT '{}',
            user_id TEXT NOT NULL DEFAULT 'default',
            processed INTEGER NOT NULL DEFAULT 0,
            agent_invoked INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )"""
    )
    conn.commit()
    conn.close()


def _write_event(
    event_type: str,
    payload: dict[str, Any],
    user_id: str = "default",
) -> None:
    conn = sqlite3.connect(_db_path())
    conn.execute(
        "INSERT INTO skill_inbound_events (skill_key, event_type, payload, user_id) "
        "VALUES (?, ?, ?, ?)",
        ("telegram", event_type, json.dumps(payload), user_id),
    )
    conn.commit()
    conn.close()


def _api_request(
    method: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    import urllib.error
    import urllib.parse
    import urllib.request

    token = _bot_token()
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = json.dumps(params).encode() if params else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        logger.error("Telegram API error %s: %s", exc.code, body)
        return {"ok": False, "description": body}
    except Exception as exc:
        logger.error("Telegram API request failed: %s", exc)
        return {"ok": False, "description": str(exc)}
    return result


def _poll_loop() -> None:
    """Background thread: long-poll Telegram for updates."""
    global _POLLING
    if _POLLING:
        return
    _POLLING = True
    _ensure_inbound_table()
    offset = 0
    logger.info("Telegram polling thread started")

    while _POLLING:
        try:
            result = _api_request("getUpdates", {"offset": offset, "timeout": 30})
            if not isinstance(result, dict) or not result.get("ok"):
                time.sleep(10)
                continue
            updates = result.get("result", [])
            for update in updates:
                update_id = update.get("update_id", 0)
                if update_id >= offset:
                    offset = update_id + 1
                message = update.get("message") or update.get("callback_query", {}).get("message", {})
                if not message:
                    continue
                chat_id = message.get("chat", {}).get("id")
                text = message.get("text", "")
                user = message.get("from", {})
                if chat_id and text:
                    _write_event(
                        "message",
                        {
                            "chat_id": chat_id,
                            "text": text,
                            "from_id": user.get("id"),
                            "from_name": user.get("first_name", ""),
                        },
                        user_id=str(chat_id),
                    )
        except Exception as exc:
            logger.error("Telegram poll error: %s", exc)
            time.sleep(10)

    logger.info("Telegram polling thread stopped")


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


@server.tool(
    name="send_telegram_message",
    description="Send a message to a Telegram chat.",
    input_schema={
        "type": "object",
        "properties": {
            "chat_id": {
                "type": ["string", "integer"],
                "description": "Telegram chat ID.",
            },
            "text": {"type": "string", "description": "Message text."},
            "parse_mode": {
                "type": "string",
                "default": "",
                "description": "Markdown or HTML.",
            },
        },
        "required": ["chat_id", "text"],
    },
)
def send_telegram_message(
    chat_id: int | str,
    text: str,
    parse_mode: str = "",
) -> str:
    params: dict[str, Any] = {
        "chat_id": int(chat_id) if isinstance(chat_id, str) and chat_id.lstrip("-").isdigit() else chat_id,
        "text": text,
    }
    if parse_mode:
        params["parse_mode"] = parse_mode
    result = _api_request("sendMessage", params)
    if isinstance(result, dict) and result.get("ok"):
        return "Message sent."
    desc = ""
    if isinstance(result, dict):
        desc = result.get("description", str(result))
    return f"Failed to send message: {desc}"


@server.tool(
    name="get_telegram_bot_status",
    description="Check if the Telegram bot is connected and running.",
    input_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
)
def get_telegram_bot_status() -> str:
    if not _BOT_TOKEN and not os.environ.get("JUNE_TELEGRAM_BOT_TOKEN"):
        return "Not configured: JUNE_TELEGRAM_BOT_TOKEN not set."
    result = _api_request("getMe")
    if isinstance(result, dict) and result.get("ok"):
        user = result["result"]
        global _BOT_USERNAME
        _BOT_USERNAME = user.get("username", "")
        return (
            f"Connected as @{_BOT_USERNAME} (id {user.get('id')}). "
            f"Polling: {'active' if _POLLING else 'inactive'}."
        )
    desc = ""
    if isinstance(result, dict):
        desc = result.get("description", str(result))
    return f"Not connected: {desc}"


def main() -> None:
    # Start background polling
    t = threading.Thread(target=_poll_loop, daemon=True)
    t.start()
    server.run()


def stop() -> None:
    global _POLLING
    _POLLING = False
