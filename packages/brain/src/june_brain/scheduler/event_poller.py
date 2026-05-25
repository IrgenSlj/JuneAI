"""Poll the ``skill_inbound_events`` table for unprocessed events and dispatch them.

Part of the daemon MCP skill pattern (see ADR 0013).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def poll_events(
    db_path: str,
    user_id: str,
    agent_invoke: Any | None = None,
    batch_size: int = 10,
) -> list[dict[str, Any]]:
    """Poll for unprocessed skill events and dispatch them.

    If ``agent_invoke`` is provided (a callable ``fn(prompt, user_id) -> str``),
    the event payload is forwarded to the agent for processing.

    Returns the list of processed events.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """SELECT * FROM skill_inbound_events
           WHERE processed = 0
           ORDER BY id ASC
           LIMIT ?""",
        (batch_size,),
    ).fetchall()

    processed: list[dict[str, Any]] = []
    for row in rows:
        event = dict(row)
        try:
            payload = json.loads(event.get("payload", "{}"))
        except (json.JSONDecodeError, TypeError):
            payload = {}

        if agent_invoke and event["event_type"] in ("message", "command"):
            prompt = payload.get("text", "") or payload.get("message", "")
            if prompt:
                try:
                    result = agent_invoke(prompt, user_id=user_id)
                    conn.execute(
                        "UPDATE skill_inbound_events SET processed=1, agent_invoked=1 WHERE id=?",
                        (event["id"],),
                    )
                    logger.info("Event %d dispatched to agent: %s", event["id"], str(result)[:100])
                except Exception as exc:
                    logger.error("Event %d agent invoke failed: %s", event["id"], exc)
                    conn.execute(
                        "UPDATE skill_inbound_events SET processed=1 WHERE id=?",
                        (event["id"],),
                    )
            else:
                conn.execute(
                    "UPDATE skill_inbound_events SET processed=1 WHERE id=?",
                    (event["id"],),
                )
        else:
            conn.execute(
                "UPDATE skill_inbound_events SET processed=1 WHERE id=?",
                (event["id"],),
            )

        processed.append(event)

    conn.commit()
    conn.close()
    return processed
