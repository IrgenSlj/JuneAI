"""Chat DAO — chat_messages table."""

from __future__ import annotations

from datetime import UTC, datetime


def _now():
    return datetime.now(UTC).isoformat()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    timestamp   TEXT NOT NULL
);
"""


class ChatDAO:
    def __init__(self, conn, user_id: str):
        self._conn = conn
        self.user_id = user_id

    def save_message(self, role: str, content: str) -> None:
        conn = self._conn
        conn.execute(
            "INSERT INTO chat_messages (user_id, role, content, timestamp) VALUES (?,?,?,?)",
            (self.user_id, role, content, _now()),
        )
        conn.execute(
            """DELETE FROM chat_messages WHERE user_id=? AND id NOT IN (
                SELECT id FROM chat_messages WHERE user_id=? ORDER BY id DESC LIMIT 50
            )""",
            (self.user_id, self.user_id),
        )
        conn.commit()

    def load_chat(self) -> list:
        rows = self._conn.execute(
            "SELECT role, content, timestamp FROM chat_messages WHERE user_id=? ORDER BY id",
            (self.user_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def load_chat_messages(self) -> list:
        from langchain_core.messages import AIMessage, HumanMessage

        messages: list = []
        for item in self.load_chat():
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages
