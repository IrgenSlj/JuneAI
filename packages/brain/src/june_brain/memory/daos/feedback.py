"""Feedback DAO — memory_feedback table."""

from datetime import UTC, datetime


def _now():
    return datetime.now(UTC).isoformat()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_feedback (
    user_id     TEXT NOT NULL,
    ref         TEXT NOT NULL,
    vote        TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, ref)
);
"""


class FeedbackDAO:
    def __init__(self, conn, user_id: str):
        self._conn = conn
        self.user_id = user_id

    def set_feedback(self, ref: str, vote: str) -> dict | None:
        ref = ref.strip()
        vote = vote.strip().lower()
        if not ref or vote not in ("up", "down"):
            return None
        now = _now()
        self._conn.execute(
            """INSERT INTO memory_feedback (user_id, ref, vote, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id, ref) DO UPDATE SET
                 vote=excluded.vote, updated_at=excluded.updated_at""",
            (self.user_id, ref, vote, now),
        )
        self._conn.commit()
        return {"ref": ref, "vote": vote, "updated_at": now}

    def clear_feedback(self, ref: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM memory_feedback WHERE user_id=? AND ref=?",
            (self.user_id, ref.strip()),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get_feedback_map(self) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT ref, vote FROM memory_feedback WHERE user_id=?",
            (self.user_id,),
        ).fetchall()
        return {row["ref"]: row["vote"] for row in rows}
