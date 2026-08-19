"""Journal DAO — moods, journal tables."""

from datetime import UTC, datetime


def _now():
    return datetime.now(UTC).isoformat()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS moods (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    mood        TEXT NOT NULL,
    note        TEXT NOT NULL DEFAULT '',
    timestamp   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS journal (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    entry       TEXT NOT NULL,
    timestamp   TEXT NOT NULL
);
"""


class JournalDAO:
    def __init__(self, conn, user_id: str):
        self._conn = conn
        self.user_id = user_id

    def save_journal(self, entry: str) -> dict:
        item = {"entry": entry.strip(), "timestamp": _now()}
        self._conn.execute(
            "INSERT INTO journal (user_id, entry, timestamp) VALUES (?,?,?)",
            (self.user_id, item["entry"], item["timestamp"]),
        )
        self._conn.commit()
        return item

    def get_journal(self, limit: int = 5) -> list:
        rows = self._conn.execute(
            "SELECT id, entry, timestamp FROM journal WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (self.user_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def delete_journal_entry(self, entry_id: int) -> bool:
        cur = self._conn.execute(
            "DELETE FROM journal WHERE user_id=? AND id=?",
            (self.user_id, int(entry_id)),
        )
        self._conn.commit()
        return cur.rowcount > 0
