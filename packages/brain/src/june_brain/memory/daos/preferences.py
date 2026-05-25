"""Preferences DAO — preferences, favorites tables."""

from datetime import UTC, datetime


def _now():
    return datetime.now(UTC).isoformat()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS preferences (
    user_id     TEXT NOT NULL,
    category    TEXT NOT NULL,
    value       TEXT NOT NULL,
    context     TEXT NOT NULL DEFAULT '',
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, category, value)
);
CREATE TABLE IF NOT EXISTS favorites (
    user_id     TEXT NOT NULL,
    category    TEXT NOT NULL,
    title       TEXT NOT NULL,
    reason      TEXT NOT NULL DEFAULT '',
    creator     TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'saved',
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, category, title)
);
"""


class PreferenceDAO:
    def __init__(self, conn, user_id: str):
        self._conn = conn
        self.user_id = user_id

    def save_preference(self, category: str, value: str, context: str = "") -> dict:
        item = {
            "category": category.strip(),
            "value": value.strip(),
            "context": context.strip(),
            "updated_at": _now(),
        }
        self._conn.execute(
            """INSERT INTO preferences (user_id,category,value,context,updated_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(user_id,category,value) DO UPDATE SET
                 context=excluded.context, updated_at=excluded.updated_at""",
            (self.user_id, item["category"], item["value"], item["context"], item["updated_at"]),
        )
        self._conn.commit()
        return item

    def get_preferences(self, category: str = "", limit: int = 20) -> list:
        if category.strip():
            rows = self._conn.execute(
                "SELECT category,value,context,updated_at FROM preferences "
                "WHERE user_id=? AND lower(category)=lower(?) ORDER BY rowid DESC LIMIT ?",
                (self.user_id, category.strip(), limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT category,value,context,updated_at FROM preferences "
                "WHERE user_id=? ORDER BY rowid DESC LIMIT ?",
                (self.user_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def save_favorite(
        self,
        category: str,
        title: str,
        reason: str = "",
        creator: str = "",
        status: str = "saved",
    ) -> dict:
        item = {
            "category": category.strip(),
            "title": title.strip(),
            "reason": reason.strip(),
            "creator": creator.strip(),
            "status": status.strip() or "saved",
            "updated_at": _now(),
        }
        self._conn.execute(
            """INSERT INTO favorites (user_id,category,title,reason,creator,status,updated_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(user_id,category,title) DO UPDATE SET
                 reason=excluded.reason, creator=excluded.creator,
                 status=excluded.status, updated_at=excluded.updated_at""",
            (self.user_id, item["category"], item["title"], item["reason"],
             item["creator"], item["status"], item["updated_at"]),
        )
        self._conn.commit()
        return item

    def get_favorites(self, category: str = "", limit: int = 20) -> list:
        if category.strip():
            rows = self._conn.execute(
                "SELECT category,title,reason,creator,status,updated_at FROM favorites "
                "WHERE user_id=? AND lower(category)=lower(?) ORDER BY rowid DESC LIMIT ?",
                (self.user_id, category.strip(), limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT category,title,reason,creator,status,updated_at FROM favorites "
                "WHERE user_id=? ORDER BY rowid DESC LIMIT ?",
                (self.user_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]
