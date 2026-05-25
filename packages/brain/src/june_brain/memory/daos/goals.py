"""Goals DAO — goals, open_loops tables."""

from datetime import UTC, datetime


def _now():
    return datetime.now(UTC).isoformat()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS goals (
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'personal',
    target_date TEXT NOT NULL DEFAULT '',
    next_step   TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'active',
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, title)
);
CREATE TABLE IF NOT EXISTS open_loops (
    user_id     TEXT NOT NULL,
    topic       TEXT NOT NULL,
    next_step   TEXT NOT NULL DEFAULT '',
    due_date    TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'open',
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, topic)
);
"""


class GoalDAO:
    def __init__(self, conn, user_id: str):
        self._conn = conn
        self.user_id = user_id

    def save_goal(
        self,
        title: str,
        category: str = "personal",
        target_date: str = "",
        next_step: str = "",
        status: str = "active",
    ) -> dict:
        item = {
            "title": title.strip(),
            "category": category.strip() or "personal",
            "target_date": target_date.strip(),
            "next_step": next_step.strip(),
            "status": status.strip() or "active",
            "updated_at": _now(),
        }
        self._conn.execute(
            """INSERT INTO goals (user_id,title,category,target_date,next_step,status,updated_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(user_id,title) DO UPDATE SET
                 category=excluded.category, target_date=excluded.target_date,
                 next_step=excluded.next_step, status=excluded.status,
                 updated_at=excluded.updated_at""",
            (self.user_id, item["title"], item["category"], item["target_date"],
             item["next_step"], item["status"], item["updated_at"]),
        )
        self._conn.commit()
        return item

    def update_goal_status(self, title: str, status: str) -> dict | None:
        now = _now()
        self._conn.execute(
            "UPDATE goals SET status=?, updated_at=? WHERE user_id=? AND lower(title)=lower(?)",
            (status.strip() or "active", now, self.user_id, title.strip()),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT title,category,target_date,next_step,status,updated_at "
            "FROM goals WHERE user_id=? AND lower(title)=lower(?)",
            (self.user_id, title.strip()),
        ).fetchone()
        return dict(row) if row else None

    def update_goal(self, old_title: str, **fields: str) -> dict | None:
        rows = self._conn.execute(
            "SELECT title,category,target_date,next_step,status,updated_at "
            "FROM goals WHERE user_id=? AND lower(title)=lower(?)",
            (self.user_id, old_title.strip()),
        ).fetchall()
        if not rows:
            return None
        existing = dict(rows[0])
        merged = {**existing, **{k: v for k, v in fields.items() if v is not None}}
        new_title = (merged.get("title") or "").strip() or existing["title"]
        if new_title.lower() != existing["title"].lower():
            self.delete_goal(existing["title"])
        return self.save_goal(
            title=new_title,
            category=merged.get("category", "personal"),
            target_date=merged.get("target_date", ""),
            next_step=merged.get("next_step", ""),
            status=merged.get("status", "active"),
        )

    def delete_goal(self, title: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM goals WHERE user_id=? AND lower(title)=lower(?)",
            (self.user_id, title.strip()),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get_goals(self, status: str = "", limit: int = 10) -> list:
        if status.strip():
            rows = self._conn.execute(
                "SELECT title,category,target_date,next_step,status,updated_at "
                "FROM goals WHERE user_id=? AND lower(status)=lower(?) ORDER BY rowid DESC LIMIT ?",
                (self.user_id, status.strip(), limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT title,category,target_date,next_step,status,updated_at "
                "FROM goals WHERE user_id=? ORDER BY rowid DESC LIMIT ?",
                (self.user_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def save_open_loop(
        self,
        topic: str,
        next_step: str = "",
        due_date: str = "",
        status: str = "open",
    ) -> dict:
        item = {
            "topic": topic.strip(),
            "next_step": next_step.strip(),
            "due_date": due_date.strip(),
            "status": status.strip() or "open",
            "updated_at": _now(),
        }
        self._conn.execute(
            """INSERT INTO open_loops (user_id,topic,next_step,due_date,status,updated_at)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(user_id,topic) DO UPDATE SET
                 next_step=excluded.next_step, due_date=excluded.due_date,
                 status=excluded.status, updated_at=excluded.updated_at""",
            (self.user_id, item["topic"], item["next_step"], item["due_date"],
             item["status"], item["updated_at"]),
        )
        self._conn.commit()
        return item

    def update_open_loop_status(self, topic: str, status: str) -> dict | None:
        now = _now()
        self._conn.execute(
            "UPDATE open_loops SET status=?, updated_at=? WHERE user_id=? AND lower(topic)=lower(?)",
            (status.strip() or "open", now, self.user_id, topic.strip()),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT topic,next_step,due_date,status,updated_at "
            "FROM open_loops WHERE user_id=? AND lower(topic)=lower(?)",
            (self.user_id, topic.strip()),
        ).fetchone()
        return dict(row) if row else None

    def update_open_loop(self, old_topic: str, **fields: str) -> dict | None:
        rows = self._conn.execute(
            "SELECT topic,next_step,due_date,status,updated_at "
            "FROM open_loops WHERE user_id=? AND lower(topic)=lower(?)",
            (self.user_id, old_topic.strip()),
        ).fetchall()
        if not rows:
            return None
        existing = dict(rows[0])
        merged = {**existing, **{k: v for k, v in fields.items() if v is not None}}
        new_topic = (merged.get("topic") or "").strip() or existing["topic"]
        if new_topic.lower() != existing["topic"].lower():
            self.delete_open_loop(existing["topic"])
        return self.save_open_loop(
            topic=new_topic,
            next_step=merged.get("next_step", ""),
            due_date=merged.get("due_date", ""),
            status=merged.get("status", "open"),
        )

    def delete_open_loop(self, topic: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM open_loops WHERE user_id=? AND lower(topic)=lower(?)",
            (self.user_id, topic.strip()),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get_open_loops(self, status: str = "open", limit: int = 10) -> list:
        if status.strip():
            rows = self._conn.execute(
                "SELECT topic,next_step,due_date,status,updated_at "
                "FROM open_loops WHERE user_id=? AND lower(status)=lower(?) ORDER BY rowid DESC LIMIT ?",
                (self.user_id, status.strip(), limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT topic,next_step,due_date,status,updated_at "
                "FROM open_loops WHERE user_id=? ORDER BY rowid DESC LIMIT ?",
                (self.user_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]
