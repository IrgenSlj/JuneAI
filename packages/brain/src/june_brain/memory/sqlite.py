"""JuneAI memory system — SQLite backend.

Single june.db per MEMORY_DIR. user_id is a column in every table.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4


def _current_memory_dir() -> str:
    """Resolve MEMORY_DIR lazily so tests that patch june_brain.memory.MEMORY_DIR win."""
    from . import MEMORY_DIR  # re-read package attribute each call
    return MEMORY_DIR

APP_STATE_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Connection pool — one SQLite connection per (thread, db_path)
# ---------------------------------------------------------------------------
_local = threading.local()


def _get_connection(db_path: str) -> sqlite3.Connection:
    conns = getattr(_local, "conns", None)
    if conns is None:
        _local.conns = {}
        conns = _local.conns
    if db_path not in conns:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conns[db_path] = conn
    return conns[db_path]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    timestamp   TEXT NOT NULL
);
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
CREATE TABLE IF NOT EXISTS relationships (
    user_id         TEXT NOT NULL,
    person          TEXT NOT NULL,
    relationship    TEXT NOT NULL DEFAULT '',
    summary         TEXT NOT NULL DEFAULT '',
    user_needs      TEXT NOT NULL DEFAULT '',
    cautions        TEXT NOT NULL DEFAULT '',
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (user_id, person)
);
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
CREATE TABLE IF NOT EXISTS preferences (
    user_id     TEXT NOT NULL,
    category    TEXT NOT NULL,
    value       TEXT NOT NULL,
    context     TEXT NOT NULL DEFAULT '',
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, category, value)
);
CREATE TABLE IF NOT EXISTS calendar_items (
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    date        TEXT NOT NULL DEFAULT '',
    time        TEXT NOT NULL DEFAULT '',
    details     TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'planned',
    source      TEXT NOT NULL DEFAULT 'conversation',
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, title, date, time)
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
CREATE TABLE IF NOT EXISTS gym_plans (
    user_id     TEXT NOT NULL,
    name        TEXT NOT NULL,
    schedule    TEXT NOT NULL DEFAULT '',
    goal        TEXT NOT NULL DEFAULT '',
    notes       TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'active',
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, name)
);
CREATE TABLE IF NOT EXISTS food_programs (
    user_id         TEXT NOT NULL,
    name            TEXT NOT NULL,
    goal            TEXT NOT NULL DEFAULT '',
    daily_structure TEXT NOT NULL DEFAULT '',
    notes           TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'active',
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (user_id, name)
);
CREATE TABLE IF NOT EXISTS workout_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,
    date            TEXT NOT NULL,
    plan_name       TEXT NOT NULL DEFAULT '',
    exercises       TEXT NOT NULL DEFAULT '',
    duration_min    INTEGER NOT NULL DEFAULT 0,
    notes           TEXT NOT NULL DEFAULT '',
    energy_rating   INTEGER NOT NULL DEFAULT 0,
    timestamp       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS body_metrics (
    user_id         TEXT NOT NULL,
    date            TEXT NOT NULL,
    weight_kg       REAL NOT NULL DEFAULT 0.0,
    sleep_hours     REAL NOT NULL DEFAULT 0.0,
    sleep_quality   INTEGER NOT NULL DEFAULT 0,
    energy          INTEGER NOT NULL DEFAULT 0,
    stress          INTEGER NOT NULL DEFAULT 0,
    soreness        INTEGER NOT NULL DEFAULT 0,
    resting_hr      INTEGER NOT NULL DEFAULT 0,
    steps           INTEGER NOT NULL DEFAULT 0,
    notes           TEXT NOT NULL DEFAULT '',
    timestamp       TEXT NOT NULL,
    PRIMARY KEY (user_id, date)
);
CREATE TABLE IF NOT EXISTS habits (
    user_id     TEXT NOT NULL,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'health',
    target_days TEXT NOT NULL DEFAULT 'daily',
    created_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, name)
);
CREATE TABLE IF NOT EXISTS habit_completions (
    user_id         TEXT NOT NULL,
    habit_name      TEXT NOT NULL,
    completion_date TEXT NOT NULL,
    PRIMARY KEY (user_id, habit_name, completion_date)
);
CREATE TABLE IF NOT EXISTS nutrition_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,
    date            TEXT NOT NULL,
    meal            TEXT NOT NULL DEFAULT '',
    description     TEXT NOT NULL DEFAULT '',
    calories_est    INTEGER NOT NULL DEFAULT 0,
    protein_est     INTEGER NOT NULL DEFAULT 0,
    notes           TEXT NOT NULL DEFAULT '',
    timestamp       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS water_logs (
    user_id TEXT NOT NULL,
    date    TEXT NOT NULL,
    glasses INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, date)
);
CREATE TABLE IF NOT EXISTS telemetry (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,
    schema_version  INTEGER NOT NULL DEFAULT 1,
    event_id        TEXT NOT NULL,
    event_type      TEXT NOT NULL DEFAULT 'event',
    name            TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'ok',
    source          TEXT NOT NULL DEFAULT 'memory',
    route           TEXT NOT NULL DEFAULT '',
    timestamp       TEXT NOT NULL,
    payload         TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS app_state (
    user_id TEXT NOT NULL,
    key     TEXT NOT NULL,
    value   TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
);
CREATE TABLE IF NOT EXISTS graph_nodes (
    user_id    TEXT NOT NULL,
    node_id    TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'entity',
    label      TEXT NOT NULL DEFAULT '',
    props      TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, node_id)
);
CREATE TABLE IF NOT EXISTS graph_edges (
    user_id    TEXT NOT NULL,
    src        TEXT NOT NULL,
    dst        TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'related_to',
    props      TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, src, dst, kind)
);
CREATE INDEX IF NOT EXISTS idx_graph_edges_src ON graph_edges(user_id, src);
CREATE INDEX IF NOT EXISTS idx_graph_edges_dst ON graph_edges(user_id, dst);
CREATE TABLE IF NOT EXISTS semantic_facts (
    user_id     TEXT NOT NULL,
    fact_id     TEXT NOT NULL,
    text        TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'conversation',
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, fact_id)
);
CREATE INDEX IF NOT EXISTS idx_semantic_facts_created ON semantic_facts(user_id, created_at);
"""


def _init_schema(conn: sqlite3.Connection) -> None:
    for stmt in _SCHEMA_SQL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()


# ---------------------------------------------------------------------------
# Memory class
# ---------------------------------------------------------------------------

class Memory:
    """Manages all persistent storage for a single user (SQLite backend)."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        db_dir = Path(_current_memory_dir())
        db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_dir / "june.db")
        conn = _get_connection(self._db_path)
        _init_schema(conn)
        self._migrate_from_json(db_dir)

    @property
    def _conn(self) -> sqlite3.Connection:
        return _get_connection(self._db_path)

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------

    def save_message(self, role: str, content: str) -> None:
        conn = self._conn
        conn.execute(
            "INSERT INTO chat_messages (user_id, role, content, timestamp) VALUES (?,?,?,?)",
            (self.user_id, role, content, self._now()),
        )
        # Keep latest 50 — delete older ones
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
        messages = []
        for item in self.load_chat():
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    # ------------------------------------------------------------------
    # Mood
    # ------------------------------------------------------------------

    def log_mood(self, mood: str, note: str = "") -> dict:
        entry = {"mood": mood.strip(), "note": note.strip(), "timestamp": self._now()}
        self._conn.execute(
            "INSERT INTO moods (user_id, mood, note, timestamp) VALUES (?,?,?,?)",
            (self.user_id, entry["mood"], entry["note"], entry["timestamp"]),
        )
        self._conn.commit()
        return entry

    def get_mood_history(self, limit: int = 10) -> list:
        rows = self._conn.execute(
            "SELECT mood, note, timestamp FROM moods WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (self.user_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ------------------------------------------------------------------
    # Journal
    # ------------------------------------------------------------------

    def save_journal(self, entry: str) -> dict:
        item = {"entry": entry.strip(), "timestamp": self._now()}
        self._conn.execute(
            "INSERT INTO journal (user_id, entry, timestamp) VALUES (?,?,?)",
            (self.user_id, item["entry"], item["timestamp"]),
        )
        self._conn.commit()
        return item

    def get_journal(self, limit: int = 5) -> list:
        rows = self._conn.execute(
            "SELECT entry, timestamp FROM journal WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (self.user_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def save_relationship_profile(
        self,
        person: str,
        relationship: str,
        summary: str,
        user_needs: str = "",
        cautions: str = "",
    ) -> dict:
        item = {
            "person": person.strip(),
            "relationship": relationship.strip(),
            "summary": summary.strip(),
            "user_needs": user_needs.strip(),
            "cautions": cautions.strip(),
            "updated_at": self._now(),
        }
        self._conn.execute(
            """INSERT INTO relationships (user_id,person,relationship,summary,user_needs,cautions,updated_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(user_id,person) DO UPDATE SET
                 relationship=excluded.relationship, summary=excluded.summary,
                 user_needs=excluded.user_needs, cautions=excluded.cautions,
                 updated_at=excluded.updated_at""",
            (self.user_id, item["person"], item["relationship"], item["summary"],
             item["user_needs"], item["cautions"], item["updated_at"]),
        )
        self._conn.commit()
        return item

    def get_relationship_profiles(self, person: str = "") -> list:
        if person.strip():
            rows = self._conn.execute(
                "SELECT person,relationship,summary,user_needs,cautions,updated_at "
                "FROM relationships WHERE user_id=? AND lower(person)=lower(?)",
                (self.user_id, person.strip()),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT person,relationship,summary,user_needs,cautions,updated_at "
                "FROM relationships WHERE user_id=?",
                (self.user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Goals
    # ------------------------------------------------------------------

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
            "updated_at": self._now(),
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
        now = self._now()
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

    # ------------------------------------------------------------------
    # Open loops
    # ------------------------------------------------------------------

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
            "updated_at": self._now(),
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
        now = self._now()
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

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    def save_preference(self, category: str, value: str, context: str = "") -> dict:
        item = {
            "category": category.strip(),
            "value": value.strip(),
            "context": context.strip(),
            "updated_at": self._now(),
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

    # ------------------------------------------------------------------
    # Calendar
    # ------------------------------------------------------------------

    def save_calendar_item(
        self,
        title: str,
        date: str,
        time: str = "",
        details: str = "",
        status: str = "planned",
        source: str = "conversation",
    ) -> dict:
        item = {
            "title": title.strip(),
            "date": date.strip(),
            "time": time.strip(),
            "details": details.strip(),
            "status": status.strip() or "planned",
            "source": source.strip() or "conversation",
            "updated_at": self._now(),
        }
        self._conn.execute(
            """INSERT INTO calendar_items (user_id,title,date,time,details,status,source,updated_at)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(user_id,title,date,time) DO UPDATE SET
                 details=excluded.details, status=excluded.status,
                 source=excluded.source, updated_at=excluded.updated_at""",
            (self.user_id, item["title"], item["date"], item["time"],
             item["details"], item["status"], item["source"], item["updated_at"]),
        )
        self._conn.commit()
        return item

    def update_calendar_item_status(
        self,
        title: str,
        status: str,
        date: str = "",
        time: str = "",
    ) -> dict | None:
        now = self._now()
        query = "UPDATE calendar_items SET status=?, updated_at=? WHERE user_id=? AND lower(title)=lower(?)"
        params: list = [status.strip() or "planned", now, self.user_id, title.strip()]
        if date.strip():
            query += " AND lower(date)=lower(?)"
            params.append(date.strip())
        if time.strip():
            query += " AND lower(time)=lower(?)"
            params.append(time.strip())
        self._conn.execute(query, params)
        self._conn.commit()
        row = self._conn.execute(
            "SELECT title,date,time,details,status,source,updated_at FROM calendar_items "
            "WHERE user_id=? AND lower(title)=lower(?)",
            (self.user_id, title.strip()),
        ).fetchone()
        return dict(row) if row else None

    def delete_calendar_item(self, title: str, date: str = "", time: str = "") -> bool:
        query = "DELETE FROM calendar_items WHERE user_id=? AND lower(title)=lower(?)"
        params: list = [self.user_id, title.strip()]
        if date.strip():
            query += " AND lower(date)=lower(?)"
            params.append(date.strip())
        if time.strip():
            query += " AND lower(time)=lower(?)"
            params.append(time.strip())
        cur = self._conn.execute(query, params)
        self._conn.commit()
        return cur.rowcount > 0

    def get_calendar_items(self, status: str = "", limit: int = 20) -> list:
        if status.strip():
            rows = self._conn.execute(
                "SELECT title,date,time,details,status,source,updated_at FROM calendar_items "
                "WHERE user_id=? AND lower(status)=lower(?)",
                (self.user_id, status.strip()),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT title,date,time,details,status,source,updated_at FROM calendar_items "
                "WHERE user_id=?",
                (self.user_id,),
            ).fetchall()
        items = [dict(r) for r in rows]
        # Same proximity sort as before
        today = _today()
        dated, undated = [], []
        for item in items:
            parsed = _parse_date(item.get("date", ""))
            if parsed is None:
                undated.append(item)
            else:
                dated.append((parsed, item))
        dated.sort(key=lambda e: (abs((e[0] - today).days), e[0], e[1].get("time", ""), e[1].get("title", "")))
        undated.sort(key=lambda i: (i.get("title", ""), i.get("updated_at", "")))
        return ([i for _, i in dated] + undated)[:limit]

    # ------------------------------------------------------------------
    # Favorites
    # ------------------------------------------------------------------

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
            "updated_at": self._now(),
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

    # ------------------------------------------------------------------
    # Wellness plans
    # ------------------------------------------------------------------

    def save_gym_plan(
        self,
        name: str,
        schedule: str,
        goal: str = "",
        notes: str = "",
        status: str = "active",
    ) -> dict:
        item = {
            "name": name.strip(),
            "schedule": schedule.strip(),
            "goal": goal.strip(),
            "notes": notes.strip(),
            "status": status.strip() or "active",
            "updated_at": self._now(),
        }
        self._conn.execute(
            """INSERT INTO gym_plans (user_id,name,schedule,goal,notes,status,updated_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(user_id,name) DO UPDATE SET
                 schedule=excluded.schedule, goal=excluded.goal,
                 notes=excluded.notes, status=excluded.status,
                 updated_at=excluded.updated_at""",
            (self.user_id, item["name"], item["schedule"], item["goal"],
             item["notes"], item["status"], item["updated_at"]),
        )
        self._conn.commit()
        return item

    def get_gym_plans(self, status: str = "", limit: int = 10) -> list:
        if status.strip():
            rows = self._conn.execute(
                "SELECT name,schedule,goal,notes,status,updated_at FROM gym_plans "
                "WHERE user_id=? AND lower(status)=lower(?) ORDER BY rowid DESC LIMIT ?",
                (self.user_id, status.strip(), limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT name,schedule,goal,notes,status,updated_at FROM gym_plans "
                "WHERE user_id=? ORDER BY rowid DESC LIMIT ?",
                (self.user_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def save_food_program(
        self,
        name: str,
        goal: str,
        daily_structure: str,
        notes: str = "",
        status: str = "active",
    ) -> dict:
        item = {
            "name": name.strip(),
            "goal": goal.strip(),
            "daily_structure": daily_structure.strip(),
            "notes": notes.strip(),
            "status": status.strip() or "active",
            "updated_at": self._now(),
        }
        self._conn.execute(
            """INSERT INTO food_programs (user_id,name,goal,daily_structure,notes,status,updated_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(user_id,name) DO UPDATE SET
                 goal=excluded.goal, daily_structure=excluded.daily_structure,
                 notes=excluded.notes, status=excluded.status,
                 updated_at=excluded.updated_at""",
            (self.user_id, item["name"], item["goal"], item["daily_structure"],
             item["notes"], item["status"], item["updated_at"]),
        )
        self._conn.commit()
        return item

    def get_food_programs(self, status: str = "", limit: int = 10) -> list:
        if status.strip():
            rows = self._conn.execute(
                "SELECT name,goal,daily_structure,notes,status,updated_at FROM food_programs "
                "WHERE user_id=? AND lower(status)=lower(?) ORDER BY rowid DESC LIMIT ?",
                (self.user_id, status.strip(), limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT name,goal,daily_structure,notes,status,updated_at FROM food_programs "
                "WHERE user_id=? ORDER BY rowid DESC LIMIT ?",
                (self.user_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ------------------------------------------------------------------
    # Workout sessions
    # ------------------------------------------------------------------

    def log_workout_session(
        self,
        plan_name: str,
        exercises: str = "",
        duration_min: int = 0,
        notes: str = "",
        energy_rating: int = 0,
    ) -> dict:
        item = {
            "date": _today().isoformat(),
            "plan_name": plan_name.strip(),
            "exercises": exercises.strip(),
            "duration_min": max(0, int(duration_min)),
            "notes": notes.strip(),
            "energy_rating": max(0, min(5, int(energy_rating))),
            "timestamp": self._now(),
        }
        self._conn.execute(
            """INSERT INTO workout_sessions
               (user_id,date,plan_name,exercises,duration_min,notes,energy_rating,timestamp)
               VALUES (?,?,?,?,?,?,?,?)""",
            (self.user_id, item["date"], item["plan_name"], item["exercises"],
             item["duration_min"], item["notes"], item["energy_rating"], item["timestamp"]),
        )
        self._conn.commit()
        return item

    def get_workout_sessions(self, limit: int = 10) -> list:
        rows = self._conn.execute(
            "SELECT date,plan_name,exercises,duration_min,notes,energy_rating,timestamp "
            "FROM workout_sessions WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (self.user_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_today_workout(self) -> dict | None:
        row = self._conn.execute(
            "SELECT date,plan_name,exercises,duration_min,notes,energy_rating,timestamp "
            "FROM workout_sessions WHERE user_id=? AND date=? ORDER BY id DESC LIMIT 1",
            (self.user_id, _today().isoformat()),
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Body metrics
    # ------------------------------------------------------------------

    def log_body_metrics(
        self,
        weight_kg: float = 0.0,
        sleep_hours: float = 0.0,
        sleep_quality: int = 0,
        energy: int = 0,
        stress: int = 0,
        soreness: int = 0,
        resting_hr: int = 0,
        steps: int = 0,
        notes: str = "",
    ) -> dict:
        today = _today().isoformat()
        item = {
            "date": today,
            "weight_kg": round(float(weight_kg), 1) if weight_kg else 0.0,
            "sleep_hours": round(float(sleep_hours), 1) if sleep_hours else 0.0,
            "sleep_quality": max(0, min(5, int(sleep_quality))),
            "energy": max(0, min(5, int(energy))),
            "stress": max(0, min(5, int(stress))),
            "soreness": max(0, min(5, int(soreness))),
            "resting_hr": max(0, int(resting_hr)),
            "steps": max(0, int(steps)),
            "notes": notes.strip(),
            "timestamp": self._now(),
        }
        self._conn.execute(
            """INSERT INTO body_metrics
               (user_id,date,weight_kg,sleep_hours,sleep_quality,energy,stress,
                soreness,resting_hr,steps,notes,timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(user_id,date) DO UPDATE SET
                 weight_kg=excluded.weight_kg, sleep_hours=excluded.sleep_hours,
                 sleep_quality=excluded.sleep_quality, energy=excluded.energy,
                 stress=excluded.stress, soreness=excluded.soreness,
                 resting_hr=excluded.resting_hr, steps=excluded.steps,
                 notes=excluded.notes, timestamp=excluded.timestamp""",
            (self.user_id, item["date"], item["weight_kg"], item["sleep_hours"],
             item["sleep_quality"], item["energy"], item["stress"], item["soreness"],
             item["resting_hr"], item["steps"], item["notes"], item["timestamp"]),
        )
        self._conn.commit()
        return item

    def get_body_metrics(self, days: int = 30) -> list:
        rows = self._conn.execute(
            "SELECT date,weight_kg,sleep_hours,sleep_quality,energy,stress,"
            "soreness,resting_hr,steps,notes,timestamp "
            "FROM body_metrics WHERE user_id=? ORDER BY date DESC LIMIT ?",
            (self.user_id, days),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_today_body_metrics(self) -> dict | None:
        row = self._conn.execute(
            "SELECT date,weight_kg,sleep_hours,sleep_quality,energy,stress,"
            "soreness,resting_hr,steps,notes,timestamp "
            "FROM body_metrics WHERE user_id=? AND date=?",
            (self.user_id, _today().isoformat()),
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Habits
    # ------------------------------------------------------------------

    def create_or_update_habit(
        self,
        name: str,
        category: str = "health",
        target_days: str = "daily",
    ) -> dict:
        now = self._now()
        self._conn.execute(
            """INSERT INTO habits (user_id,name,category,target_days,created_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(user_id,name) DO UPDATE SET
                 category=excluded.category, target_days=excluded.target_days""",
            (self.user_id, name.strip(), category.strip() or "health",
             target_days.strip() or "daily", now),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT name,category,target_days,created_at FROM habits WHERE user_id=? AND lower(name)=lower(?)",
            (self.user_id, name.strip()),
        ).fetchone()
        return self._habit_snapshot(dict(row))

    def log_habit_completion(self, habit_name: str, date_str: str = "") -> dict:
        target_date = date_str.strip() if date_str.strip() else _today().isoformat()
        name = habit_name.strip()
        # Auto-create if missing
        self._conn.execute(
            "INSERT OR IGNORE INTO habits (user_id,name,category,target_days,created_at) VALUES (?,?,?,?,?)",
            (self.user_id, name, "health", "daily", self._now()),
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO habit_completions (user_id,habit_name,completion_date) VALUES (?,?,?)",
            (self.user_id, name, target_date),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT name,category,target_days,created_at FROM habits WHERE user_id=? AND lower(name)=lower(?)",
            (self.user_id, name),
        ).fetchone()
        return self._habit_snapshot(dict(row))

    def get_habits(self) -> list:
        rows = self._conn.execute(
            "SELECT name,category,target_days,created_at FROM habits WHERE user_id=?",
            (self.user_id,),
        ).fetchall()
        result = []
        for row in rows:
            result.append(self._habit_snapshot(dict(row)))
        return result

    # ------------------------------------------------------------------
    # Nutrition
    # ------------------------------------------------------------------

    def log_nutrition(
        self,
        meal: str,
        description: str,
        calories_est: int = 0,
        protein_est: int = 0,
        notes: str = "",
    ) -> dict:
        item = {
            "date": _today().isoformat(),
            "meal": meal.strip().lower(),
            "description": description.strip(),
            "calories_est": max(0, int(calories_est)),
            "protein_est": max(0, int(protein_est)),
            "notes": notes.strip(),
            "timestamp": self._now(),
        }
        self._conn.execute(
            """INSERT INTO nutrition_logs
               (user_id,date,meal,description,calories_est,protein_est,notes,timestamp)
               VALUES (?,?,?,?,?,?,?,?)""",
            (self.user_id, item["date"], item["meal"], item["description"],
             item["calories_est"], item["protein_est"], item["notes"], item["timestamp"]),
        )
        self._conn.commit()
        return item

    def get_nutrition_today(self) -> list:
        rows = self._conn.execute(
            "SELECT date,meal,description,calories_est,protein_est,notes,timestamp "
            "FROM nutrition_logs WHERE user_id=? AND date=? ORDER BY id",
            (self.user_id, _today().isoformat()),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_nutrition_recent(self, limit: int = 28) -> list:
        rows = self._conn.execute(
            "SELECT date,meal,description,calories_est,protein_est,notes,timestamp "
            "FROM nutrition_logs WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (self.user_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ------------------------------------------------------------------
    # Water
    # ------------------------------------------------------------------

    def log_water(self, glasses: int = 1) -> int:
        today = _today().isoformat()
        self._conn.execute(
            """INSERT INTO water_logs (user_id,date,glasses) VALUES (?,?,?)
               ON CONFLICT(user_id,date) DO UPDATE SET glasses=glasses+?""",
            (self.user_id, today, max(0, int(glasses)), max(0, int(glasses))),
        )
        self._conn.commit()
        return self.get_water_today()

    def set_water(self, glasses: int) -> int:
        today = _today().isoformat()
        self._conn.execute(
            """INSERT INTO water_logs (user_id,date,glasses) VALUES (?,?,?)
               ON CONFLICT(user_id,date) DO UPDATE SET glasses=?""",
            (self.user_id, today, max(0, int(glasses)), max(0, int(glasses))),
        )
        self._conn.commit()
        return self.get_water_today()

    def get_water_today(self) -> int:
        row = self._conn.execute(
            "SELECT glasses FROM water_logs WHERE user_id=? AND date=?",
            (self.user_id, _today().isoformat()),
        ).fetchone()
        return int(row["glasses"]) if row else 0

    # ------------------------------------------------------------------
    # Chapter completeness
    # ------------------------------------------------------------------

    def get_chapter_completeness(self) -> dict:
        calendar_all = self.get_calendar_items(limit=200)

        def _text(item: dict) -> str:
            return " ".join(str(item.get(f, "")).lower() for f in ("title", "details", "source"))

        return {
            "calendar": len(calendar_all),
            "birthdays": len([i for i in calendar_all if "birthday" in _text(i)]),
            "trips": len([i for i in calendar_all if any(k in _text(i) for k in ("trip", "travel", "flight"))]),
            "gym": len(self.get_gym_plans(limit=100)),
            "food": len(self.get_food_programs(limit=100)),
            "goals": len(self.get_goals(status="", limit=100)),
            "open_loops": len(self.get_open_loops(status="", limit=100)),
            "dating": len([
                p for p in self.get_relationship_profiles()
                if any(t in p.get("relationship", "").lower()
                       for t in ("dating", "love", "partner", "girlfriend", "boyfriend", "romantic", "spouse"))
            ]),
            "family": len([
                p for p in self.get_relationship_profiles()
                if any(t in p.get("relationship", "").lower()
                       for t in ("family", "mother", "father", "mom", "dad", "brother",
                                 "sister", "parent", "child", "cousin", "uncle", "aunt"))
            ]),
            "habits": len(self.get_habits()),
            "body_metrics": len(self.get_body_metrics(days=30)),
            "workout_sessions": len(self.get_workout_sessions(limit=50)),
        }

    def get_chapters_needing_attention(self) -> list:
        c = self.get_chapter_completeness()
        chapter_checks = [
            ("habits",     "Habits",         c["habits"]),
            ("gym",        "Gym Schedule",   c["gym"]),
            ("food",       "Food Schedule",  c["food"]),
            ("calendar",   "Calendar",       c["calendar"]),
            ("birthdays",  "Birthdays",      c["birthdays"]),
            ("family",     "Family",         c["family"]),
            ("plans",      "Goals & Plans",  c["goals"] + c["open_loops"]),
            ("trips",      "Trips",          c["trips"]),
            ("dating",     "Dating/Love",    c["dating"]),
        ]
        return [label for _, label, count in chapter_checks if count == 0]

    # ------------------------------------------------------------------
    # Today summary
    # ------------------------------------------------------------------

    def get_today_summary(self) -> dict:
        today_metrics = self.get_today_body_metrics()
        today_workout = self.get_today_workout()
        today_nutrition = self.get_nutrition_today()
        habits = self.get_habits()
        water = self.get_water_today()
        habits_done = [h for h in habits if h.get("done_today")]
        habits_pending = [h for h in habits if not h.get("done_today")]
        return {
            "date": _today().isoformat(),
            "body_metrics": today_metrics,
            "recent_body_metrics": self.get_body_metrics(days=7),
            "workout": today_workout,
            "habits_total": len(habits),
            "habits_done": len(habits_done),
            "habits_done_names": [h["name"] for h in habits_done],
            "habits_pending_names": [h["name"] for h in habits_pending],
            "water_glasses": water,
            "meals_logged": len(today_nutrition),
            "calories_est": sum(e.get("calories_est", 0) for e in today_nutrition),
            "protein_est": sum(e.get("protein_est", 0) for e in today_nutrition),
        }

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def append_event(
        self,
        event_type: str,
        name: str = "",
        status: str = "ok",
        source: str = "memory",
        route: str = "",
        payload: Mapping[str, object] | None = None,
    ) -> dict:
        event = {
            "schema_version": 1,
            "event_id": uuid4().hex,
            "event_type": event_type.strip() or "event",
            "name": name.strip(),
            "status": status.strip() or "ok",
            "source": source.strip() or "memory",
            "route": route.strip(),
            "timestamp": self._now(),
            "payload": self._json_safe_value(payload or {}),
        }
        self._conn.execute(
            """INSERT INTO telemetry
               (user_id,schema_version,event_id,event_type,name,status,source,route,timestamp,payload)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (self.user_id, 1, event["event_id"], event["event_type"], event["name"],
             event["status"], event["source"], event["route"], event["timestamp"],
             json.dumps(event["payload"])),
        )
        self._conn.commit()
        return event

    def record_tool_call(
        self,
        tool_name: str,
        status: str = "started",
        source: str = "graph",
        route: str = "",
        payload: Mapping[str, object] | None = None,
    ) -> dict:
        details = dict(payload or {})
        details.setdefault("tool_name", tool_name.strip())
        return self.append_event(event_type="tool_call", name=tool_name, status=status,
                                 source=source, route=route, payload=details)

    def record_route_selection(
        self,
        route: str,
        source: str = "graph",
        payload: Mapping[str, object] | None = None,
    ) -> dict:
        details = dict(payload or {})
        details.setdefault("route", route.strip())
        return self.append_event(event_type="route_selection", name=route, status="selected",
                                 source=source, route=route, payload=details)

    def record_save_event(
        self,
        kind: str,
        name: str,
        status: str = "saved",
        source: str = "memory",
        route: str = "",
        payload: Mapping[str, object] | None = None,
    ) -> dict:
        details = dict(payload or {})
        details.setdefault("kind", kind.strip())
        return self.append_event(event_type="save_event", name=name, status=status,
                                 source=source, route=route, payload=details)

    def get_recent_events(self, limit: int = 20, event_type: str = "") -> list:
        if event_type.strip():
            rows = self._conn.execute(
                "SELECT schema_version,event_id,event_type,name,status,source,route,timestamp,payload "
                "FROM telemetry WHERE user_id=? AND lower(event_type)=lower(?) ORDER BY id DESC LIMIT ?",
                (self.user_id, event_type.strip(), limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT schema_version,event_id,event_type,name,status,source,route,timestamp,payload "
                "FROM telemetry WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (self.user_id, limit),
            ).fetchall()
        result = []
        for r in reversed(rows):
            d = dict(r)
            try:
                d["payload"] = json.loads(d["payload"])
            except (json.JSONDecodeError, TypeError):
                d["payload"] = {}
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # App state
    # ------------------------------------------------------------------

    def get_app_state(self) -> dict:
        rows = self._conn.execute(
            "SELECT key, value FROM app_state WHERE user_id=?",
            (self.user_id,),
        ).fetchall()
        result = {}
        for row in rows:
            try:
                result[row["key"]] = json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                result[row["key"]] = row["value"]
        return result

    def set_app_state_value(self, key: str, value) -> dict:
        self._conn.execute(
            """INSERT INTO app_state (user_id,key,value) VALUES (?,?,?)
               ON CONFLICT(user_id,key) DO UPDATE SET value=excluded.value""",
            (self.user_id, key, json.dumps(value)),
        )
        self._conn.commit()
        return self.get_app_state()

    def should_send_daily_checkin(self) -> bool:
        state = self.get_app_state()
        return state.get("last_daily_checkin_date") != _today().isoformat()

    def mark_daily_checkin_sent(self) -> None:
        self.set_app_state_value("last_daily_checkin_date", _today().isoformat())

    def get_upcoming_notifications(self, limit: int = 5) -> list[dict]:
        today = _today()
        notifications = []
        for item in self.get_calendar_items(status="", limit=50):
            parsed = _parse_date(item.get("date", ""))
            if parsed is None:
                continue
            status = item.get("status", "").strip().lower()
            if status in {"completed", "done", "canceled", "cancelled", "archived"}:
                continue
            days_until = (parsed - today).days
            if -1 <= days_until <= 14:
                notifications.append({
                    "title": item.get("title", "Event"),
                    "kind": self._infer_calendar_kind(item),
                    "when": item.get("date", ""),
                    "details": item.get("details", ""),
                    "days_until": days_until,
                })
        for loop in self.get_open_loops(status="open", limit=20):
            parsed = _parse_date(loop.get("due_date", ""))
            if parsed is None:
                continue
            days_until = (parsed - today).days
            if -1 <= days_until <= 14:
                notifications.append({
                    "title": loop.get("topic", "Open loop"),
                    "kind": "plan",
                    "when": loop.get("due_date", ""),
                    "details": loop.get("next_step", ""),
                    "days_until": days_until,
                })
        notifications.sort(key=lambda i: (i["days_until"], i["when"], i["title"]))
        return notifications[:limit]

    # ------------------------------------------------------------------
    # Progress snapshot
    # ------------------------------------------------------------------

    def get_progress_snapshot(self) -> dict:
        moods = self.get_mood_history(10)
        journal = self.get_journal(5)
        goals = self.get_goals(limit=20)
        open_loops = self.get_open_loops(status="", limit=20)
        relationships = self.get_relationship_profiles()
        preferences = self.get_preferences(limit=50)
        calendar_items = self.get_calendar_items(limit=50)
        favorites = self.get_favorites(limit=50)
        gym_plans = self.get_gym_plans(limit=20)
        food_programs = self.get_food_programs(limit=20)
        habits = self.get_habits()
        active_goals = [g for g in goals if g.get("status") == "active"]
        unresolved = [loop for loop in open_loops if loop.get("status", "open") == "open"]
        return {
            "mood_count": len(moods),
            "latest_mood": moods[-1]["mood"] if moods else "",
            "journal_count": len(journal),
            "relationship_count": len(relationships),
            "goal_count": len(goals),
            "active_goal_count": len(active_goals),
            "open_loop_count": len(unresolved),
            "preference_count": len(preferences),
            "calendar_count": len(calendar_items),
            "favorite_count": len(favorites),
            "gym_plan_count": len(gym_plans),
            "food_program_count": len(food_programs),
            "habit_count": len(habits),
            "habits_done_today": len([h for h in habits if h.get("done_today")]),
            "water_today": self.get_water_today(),
            "workout_session_count": len(self.get_workout_sessions(limit=50)),
        }

    # ------------------------------------------------------------------
    # JSON migration (one-time, on first init with existing data)
    # ------------------------------------------------------------------

    def _migrate_from_json(self, db_dir: Path) -> None:
        """Import any legacy per-user JSON files into SQLite, then rename them."""
        uid = self.user_id
        file_map = {
            "chat":             db_dir / f"{uid}_chat.json",
            "moods":            db_dir / f"{uid}_moods.json",
            "journal":          db_dir / f"{uid}_journal.json",
            "relationships":    db_dir / f"{uid}_relationships.json",
            "goals":            db_dir / f"{uid}_goals.json",
            "open_loops":       db_dir / f"{uid}_open_loops.json",
            "preferences":      db_dir / f"{uid}_preferences.json",
            "calendar":         db_dir / f"{uid}_calendar.json",
            "favorites":        db_dir / f"{uid}_favorites.json",
            "gym":              db_dir / f"{uid}_gym_plans.json",
            "food":             db_dir / f"{uid}_food_programs.json",
            "workouts":         db_dir / f"{uid}_workout_sessions.json",
            "body":             db_dir / f"{uid}_body_metrics.json",
            "habits":           db_dir / f"{uid}_habits.json",
            "nutrition":        db_dir / f"{uid}_nutrition_logs.json",
            "water":            db_dir / f"{uid}_water_logs.json",
            "telemetry":        db_dir / f"{uid}_telemetry.json",
            "app_state":        db_dir / f"{uid}_app_state.json",
        }

        migrated_any = False
        for key, path in file_map.items():
            if not path.exists():
                continue
            try:
                raw = path.read_text(encoding="utf-8")
                data = json.loads(raw) if raw.strip() else None
            except Exception:
                data = None
            if data is not None:
                try:
                    self._import_json_table(key, data)
                    migrated_any = True
                except Exception as exc:
                    logging.warning("JuneAI migration: failed to import %s: %s", key, exc)
            path.rename(path.with_suffix(".json.migrated"))

        if migrated_any:
            logging.info("JuneAI: migrated JSON memory to SQLite for user '%s'", uid)

    def _import_json_table(self, key: str, data) -> None:
        conn = self._conn
        uid = self.user_id

        if key == "chat" and isinstance(data, list):
            for m in data[-50:]:
                conn.execute(
                    "INSERT OR IGNORE INTO chat_messages (user_id,role,content,timestamp) VALUES (?,?,?,?)",
                    (uid, m.get("role",""), m.get("content",""), m.get("timestamp", self._now())),
                )
        elif key == "moods" and isinstance(data, list):
            for m in data:
                conn.execute(
                    "INSERT OR IGNORE INTO moods (user_id,mood,note,timestamp) VALUES (?,?,?,?)",
                    (uid, m.get("mood",""), m.get("note",""), m.get("timestamp", self._now())),
                )
        elif key == "journal" and isinstance(data, list):
            for m in data:
                conn.execute(
                    "INSERT OR IGNORE INTO journal (user_id,entry,timestamp) VALUES (?,?,?)",
                    (uid, m.get("entry",""), m.get("timestamp", self._now())),
                )
        elif key == "relationships" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO relationships
                       (user_id,person,relationship,summary,user_needs,cautions,updated_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (uid, m.get("person",""), m.get("relationship",""), m.get("summary",""),
                     m.get("user_needs",""), m.get("cautions",""), m.get("updated_at", self._now())),
                )
        elif key == "goals" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO goals
                       (user_id,title,category,target_date,next_step,status,updated_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (uid, m.get("title",""), m.get("category","personal"), m.get("target_date",""),
                     m.get("next_step",""), m.get("status","active"), m.get("updated_at", self._now())),
                )
        elif key == "open_loops" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO open_loops
                       (user_id,topic,next_step,due_date,status,updated_at) VALUES (?,?,?,?,?,?)""",
                    (uid, m.get("topic",""), m.get("next_step",""), m.get("due_date",""),
                     m.get("status","open"), m.get("updated_at", self._now())),
                )
        elif key == "preferences" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO preferences
                       (user_id,category,value,context,updated_at) VALUES (?,?,?,?,?)""",
                    (uid, m.get("category",""), m.get("value",""), m.get("context",""),
                     m.get("updated_at", self._now())),
                )
        elif key == "calendar" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO calendar_items
                       (user_id,title,date,time,details,status,source,updated_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (uid, m.get("title",""), m.get("date",""), m.get("time",""),
                     m.get("details",""), m.get("status","planned"), m.get("source","conversation"),
                     m.get("updated_at", self._now())),
                )
        elif key == "favorites" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO favorites
                       (user_id,category,title,reason,creator,status,updated_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (uid, m.get("category",""), m.get("title",""), m.get("reason",""),
                     m.get("creator",""), m.get("status","saved"), m.get("updated_at", self._now())),
                )
        elif key == "gym" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO gym_plans
                       (user_id,name,schedule,goal,notes,status,updated_at) VALUES (?,?,?,?,?,?,?)""",
                    (uid, m.get("name",""), m.get("schedule",""), m.get("goal",""),
                     m.get("notes",""), m.get("status","active"), m.get("updated_at", self._now())),
                )
        elif key == "food" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO food_programs
                       (user_id,name,goal,daily_structure,notes,status,updated_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (uid, m.get("name",""), m.get("goal",""), m.get("daily_structure",""),
                     m.get("notes",""), m.get("status","active"), m.get("updated_at", self._now())),
                )
        elif key == "workouts" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR IGNORE INTO workout_sessions
                       (user_id,date,plan_name,exercises,duration_min,notes,energy_rating,timestamp)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (uid, m.get("date",""), m.get("plan_name",""), m.get("exercises",""),
                     m.get("duration_min",0), m.get("notes",""), m.get("energy_rating",0),
                     m.get("timestamp", self._now())),
                )
        elif key == "body" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO body_metrics
                       (user_id,date,weight_kg,sleep_hours,sleep_quality,energy,stress,
                        soreness,resting_hr,steps,notes,timestamp)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (uid, m.get("date",""), m.get("weight_kg",0.0), m.get("sleep_hours",0.0),
                     m.get("sleep_quality",0), m.get("energy",0), m.get("stress",0),
                     m.get("soreness",0), m.get("resting_hr",0), m.get("steps",0),
                     m.get("notes",""), m.get("timestamp", self._now())),
                )
        elif key == "habits" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO habits (user_id,name,category,target_days,created_at)
                       VALUES (?,?,?,?,?)""",
                    (uid, m.get("name",""), m.get("category","health"),
                     m.get("target_days","daily"), m.get("created_at", self._now())),
                )
                for c in m.get("completions", []):
                    conn.execute(
                        "INSERT OR IGNORE INTO habit_completions (user_id,habit_name,completion_date) VALUES (?,?,?)",
                        (uid, m.get("name",""), c),
                    )
        elif key == "nutrition" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR IGNORE INTO nutrition_logs
                       (user_id,date,meal,description,calories_est,protein_est,notes,timestamp)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (uid, m.get("date",""), m.get("meal",""), m.get("description",""),
                     m.get("calories_est",0), m.get("protein_est",0),
                     m.get("notes",""), m.get("timestamp", self._now())),
                )
        elif key == "water" and isinstance(data, dict):
            for d, g in data.items():
                conn.execute(
                    "INSERT OR REPLACE INTO water_logs (user_id,date,glasses) VALUES (?,?,?)",
                    (uid, d, max(0, int(g))),
                )
        elif key == "telemetry" and isinstance(data, dict):
            for evt in data.get("events", []):
                conn.execute(
                    """INSERT OR IGNORE INTO telemetry
                       (user_id,schema_version,event_id,event_type,name,status,source,route,timestamp,payload)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (uid, evt.get("schema_version",1), evt.get("event_id", uuid4().hex),
                     evt.get("event_type","event"), evt.get("name",""), evt.get("status","ok"),
                     evt.get("source","memory"), evt.get("route",""), evt.get("timestamp", self._now()),
                     json.dumps(evt.get("payload",{}))),
                )
        elif key == "app_state":
            if isinstance(data, dict):
                kv = data.get("data", data) if "data" in data else data
                if isinstance(kv, dict):
                    for k, v in kv.items():
                        if k == "schema_version":
                            continue
                        conn.execute(
                            "INSERT OR REPLACE INTO app_state (user_id,key,value) VALUES (?,?,?)",
                            (uid, k, json.dumps(v)),
                        )
        conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _habit_snapshot(self, habit: dict) -> dict:
        name = habit.get("name", "")
        rows = self._conn.execute(
            "SELECT completion_date FROM habit_completions WHERE user_id=? AND lower(habit_name)=lower(?)",
            (self.user_id, name),
        ).fetchall()
        completions = {r["completion_date"] for r in rows}
        today = _today()
        done_today = today.isoformat() in completions
        return {**habit, "completions": sorted(completions), "done_today": done_today,
                "streak": _habit_streak(completions, today)}

    def _infer_calendar_kind(self, item: dict) -> str:
        haystack = " ".join(str(item.get(f, "")).lower() for f in ("title", "details", "source", "status"))
        if "birthday" in haystack:
            return "birthday"
        if "trip" in haystack or "travel" in haystack or "flight" in haystack:
            return "trip"
        if "date" in haystack or "anniversary" in haystack:
            return "dating"
        return "calendar"

    def _json_safe_value(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(k): self._json_safe_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe_value(i) for i in value]
        if isinstance(value, set):
            return [self._json_safe_value(i) for i in sorted(value, key=str)]
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return str(value)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _today() -> date:
    return date.today()


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value.strip())
    except (TypeError, ValueError, AttributeError):
        return None


def _habit_streak(completions: set[str], start_date: date | None = None) -> int:
    from datetime import timedelta
    check = start_date or _today()
    streak = 0
    while check.isoformat() in completions:
        streak += 1
        check = check - timedelta(days=1)
    return streak
