"""JuneAI memory system — SQLite backend.

Single june.db per data-directory memory folder. user_id is a column in every
table.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import threading
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

from ..config import MEMORY_DIR as _IMPORTED_MEMORY_DIR
from ..failure import degrade_quietly
from .daos import (
    CalendarDAO,
    ChatDAO,
    FeedbackDAO,
    FitnessDAO,
    GoalDAO,
    JournalDAO,
    PreferenceDAO,
    RelationshipDAO,
    TelemetryDAO,
)
from .migration import ensure_schema

_LEGACY_JSON_SUFFIXES = (
    "chat",
    "moods",
    "journal",
    "relationships",
    "goals",
    "open_loops",
    "preferences",
    "calendar",
    "favorites",
    "gym_plans",
    "food_programs",
    "workout_sessions",
    "body_metrics",
    "habits",
    "nutrition_logs",
    "water_logs",
    "telemetry",
    "app_state",
)


def _memory_package_dir_override() -> str | None:
    """Return the test/legacy package-level override, if one is active."""
    from . import MEMORY_DIR  # re-read package attribute each call

    memory_dir = str(MEMORY_DIR)
    if memory_dir != str(_IMPORTED_MEMORY_DIR):
        return memory_dir
    return None


def _current_data_dir() -> Path:
    override = _memory_package_dir_override()
    if override is not None:
        return Path(override).expanduser()

    import june_brain.config as _cfg  # noqa: PLC0415

    return Path(_cfg.MEMORY_DIR).expanduser()


def _canonical_memory_dir() -> Path:
    override = _memory_package_dir_override()
    if override is not None:
        return Path(override).expanduser() / "memory"

    from ..datadir.layout import memory_dir  # noqa: PLC0415

    return memory_dir().expanduser()


def _memory_store_exists(path: Path) -> bool:
    return any(
        (path / name).exists()
        for name in ("june.db", "june.db-wal", "june.db-shm", "chroma")
    )


def _legacy_json_exists(path: Path) -> bool:
    # NB: ``path.glob(...)`` returns a generator, which is always truthy — so the
    # match must be consumed with an inner ``any(...)``; testing the generator
    # object itself would report "legacy data" for every directory, including
    # fresh ones, and wrongly pin stores to the data-dir root.
    return any(
        any(path.glob(f"*_{suffix}.json")) for suffix in _LEGACY_JSON_SUFFIXES
    )


def _current_memory_dir() -> str:
    """Resolve the directory containing persisted memory stores.

    New data dirs use ``june_brain.datadir.layout.memory_dir()``. If an
    existing install still has root-level memory artifacts and no canonical
    ``memory/`` artifacts, keep using the legacy root so upgrades remain
    non-destructive. Tests that patch ``june_brain.memory.MEMORY_DIR`` still
    get an isolated data root.
    """
    data_root = _current_data_dir()
    canonical = _canonical_memory_dir()
    if (
        canonical != data_root
        and not _memory_store_exists(canonical)
        and (_memory_store_exists(data_root) or _legacy_json_exists(data_root))
    ):
        return str(data_root)
    return str(canonical)


def db_path() -> str:
    """Absolute path to the shared june.db.

    This is the one correct construction; use it instead of rebuilding the path
    by hand so callers get the canonical ``<data>/memory`` location and the
    legacy root-level fallback.
    """
    return str(Path(_current_memory_dir()) / "june.db")


APP_STATE_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Corrupt-database startup recovery
# ---------------------------------------------------------------------------
_log = logging.getLogger(__name__)

# Populated when a corrupt DB is moved aside on startup; None otherwise.
# Consumers (e.g. /system) may read this to surface the recovery event.
_last_recovery_backup: str | None = None

# Serializes corruption recovery so two thread-pool workers that miss the
# connection cache in the same instant cannot both move the same file aside
# (shutil.move overwrites its destination, which would silently destroy the
# first thread's backup). Recovery is a rare startup event, so contention here
# is effectively nil. Also guards the _last_recovery_backup write.
_recovery_lock = threading.Lock()


def _utc_stamp() -> str:
    # Microsecond precision so two recoveries that land in the same wall-clock
    # second (e.g. two processes, or the sub-second gap between siblings) still
    # get distinct backup names.
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")


def _unique_dest(path: Path, stamp: str) -> Path:
    """A .corrupt-<stamp> destination that does not already exist.

    Appends a counter on the astronomically-unlikely stamp collision so a
    prior backup is never overwritten (shutil.move would otherwise clobber it).
    """
    dest = path.with_name(path.name + f".corrupt-{stamp}")
    n = 1
    while dest.exists():
        dest = path.with_name(path.name + f".corrupt-{stamp}.{n}")
        n += 1
    return dest


def _move_aside(path: Path, stamp: str) -> None:
    """Move *path* to *path*.corrupt-<stamp>; fall back to deletion so startup always succeeds."""
    dest = _unique_dest(path, stamp)
    try:
        shutil.move(str(path), str(dest))
        _log.warning("june.db recovery: moved %s -> %s", path.name, dest.name)
    except Exception as exc:  # noqa: BLE001
        _log.warning(
            "june.db recovery: could not move %s (%s); deleting to allow fresh start", path.name, exc
        )
        try:
            path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            degrade_quietly("moving a corrupt database aside")


def _recover_corrupt_db(path_str: str) -> None:
    """Move the corrupt DB file and its WAL/SHM siblings aside; set the module flag."""
    global _last_recovery_backup
    path = Path(path_str)
    stamp = _utc_stamp()
    backup_name = path.name + f".corrupt-{stamp}"
    _log.warning(
        "june.db at %s failed integrity check — starting fresh. "
        "Corrupt file backed up as %s",
        path,
        path.parent / backup_name,
    )
    for sibling in (path, Path(path_str + "-wal"), Path(path_str + "-shm")):
        if sibling.exists():
            _move_aside(sibling, stamp)
    _last_recovery_backup = str(path.parent / backup_name)


def _check_and_recover_if_corrupt(path_str: str) -> None:
    """Probe an existing DB file with PRAGMA quick_check; recover if corrupt.

    Only called for files that already exist (not first-run, not :memory:).
    Raises sqlite3.OperationalError for transient errors (e.g. database is locked)
    so those propagate normally without triggering recovery.
    """
    probe: sqlite3.Connection | None = None
    is_corrupt = False
    try:
        probe = sqlite3.connect(path_str)
        row = probe.execute("PRAGMA quick_check").fetchone()
        if row is None or row[0] != "ok":
            is_corrupt = True
    except sqlite3.OperationalError:
        # Transient: lock, busy, permission — not corruption.  Let it propagate.
        # ORDERING IS LOAD-BEARING: OperationalError IS-A DatabaseError in the
        # sqlite3 exception hierarchy, so this clause MUST stay above the
        # DatabaseError clause below. Swapping them would misclassify a locked
        # database as corrupt and destroy a healthy DB that is merely in use.
        raise
    except sqlite3.DatabaseError:
        # Genuine malformation: "database disk image is malformed",
        # "file is not a database", etc.
        is_corrupt = True
    finally:
        if probe is not None:
            try:
                probe.close()
            except Exception:  # noqa: BLE001
                degrade_quietly("corrupt-database recovery")

    if is_corrupt:
        _recover_corrupt_db(path_str)


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
        # Ensure the memory directory exists — on a fresh install it may not
        # have been created yet, and sqlite won't make the parent dir.
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # Probe existing files for corruption before opening the real connection.
        # An absent file (first run) is fine; :memory: is always fine.
        # Serialize under the recovery lock and re-check existence inside it: a
        # concurrent worker may have already moved a corrupt file aside while we
        # waited, in which case there is nothing left to probe.
        if db_path != ":memory:" and Path(db_path).exists():
            with _recovery_lock:
                if Path(db_path).exists():
                    _check_and_recover_if_corrupt(db_path)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        # Load the sqlite-vec extension so semantic recall can use a vec0 table
        # in this same database (ADR 0019). Best-effort: on a platform where it
        # cannot load, recall degrades to the keyword scan (invariant 6).
        from .vec_index import load_extension

        load_extension(conn)
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
CREATE TABLE IF NOT EXISTS forgotten_nodes (
    user_id      TEXT NOT NULL,
    node_id      TEXT NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'entity',
    label        TEXT NOT NULL DEFAULT '',
    props        TEXT NOT NULL DEFAULT '{}',
    updated_at   TEXT NOT NULL,
    forgotten_at TEXT NOT NULL,
    PRIMARY KEY (user_id, node_id)
);
CREATE INDEX IF NOT EXISTS idx_forgotten_nodes_when ON forgotten_nodes(user_id, forgotten_at);
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
CREATE TABLE IF NOT EXISTS forgotten_facts (
    user_id      TEXT NOT NULL,
    fact_id      TEXT NOT NULL,
    text         TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'conversation',
    metadata     TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    forgotten_at TEXT NOT NULL,
    PRIMARY KEY (user_id, fact_id)
);
CREATE INDEX IF NOT EXISTS idx_forgotten_facts_when ON forgotten_facts(user_id, forgotten_at);
CREATE TABLE IF NOT EXISTS forgotten_structured (
    user_id      TEXT NOT NULL,
    ref          TEXT NOT NULL,
    kind         TEXT NOT NULL,
    summary      TEXT NOT NULL DEFAULT '',
    fields       TEXT NOT NULL DEFAULT '{}',
    source       TEXT NOT NULL DEFAULT 'manual',
    forgotten_at TEXT NOT NULL,
    PRIMARY KEY (user_id, ref)
);
CREATE INDEX IF NOT EXISTS idx_forgotten_structured_when ON forgotten_structured(user_id, forgotten_at);
CREATE TABLE IF NOT EXISTS embedding_cache (
    model       TEXT NOT NULL,
    text_hash   TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (model, text_hash)
);
CREATE TABLE IF NOT EXISTS memory_feedback (
    user_id     TEXT NOT NULL,
    ref         TEXT NOT NULL,
    vote        TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, ref)
);
CREATE TABLE IF NOT EXISTS schedules (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    cron_expression TEXT DEFAULT '',
    interval_seconds INTEGER DEFAULT 0,
    scheduled_at TEXT NOT NULL,
    last_run_at TEXT,
    action_type TEXT NOT NULL DEFAULT 'agent_invoke',
    action_config TEXT NOT NULL DEFAULT '{}',
    max_runs INTEGER DEFAULT 0,
    run_count INTEGER DEFAULT 0,
    enabled INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS skill_inbound_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_key TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    received_at TEXT NOT NULL,
    processed INTEGER DEFAULT 0,
    agent_invoked INTEGER DEFAULT 0
);
"""


def _init_schema(conn: sqlite3.Connection) -> None:
    for stmt in _SCHEMA_SQL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()
    # Apply any pending schema migrations (versioned, idempotent).
    ensure_schema(conn)


# ---------------------------------------------------------------------------
# Memory class
# ---------------------------------------------------------------------------

class Memory:
    """Manages all persistent storage for a single user (SQLite backend)."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._db_path = db_path()
        conn = _get_connection(self._db_path)
        _init_schema(conn)
        db_dir = Path(self._db_path).parent
        self._migrate_from_json(db_dir)
        # DAO layer — new code should prefer these over Memory methods
        self._chat_dao = ChatDAO(conn, user_id)
        self._journal_dao = JournalDAO(conn, user_id)
        self._relationship_dao = RelationshipDAO(conn, user_id)
        self._goal_dao = GoalDAO(conn, user_id)
        self._preference_dao = PreferenceDAO(conn, user_id)
        self._calendar_dao = CalendarDAO(conn, user_id)
        self._fitness_dao = FitnessDAO(conn, user_id)
        self._telemetry_dao = TelemetryDAO(conn, user_id)
        self._feedback_dao = FeedbackDAO(conn, user_id)

    @property
    def _conn(self) -> sqlite3.Connection:
        return _get_connection(self._db_path)

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------

    def save_message(self, role: str, content: str) -> None:
        return self._chat_dao.save_message(role, content)

    def load_chat(self) -> list:
        return self._chat_dao.load_chat()

    def load_chat_messages(self) -> list:
        return self._chat_dao.load_chat_messages()

    # ------------------------------------------------------------------
    # Mood
    # ------------------------------------------------------------------

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

    def update_goal(self, old_title: str, **fields: str) -> dict | None:
        """Update a goal in place, or rename it via delete-then-insert when the title changes."""
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

    def update_calendar_item(
        self,
        old_title: str,
        old_date: str = "",
        old_time: str = "",
        **fields: str,
    ) -> dict | None:
        query = (
            "SELECT title,date,time,details,status,source,updated_at "
            "FROM calendar_items WHERE user_id=? AND lower(title)=lower(?)"
        )
        params: list = [self.user_id, old_title.strip()]
        if old_date.strip():
            query += " AND lower(date)=lower(?)"
            params.append(old_date.strip())
        if old_time.strip():
            query += " AND lower(time)=lower(?)"
            params.append(old_time.strip())
        rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return None
        existing = dict(rows[0])
        merged = {**existing, **{k: v for k, v in fields.items() if v is not None}}
        new_title = (merged.get("title") or "").strip() or existing["title"]
        new_date = (merged.get("date") or "").strip()
        new_time = (merged.get("time") or "").strip()
        pk_changed = (
            new_title.lower() != existing["title"].lower()
            or new_date.lower() != existing["date"].lower()
            or new_time.lower() != existing["time"].lower()
        )
        if pk_changed:
            self.delete_calendar_item(existing["title"], existing["date"], existing["time"])
        return self.save_calendar_item(
            title=new_title,
            date=new_date,
            time=new_time,
            details=merged.get("details", ""),
            status=merged.get("status", "planned"),
            source=merged.get("source", "conversation"),
        )

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

    # ------------------------------------------------------------------
    # Workout sessions
    # ------------------------------------------------------------------

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
        date: str = "",
    ) -> dict:
        # Default to today; an explicit date lets a restore land on the
        # original day rather than overwriting today's metric.
        day = date.strip() or _today().isoformat()
        item = {
            "date": day,
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

    def delete_body_metric(self, date: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM body_metrics WHERE user_id=? AND date=?",
            (self.user_id, date.strip()),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Reversible forget — structured rows trash
    # ------------------------------------------------------------------

    def list_forgotten_structured(self, limit: int = 50) -> list[dict]:
        """List trashed structured rows, most recently forgotten first."""
        rows = self._conn.execute(
            "SELECT ref, kind, summary, fields, source, forgotten_at "
            "FROM forgotten_structured WHERE user_id=? ORDER BY forgotten_at DESC LIMIT ?",
            (self.user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def purge_forgotten_structured(self) -> int:
        """Permanently empty the structured trash. Returns the number of rows removed."""
        cur = self._conn.execute(
            "DELETE FROM forgotten_structured WHERE user_id=?", (self.user_id,)
        )
        self._conn.commit()
        return int(cur.rowcount or 0)

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

    # ------------------------------------------------------------------
    # Nutrition
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Water
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Chapter completeness
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Today summary
    # ------------------------------------------------------------------

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
                logging.exception("migration: failed to read %s", path)
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

    # ------------------------------------------------------------------
    # Memory feedback (B.4) — thumbs up/down on recalled memories
    # ------------------------------------------------------------------

    def set_feedback(self, ref: str, vote: str) -> dict | None:
        """Upsert a vote ('up' or 'down') for a memory ref. Returns the row."""
        ref = ref.strip()
        vote = vote.strip().lower()
        if not ref or vote not in ("up", "down"):
            return None
        now = self._now()
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
        """Return ``{ref: vote}`` for every recorded vote — used during recall ranking."""
        rows = self._conn.execute(
            "SELECT ref, vote FROM memory_feedback WHERE user_id=?",
            (self.user_id,),
        ).fetchall()
        return {row["ref"]: row["vote"] for row in rows}


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
