"""Telemetry DAO — telemetry, app_state tables + cross-domain progress methods."""

import json
import logging
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4


def _now():
    return datetime.now(UTC).isoformat()
def _today():
    return date.today()


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value.strip())
    except (TypeError, ValueError, AttributeError):
        return None


def _habit_streak(completions: set[str], start_date: date | None = None) -> int:
    check = start_date or _today()
    streak = 0
    while check.isoformat() in completions:
        streak += 1
        check = check - timedelta(days=1)
    return streak


_SCHEMA_SQL = """
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
"""


class TelemetryDAO:
    def __init__(self, conn, user_id: str):
        self._conn = conn
        self.user_id = user_id

    # ------------------------------------------------------------------
    # JSON-safe value helper
    # ------------------------------------------------------------------

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
    # Calendar kind inference helper
    # ------------------------------------------------------------------

    def _infer_calendar_kind(self, item: dict) -> str:
        haystack = " ".join(str(item.get(f, "")).lower() for f in ("title", "details", "source", "status"))
        if "birthday" in haystack:
            return "birthday"
        if "trip" in haystack or "travel" in haystack or "flight" in haystack:
            return "trip"
        if "date" in haystack or "anniversary" in haystack:
            return "dating"
        return "calendar"

    # ------------------------------------------------------------------
    # Habit snapshot helper (inlined for progress_snapshot)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Telemetry events
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
            "timestamp": _now(),
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
        rows = self._conn.execute(
            "SELECT title,date,time,details,status,source,updated_at FROM calendar_items WHERE user_id=?",
            (self.user_id,),
        ).fetchall()
        for item in [dict(r) for r in rows]:
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
        rows = self._conn.execute(
            "SELECT topic,next_step,due_date,status,updated_at "
            "FROM open_loops WHERE user_id=? AND lower(status)=lower(?) ORDER BY rowid DESC LIMIT ?",
            (self.user_id, "open", 20),
        ).fetchall()
        for loop in [dict(r) for r in reversed(rows)]:
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

    def get_progress_snapshot(self) -> dict:
        rows = self._conn.execute(
            "SELECT mood, note, timestamp FROM moods WHERE user_id=? ORDER BY id DESC LIMIT 10",
            (self.user_id,),
        ).fetchall()
        moods = [dict(r) for r in reversed(rows)]

        rows = self._conn.execute(
            "SELECT id, entry, timestamp FROM journal WHERE user_id=? ORDER BY id DESC LIMIT 5",
            (self.user_id,),
        ).fetchall()
        journal = [dict(r) for r in reversed(rows)]

        rows = self._conn.execute(
            "SELECT title,category,target_date,next_step,status,updated_at "
            "FROM goals WHERE user_id=? ORDER BY rowid DESC LIMIT 20",
            (self.user_id,),
        ).fetchall()
        goals = [dict(r) for r in reversed(rows)]

        rows = self._conn.execute(
            "SELECT topic,next_step,due_date,status,updated_at "
            "FROM open_loops WHERE user_id=? ORDER BY rowid DESC LIMIT 20",
            (self.user_id,),
        ).fetchall()
        open_loops = [dict(r) for r in reversed(rows)]

        rows = self._conn.execute(
            "SELECT person,relationship,summary,user_needs,cautions,updated_at FROM relationships WHERE user_id=?",
            (self.user_id,),
        ).fetchall()
        relationships = [dict(r) for r in rows]

        rows = self._conn.execute(
            "SELECT category,value,context,updated_at FROM preferences WHERE user_id=? ORDER BY rowid DESC LIMIT 50",
            (self.user_id,),
        ).fetchall()
        preferences = [dict(r) for r in reversed(rows)]

        rows = self._conn.execute(
            "SELECT title,date,time,details,status,source,updated_at FROM calendar_items WHERE user_id=?",
            (self.user_id,),
        ).fetchall()
        calendar_items = [dict(r) for r in rows]

        rows = self._conn.execute(
            "SELECT category,title,reason,creator,status,updated_at FROM favorites WHERE user_id=? ORDER BY rowid DESC LIMIT 50",
            (self.user_id,),
        ).fetchall()
        favorites = [dict(r) for r in reversed(rows)]

        rows = self._conn.execute(
            "SELECT name,schedule,goal,notes,status,updated_at FROM gym_plans WHERE user_id=? ORDER BY rowid DESC LIMIT 20",
            (self.user_id,),
        ).fetchall()
        gym_plans = [dict(r) for r in reversed(rows)]

        rows = self._conn.execute(
            "SELECT name,goal,daily_structure,notes,status,updated_at FROM food_programs WHERE user_id=? ORDER BY rowid DESC LIMIT 20",
            (self.user_id,),
        ).fetchall()
        food_programs = [dict(r) for r in reversed(rows)]

        rows = self._conn.execute(
            "SELECT name,category,target_days,created_at FROM habits WHERE user_id=?",
            (self.user_id,),
        ).fetchall()
        habits = []
        for row in rows:
            habits.append(self._habit_snapshot(dict(row)))

        active_goals = [g for g in goals if g.get("status") == "active"]
        unresolved = [loop for loop in open_loops if loop.get("status", "open") == "open"]

        row = self._conn.execute(
            "SELECT glasses FROM water_logs WHERE user_id=? AND date=?",
            (self.user_id, _today().isoformat()),
        ).fetchone()
        water_today = int(row["glasses"]) if row else 0

        rows = self._conn.execute(
            "SELECT date,plan_name,exercises,duration_min,notes,energy_rating,timestamp "
            "FROM workout_sessions WHERE user_id=? ORDER BY id DESC LIMIT 50",
            (self.user_id,),
        ).fetchall()
        workout_sessions = [dict(r) for r in reversed(rows)]

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
            "water_today": water_today,
            "workout_session_count": len(workout_sessions),
        }

    # ------------------------------------------------------------------
    # JSON migration (one-time, on first init with existing data)
    # ------------------------------------------------------------------

    def _migrate_from_json(self, db_dir: Path) -> None:
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
                    (uid, m.get("role",""), m.get("content",""), m.get("timestamp", _now())),
                )
        elif key == "moods" and isinstance(data, list):
            for m in data:
                conn.execute(
                    "INSERT OR IGNORE INTO moods (user_id,mood,note,timestamp) VALUES (?,?,?,?)",
                    (uid, m.get("mood",""), m.get("note",""), m.get("timestamp", _now())),
                )
        elif key == "journal" and isinstance(data, list):
            for m in data:
                conn.execute(
                    "INSERT OR IGNORE INTO journal (user_id,entry,timestamp) VALUES (?,?,?)",
                    (uid, m.get("entry",""), m.get("timestamp", _now())),
                )
        elif key == "relationships" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO relationships
                       (user_id,person,relationship,summary,user_needs,cautions,updated_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (uid, m.get("person",""), m.get("relationship",""), m.get("summary",""),
                     m.get("user_needs",""), m.get("cautions",""), m.get("updated_at", _now())),
                )
        elif key == "goals" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO goals
                       (user_id,title,category,target_date,next_step,status,updated_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (uid, m.get("title",""), m.get("category","personal"), m.get("target_date",""),
                     m.get("next_step",""), m.get("status","active"), m.get("updated_at", _now())),
                )
        elif key == "open_loops" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO open_loops
                       (user_id,topic,next_step,due_date,status,updated_at) VALUES (?,?,?,?,?,?)""",
                    (uid, m.get("topic",""), m.get("next_step",""), m.get("due_date",""),
                     m.get("status","open"), m.get("updated_at", _now())),
                )
        elif key == "preferences" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO preferences
                       (user_id,category,value,context,updated_at) VALUES (?,?,?,?,?)""",
                    (uid, m.get("category",""), m.get("value",""), m.get("context",""),
                     m.get("updated_at", _now())),
                )
        elif key == "calendar" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO calendar_items
                       (user_id,title,date,time,details,status,source,updated_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (uid, m.get("title",""), m.get("date",""), m.get("time",""),
                     m.get("details",""), m.get("status","planned"), m.get("source","conversation"),
                     m.get("updated_at", _now())),
                )
        elif key == "favorites" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO favorites
                       (user_id,category,title,reason,creator,status,updated_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (uid, m.get("category",""), m.get("title",""), m.get("reason",""),
                     m.get("creator",""), m.get("status","saved"), m.get("updated_at", _now())),
                )
        elif key == "gym" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO gym_plans
                       (user_id,name,schedule,goal,notes,status,updated_at) VALUES (?,?,?,?,?,?,?)""",
                    (uid, m.get("name",""), m.get("schedule",""), m.get("goal",""),
                     m.get("notes",""), m.get("status","active"), m.get("updated_at", _now())),
                )
        elif key == "food" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO food_programs
                       (user_id,name,goal,daily_structure,notes,status,updated_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (uid, m.get("name",""), m.get("goal",""), m.get("daily_structure",""),
                     m.get("notes",""), m.get("status","active"), m.get("updated_at", _now())),
                )
        elif key == "workouts" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR IGNORE INTO workout_sessions
                       (user_id,date,plan_name,exercises,duration_min,notes,energy_rating,timestamp)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (uid, m.get("date",""), m.get("plan_name",""), m.get("exercises",""),
                     m.get("duration_min",0), m.get("notes",""), m.get("energy_rating",0),
                     m.get("timestamp", _now())),
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
                     m.get("notes",""), m.get("timestamp", _now())),
                )
        elif key == "habits" and isinstance(data, list):
            for m in data:
                conn.execute(
                    """INSERT OR REPLACE INTO habits (user_id,name,category,target_days,created_at)
                       VALUES (?,?,?,?,?)""",
                    (uid, m.get("name",""), m.get("category","health"),
                     m.get("target_days","daily"), m.get("created_at", _now())),
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
                     m.get("notes",""), m.get("timestamp", _now())),
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
                     evt.get("source","memory"), evt.get("route",""), evt.get("timestamp", _now()),
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
