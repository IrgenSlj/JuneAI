"""Fitness DAO — gym_plans, food_programs, workout_sessions, body_metrics tables."""

from datetime import UTC, date, datetime


def _now():
    return datetime.now(UTC).isoformat()
def _today():
    return date.today()

_SCHEMA_SQL = """
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
"""


class FitnessDAO:
    def __init__(self, conn, user_id: str):
        self._conn = conn
        self.user_id = user_id

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
            "updated_at": _now(),
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
            "updated_at": _now(),
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
            "timestamp": _now(),
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
            "timestamp": _now(),
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

    def get_today_body_metrics(self) -> dict | None:
        row = self._conn.execute(
            "SELECT date,weight_kg,sleep_hours,sleep_quality,energy,stress,"
            "soreness,resting_hr,steps,notes,timestamp "
            "FROM body_metrics WHERE user_id=? AND date=?",
            (self.user_id, _today().isoformat()),
        ).fetchone()
        return dict(row) if row else None
