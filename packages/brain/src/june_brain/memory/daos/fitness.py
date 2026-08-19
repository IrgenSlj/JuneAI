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
