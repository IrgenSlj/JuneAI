"""Habits DAO — habits, habit_completions, nutrition_logs, water_logs tables."""

from datetime import UTC, date, datetime, timedelta


def _now():
    return datetime.now(UTC).isoformat()
def _today():
    return date.today()


def _habit_streak(completions: set[str], start_date: date | None = None) -> int:
    check = start_date or _today()
    streak = 0
    while check.isoformat() in completions:
        streak += 1
        check = check - timedelta(days=1)
    return streak


_SCHEMA_SQL = """
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
"""


class HabitDAO:
    def __init__(self, conn, user_id: str):
        self._conn = conn
        self.user_id = user_id

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

    def create_or_update_habit(
        self,
        name: str,
        category: str = "health",
        target_days: str = "daily",
    ) -> dict:
        now = _now()
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
        self._conn.execute(
            "INSERT OR IGNORE INTO habits (user_id,name,category,target_days,created_at) VALUES (?,?,?,?,?)",
            (self.user_id, name, "health", "daily", _now()),
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
            "timestamp": _now(),
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
