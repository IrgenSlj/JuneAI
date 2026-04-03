# Next Session — Session 3: SQLite Memory Upgrade

Pick up here. Read this file first, then start coding.

## Context

Full plan is in `docs/PLAN.md`. Sessions 1 and 2 are complete.

Repo: /Users/admin/JuneAI/  
App dir: /Users/admin/JuneAI/JuneAI-app/  
Venv: /Users/admin/JuneAI/JuneAI-app/.venv  
Run tests: `cd JuneAI-app && .venv/bin/python -m pytest tests/unit_tests/ -q`  
Current tests: 96 passing  

## Sessions 1 & 2 — Done

Session 1:
- Startup guard — `startup_error` exported from `graph.py`, shown in UI if Ollama is missing
- Removed `ast.literal_eval` — safe JSON-only extraction in `graph.py`
- Config dead code fixed — `preset.default_api_key` reachable, raises `ValueError` if missing
- Memory context cache — 30s TTL per user_id, invalidated on tool writes
- JSON corruption recovery — now logs a `WARNING` before falling back

Session 2:
- `local_mistral_7b` preset added (`mistral:7b-instruct-v0.3`, `tool_strategy="native"`)
- `detect_tool_strategy(model_name)` added to `config.py`
- Native/recovery branching in `graph.py` — native models skip the recovery pipeline
- `JUNE_TOOLS_CORE` (20 tools) in `tools.py` — used automatically for `local_mistral_3b`
- Tool success badge (`🟢 Tools: N/N saved`) in the right-rail debug panel

## Session 3 Goal

Replace the 18 JSON files in `.june_memory/` with a single SQLite database.

Same public interface — `Memory` class keeps identical method signatures.
No other code changes needed except memory.py and a migration script.
This unlocks cross-chapter queries, better performance, and proper ACID guarantees.

## Why SQLite

- Ships with Python stdlib — zero new dependencies
- Single file: `.june_memory/june.db` — still local-first, still inspectable
- Queryable: enables cross-chapter correlation (sleep vs mood, workout vs energy)
- Atomic writes: no more temp-file-replace dance
- Human-readable: `sqlite3 .june_memory/june.db .dump` works from terminal

## Four Tasks — Do in Order

### Task 1 — Create memory_sqlite.py with the full schema

Create `JuneAI-app/src/agent/memory_sqlite.py`.

This is a drop-in replacement for `memory.py`. The `Memory` class must expose **every
public method** that `memory.py` currently exposes with identical signatures.

**Schema** — create these tables in `__init__` using `CREATE TABLE IF NOT EXISTS`:

```sql
messages       (id INTEGER PRIMARY KEY, role TEXT, content TEXT, created_at TEXT)
moods          (id INTEGER PRIMARY KEY, mood TEXT, note TEXT, created_at TEXT)
journal_entries(id INTEGER PRIMARY KEY, entry TEXT, created_at TEXT)
goals          (id INTEGER PRIMARY KEY, title TEXT, description TEXT, status TEXT DEFAULT 'active', deadline TEXT, next_step TEXT, created_at TEXT, updated_at TEXT)
open_loops     (id INTEGER PRIMARY KEY, topic TEXT, detail TEXT, next_step TEXT, due_date TEXT, status TEXT DEFAULT 'open', created_at TEXT, updated_at TEXT)
relationships  (id INTEGER PRIMARY KEY, person TEXT, relation_type TEXT, context TEXT, communication_notes TEXT, birthday TEXT, created_at TEXT, updated_at TEXT)
preferences    (id INTEGER PRIMARY KEY, category TEXT, detail TEXT, created_at TEXT)
calendar_items (id INTEGER PRIMARY KEY, title TEXT, date TEXT, time TEXT, type TEXT, details TEXT, recurrence TEXT, status TEXT DEFAULT 'upcoming', created_at TEXT, updated_at TEXT)
favorites      (id INTEGER PRIMARY KEY, category TEXT, title TEXT, detail TEXT, rating TEXT, created_at TEXT)
gym_plans      (id INTEGER PRIMARY KEY, title TEXT, structure TEXT, frequency TEXT, goal TEXT, status TEXT DEFAULT 'active', created_at TEXT, updated_at TEXT)
food_programs  (id INTEGER PRIMARY KEY, title TEXT, approach TEXT, daily_structure TEXT, goal TEXT, status TEXT DEFAULT 'active', created_at TEXT, updated_at TEXT)
workouts       (id INTEGER PRIMARY KEY, date TEXT, type TEXT, duration_minutes INTEGER, exercises TEXT, notes TEXT, energy_before INTEGER, energy_after INTEGER, created_at TEXT)
body_metrics   (id INTEGER PRIMARY KEY, date TEXT UNIQUE, weight_kg REAL, sleep_hours REAL, sleep_quality INTEGER, energy INTEGER, stress INTEGER, soreness INTEGER, resting_hr INTEGER, steps INTEGER, notes TEXT, created_at TEXT)
habits         (id INTEGER PRIMARY KEY, name TEXT UNIQUE, frequency TEXT DEFAULT 'daily', target INTEGER DEFAULT 1, active INTEGER DEFAULT 1, created_at TEXT)
habit_completions (id INTEGER PRIMARY KEY, habit_name TEXT, date TEXT, created_at TEXT, UNIQUE(habit_name, date))
nutrition_logs (id INTEGER PRIMARY KEY, date TEXT, meal TEXT, description TEXT, calories_est INTEGER, protein_est INTEGER, created_at TEXT)
water_logs     (id INTEGER PRIMARY KEY, date TEXT UNIQUE, glasses INTEGER DEFAULT 0, updated_at TEXT)
events         (id INTEGER PRIMARY KEY, event_type TEXT, payload TEXT, created_at TEXT)
app_state      (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)
```

Implement every method. Mirror the exact behaviour of `memory.py`:
- `save_message`, `load_chat_messages` — messages table, cap at 50 rows
- `log_mood`, `get_mood_history` — moods table
- `save_journal`, `get_journal` — journal_entries table
- `save_relationship_profile`, `get_relationship_profiles` — relationships table
- `save_goal`, `get_goals`, `update_goal_status` — goals table
- `save_open_loop`, `get_open_loops`, `update_open_loop_status` — open_loops table
- `save_preference`, `get_preferences` — preferences table
- `save_calendar_item`, `get_calendar_items`, `update_calendar_item_status` — calendar_items
- `save_favorite`, `get_favorites` — favorites table
- `save_gym_plan`, `get_gym_plans` — gym_plans table
- `save_food_program`, `get_food_programs` — food_programs table
- `log_workout_session`, `get_workout_sessions`, `get_today_workout` — workouts table
- `log_body_metrics`, `get_body_metrics`, `get_today_body_metrics` — body_metrics table
- `create_or_update_habit`, `log_habit_completion`, `get_habits` — habits + habit_completions
- `log_nutrition`, `get_nutrition_today`, `get_nutrition_recent` — nutrition_logs table
- `log_water`, `set_water`, `get_water_today` — water_logs table
- `get_chapter_completeness`, `get_chapters_needing_attention`, `get_today_summary`
- `append_event`, `record_tool_call`, `record_route_selection`, `record_save_event`, `get_recent_events`
- `get_app_state`, `set_app_state_value`, `should_send_daily_checkin`, `mark_daily_checkin_sent`
- `get_upcoming_notifications`, `get_progress_snapshot`

Use `datetime.utcnow().isoformat()` for `created_at` fields.
Use `sqlite3.Row` as `row_factory` so rows behave like dicts.
Use a single persistent connection per `Memory` instance.

### Task 2 — Make Memory use SQLite by default

In `memory.py`, change the `Memory` class to delegate to `memory_sqlite.py`:

Option A (cleanest): rename `memory.py`'s class to `_MemoryJSON` and make `Memory` import
from `memory_sqlite.py` if `june.db` is present or if `USE_SQLITE=1` env var is set,
otherwise fall back to `_MemoryJSON`.

Option B (simpler): in `memory.py`'s `__init__`, if `june.db` does not yet exist but
`USE_SQLITE` env var is set, delegate all calls to a `memory_sqlite.Memory` instance.

**Recommended**: just replace the `Memory` class in `memory.py` entirely with an import
alias from `memory_sqlite.py`:

```python
# memory.py — after migration
from .memory_sqlite import Memory  # noqa: F401
```

And keep the original implementation as `_memory_json_legacy.py` for reference.

This is the cleanest approach — all imports of `from agent.memory import Memory` continue
to work with no other changes.

### Task 3 — Migration script

Create `JuneAI-app/scripts/migrate_json_to_sqlite.py`:

```python
#!/usr/bin/env python3
"""Migrate .june_memory JSON files to SQLite.

Usage:
    python scripts/migrate_json_to_sqlite.py [--memory-dir .june_memory]

Reads every user subdirectory under MEMORY_DIR and writes each record
into the SQLite database for that user. Non-destructive — JSON files
are left in place as backup.
"""
import argparse, json, os, sys
from pathlib import Path

# ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.environ.setdefault("USE_SQLITE", "1")

from agent.memory_sqlite import Memory


def migrate_user(user_dir: Path) -> None:
    user_id = user_dir.name
    print(f"  Migrating {user_id}...")
    mem = Memory(user_id)

    def load(name: str, default):
        f = user_dir / name
        if not f.exists():
            return default
        try:
            return json.loads(f.read_text())
        except Exception:
            return default

    # messages
    for msg in load("chat_history.json", []):
        mem.save_message(msg.get("role", "user"), msg.get("content", ""))

    # moods
    for m in load("mood.json", []):
        mem._conn.execute(
            "INSERT OR IGNORE INTO moods (mood, note, created_at) VALUES (?,?,?)",
            (m.get("mood",""), m.get("note",""), m.get("timestamp", mem._now())),
        )

    # goals
    for g in load("goals.json", []):
        mem.save_goal(
            g.get("title",""),
            description=g.get("description",""),
            next_step=g.get("next_step",""),
            deadline=g.get("deadline",""),
        )

    # calendar
    for c in load("calendar.json", []):
        mem.save_calendar_item(
            c.get("title",""),
            c.get("date",""),
            time=c.get("time",""),
            item_type=c.get("type",""),
            details=c.get("details",""),
            recurrence=c.get("recurrence",""),
        )

    # habits
    for h in load("habits.json", []):
        mem.create_or_update_habit(h.get("name",""), frequency=h.get("frequency","daily"))
        for d in h.get("completions", []):
            mem.log_habit_completion(h.get("name",""), date_str=d)

    # body metrics
    for b in load("body_metrics.json", []):
        mem.log_body_metrics(
            weight_kg=b.get("weight_kg"),
            sleep_hours=b.get("sleep_hours"),
            sleep_quality=b.get("sleep_quality"),
            energy=b.get("energy"),
            stress=b.get("stress"),
            soreness=b.get("soreness"),
            resting_hr=b.get("resting_hr"),
            steps=b.get("steps"),
            notes=b.get("notes",""),
        )

    # workouts
    for w in load("workouts.json", []):
        mem.log_workout_session(
            w.get("type",""),
            duration_min=w.get("duration_minutes",0),
            exercises=w.get("exercises",""),
            notes=w.get("notes",""),
            energy_rating=w.get("energy_after",0),
        )

    mem._conn.commit()
    print(f"    Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-dir", default=os.getenv("MEMORY_DIR", ".june_memory"))
    args = parser.parse_args()
    memory_dir = Path(args.memory_dir)
    if not memory_dir.exists():
        print(f"Memory dir not found: {memory_dir}")
        sys.exit(1)
    for user_dir in sorted(memory_dir.iterdir()):
        if user_dir.is_dir() and not user_dir.name.startswith("."):
            migrate_user(user_dir)
    print("Migration complete.")

if __name__ == "__main__":
    main()
```

Adapt method call signatures to match what is actually in `memory_sqlite.py`.

Add `make migrate-memory` to `Makefile`:
```makefile
migrate-memory:
	$(VENV_PYTHON) scripts/migrate_json_to_sqlite.py
```

### Task 4 — Add two cross-chapter query methods to Memory

These are new capabilities that JSON could not easily support. Add them to `memory_sqlite.py`:

```python
def get_weekly_pattern(self) -> list[dict]:
    """Return last 7 days of correlated body metrics + mood + workout presence."""
    rows = self._conn.execute("""
        SELECT
            b.date,
            b.sleep_hours,
            b.energy,
            b.stress,
            b.weight_kg,
            m.mood,
            CASE WHEN w.id IS NOT NULL THEN 1 ELSE 0 END AS worked_out
        FROM body_metrics b
        LEFT JOIN moods m ON m.created_at LIKE b.date || '%'
        LEFT JOIN workouts w ON w.date = b.date
        WHERE b.date >= date('now', '-7 days')
        ORDER BY b.date DESC
    """).fetchall()
    return [dict(r) for r in rows]

def get_habit_consistency(self, days: int = 14) -> list[dict]:
    """Return habit completion rate over the last N days."""
    rows = self._conn.execute("""
        SELECT
            h.name,
            h.frequency,
            COUNT(hc.id) AS completions,
            CAST(? AS FLOAT) AS window_days,
            ROUND(COUNT(hc.id) * 100.0 / ?, 1) AS pct
        FROM habits h
        LEFT JOIN habit_completions hc
            ON hc.habit_name = h.name
            AND hc.date >= date('now', '-' || ? || ' days')
        WHERE h.active = 1
        GROUP BY h.name
        ORDER BY pct DESC
    """, (days, days, days)).fetchall()
    return [dict(r) for r in rows]
```

## Tests to write

Add `JuneAI-app/tests/unit_tests/test_memory_sqlite.py`:

```python
"""Tests for the SQLite memory backend."""
import os
from unittest.mock import patch

import pytest

os.environ["USE_SQLITE"] = "1"
from agent.memory_sqlite import Memory


def _mem(tmp_path, uid="sqlite_test"):
    with patch("agent.memory_sqlite.MEMORY_DIR", str(tmp_path)):
        return Memory(uid)


def test_sqlite_db_file_is_created(tmp_path):
    _mem(tmp_path)
    assert (tmp_path / "sqlite_test" / "june.db").exists()


def test_save_and_load_message(tmp_path):
    mem = _mem(tmp_path)
    mem.save_message("user", "hello")
    mem.save_message("assistant", "hi there")
    msgs = mem.load_chat_messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"


def test_log_and_get_mood(tmp_path):
    mem = _mem(tmp_path)
    mem.log_mood("good", "great day")
    history = mem.get_mood_history()
    assert len(history) == 1
    assert history[0]["mood"] == "good"


def test_save_and_get_goal(tmp_path):
    mem = _mem(tmp_path)
    mem.save_goal("Run a 5k", next_step="Buy shoes", deadline="2026-06-01")
    goals = mem.get_goals()
    assert len(goals) == 1
    assert goals[0]["title"] == "Run a 5k"


def test_update_goal_status(tmp_path):
    mem = _mem(tmp_path)
    mem.save_goal("Read 12 books")
    mem.update_goal_status("Read 12 books", "completed")
    completed = mem.get_goals(status="completed")
    assert any(g["title"] == "Read 12 books" for g in completed)


def test_calendar_item_save_and_retrieve(tmp_path):
    mem = _mem(tmp_path)
    mem.save_calendar_item("Dentist", "2026-05-10", details="Annual checkup")
    items = mem.get_calendar_items()
    assert any(i["title"] == "Dentist" for i in items)


def test_habit_streak(tmp_path):
    mem = _mem(tmp_path)
    mem.create_or_update_habit("Walk")
    mem.log_habit_completion("Walk")
    habits = mem.get_habits()
    assert any(h["name"] == "Walk" for h in habits)


def test_body_metrics_log_and_read(tmp_path):
    mem = _mem(tmp_path)
    mem.log_body_metrics(weight_kg=73.0, sleep_hours=7.5, energy=8)
    today = mem.get_today_body_metrics()
    assert today is not None
    assert today["weight_kg"] == 73.0


def test_water_log(tmp_path):
    mem = _mem(tmp_path)
    mem.log_water(3)
    mem.log_water(2)
    assert mem.get_water_today() == 5


def test_cross_chapter_weekly_pattern(tmp_path):
    mem = _mem(tmp_path)
    mem.log_body_metrics(weight_kg=72.0, sleep_hours=7.0, energy=7, stress=3)
    mem.log_mood("focused")
    pattern = mem.get_weekly_pattern()
    assert len(pattern) >= 1


def test_habit_consistency(tmp_path):
    mem = _mem(tmp_path)
    mem.create_or_update_habit("Meditate")
    mem.log_habit_completion("Meditate")
    consistency = mem.get_habit_consistency()
    assert any(h["name"] == "Meditate" for h in consistency)
```

## Important notes

- All existing tests must continue to pass — they test `memory.py` which becomes an alias
- The `Memory` class signature must be identical: `Memory(user_id: str)` with `MEMORY_DIR` read from env/config
- SQLite file goes at `<MEMORY_DIR>/<user_id>/june.db`
- Use `with patch("agent.memory_sqlite.MEMORY_DIR", ...)` in tests (same pattern as existing tests use for `agent.memory.MEMORY_DIR`)

## Success criteria
- [ ] `memory_sqlite.py` exists and implements all public methods
- [ ] `from agent.memory import Memory` returns the SQLite-backed class
- [ ] All 96 existing tests still pass (they test the same interface)
- [ ] All new SQLite tests pass
- [ ] `scripts/migrate_json_to_sqlite.py` exists and runs without error on an empty directory
- [ ] `make migrate-memory` target exists in Makefile
- [ ] `get_weekly_pattern()` and `get_habit_consistency()` work and are tested

## Final step
After all tests pass, commit with message:
"Session 3: SQLite memory backend — drop-in replacement, cross-chapter queries, migration script"

Do NOT push — just commit.
