"""JuneAI memory system.

Stores conversation history and assistant artifacts as local JSON files.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from json import JSONDecodeError
from pathlib import Path

from .config import MEMORY_DIR


class Memory:
    """Manages all persistent storage for a single user."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.dir = Path(MEMORY_DIR)
        self.dir.mkdir(exist_ok=True)

        self.chat_file = self.dir / f"{user_id}_chat.json"
        self.mood_file = self.dir / f"{user_id}_moods.json"
        self.journal_file = self.dir / f"{user_id}_journal.json"
        self.relationship_file = self.dir / f"{user_id}_relationships.json"
        self.goals_file = self.dir / f"{user_id}_goals.json"
        self.open_loops_file = self.dir / f"{user_id}_open_loops.json"
        self.preferences_file = self.dir / f"{user_id}_preferences.json"
        self.calendar_file = self.dir / f"{user_id}_calendar.json"
        self.favorites_file = self.dir / f"{user_id}_favorites.json"
        self.gym_file = self.dir / f"{user_id}_gym_plans.json"
        self.food_file = self.dir / f"{user_id}_food_programs.json"
        self.app_state_file = self.dir / f"{user_id}_app_state.json"
        self.workout_sessions_file = self.dir / f"{user_id}_workout_sessions.json"
        self.body_metrics_file = self.dir / f"{user_id}_body_metrics.json"
        self.habits_file = self.dir / f"{user_id}_habits.json"
        self.nutrition_file = self.dir / f"{user_id}_nutrition_logs.json"
        self.water_file = self.dir / f"{user_id}_water_logs.json"

    # --- Chat history ---

    def save_message(self, role: str, content: str) -> None:
        """Append a message to chat history and keep the latest 50."""
        history = self.load_chat()
        history.append({
            "role": role,
            "content": content,
            "timestamp": self._now(),
        })
        if len(history) > 50:
            history = history[-50:]
        self._write(self.chat_file, history)

    def load_chat(self) -> list:
        """Load the raw chat history."""
        return self._read(self.chat_file, [])

    def load_chat_messages(self) -> list:
        """Load chat history as LangChain message objects."""
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

    # --- Mood tracking ---

    def log_mood(self, mood: str, note: str = "") -> dict:
        """Save a mood entry with optional note."""
        moods = self._read(self.mood_file, [])
        entry = {
            "mood": mood.strip(),
            "note": note.strip(),
            "timestamp": self._now(),
        }
        moods.append(entry)
        self._write(self.mood_file, moods)
        return entry

    def get_mood_history(self, limit: int = 10) -> list:
        """Get the most recent mood entries."""
        return self._read(self.mood_file, [])[-limit:]

    # --- Journal ---

    def save_journal(self, entry: str) -> dict:
        """Save a journal or reflection note."""
        journal = self._read(self.journal_file, [])
        item = {
            "entry": entry.strip(),
            "timestamp": self._now(),
        }
        journal.append(item)
        self._write(self.journal_file, journal)
        return item

    def get_journal(self, limit: int = 5) -> list:
        """Get the most recent journal entries."""
        return self._read(self.journal_file, [])[-limit:]

    # --- Relationship context ---

    def save_relationship_profile(
        self,
        person: str,
        relationship: str,
        summary: str,
        user_needs: str = "",
        cautions: str = "",
    ) -> dict:
        """Create or update a structured relationship profile."""
        profiles = self._read(self.relationship_file, [])
        key = person.strip().lower()
        item = {
            "person": person.strip(),
            "relationship": relationship.strip(),
            "summary": summary.strip(),
            "user_needs": user_needs.strip(),
            "cautions": cautions.strip(),
            "updated_at": self._now(),
        }

        for index, existing in enumerate(profiles):
            if existing.get("person", "").strip().lower() == key:
                profiles[index] = item
                self._write(self.relationship_file, profiles)
                return item

        profiles.append(item)
        self._write(self.relationship_file, profiles)
        return item

    def get_relationship_profiles(self, person: str = "") -> list:
        """Get all relationship profiles or a single person profile."""
        profiles = self._read(self.relationship_file, [])
        if not person.strip():
            return profiles
        key = person.strip().lower()
        return [
            profile
            for profile in profiles
            if profile.get("person", "").strip().lower() == key
        ]

    # --- Goals ---

    def save_goal(
        self,
        title: str,
        category: str = "personal",
        target_date: str = "",
        next_step: str = "",
        status: str = "active",
    ) -> dict:
        """Save or update a goal by title."""
        goals = self._read(self.goals_file, [])
        key = title.strip().lower()
        item = {
            "title": title.strip(),
            "category": category.strip() or "personal",
            "target_date": target_date.strip(),
            "next_step": next_step.strip(),
            "status": status.strip() or "active",
            "updated_at": self._now(),
        }

        for index, existing in enumerate(goals):
            if existing.get("title", "").strip().lower() == key:
                goals[index] = item
                self._write(self.goals_file, goals)
                return item

        goals.append(item)
        self._write(self.goals_file, goals)
        return item

    def update_goal_status(self, title: str, status: str) -> dict | None:
        """Update a goal's status by title."""
        goals = self._read(self.goals_file, [])
        key = title.strip().lower()
        for index, existing in enumerate(goals):
            if existing.get("title", "").strip().lower() == key:
                goals[index]["status"] = status.strip() or existing.get("status", "active")
                goals[index]["updated_at"] = self._now()
                self._write(self.goals_file, goals)
                return goals[index]
        return None

    def get_goals(self, status: str = "", limit: int = 10) -> list:
        """Get goals filtered by status."""
        goals = self._read(self.goals_file, [])
        if status.strip():
            goals = [
                goal
                for goal in goals
                if goal.get("status", "").strip().lower() == status.strip().lower()
            ]
        return goals[-limit:]

    # --- Open loops ---

    def save_open_loop(
        self,
        topic: str,
        next_step: str = "",
        due_date: str = "",
        status: str = "open",
    ) -> dict:
        """Save or update an unresolved issue or follow-up."""
        loops = self._read(self.open_loops_file, [])
        key = topic.strip().lower()
        item = {
            "topic": topic.strip(),
            "next_step": next_step.strip(),
            "due_date": due_date.strip(),
            "status": status.strip() or "open",
            "updated_at": self._now(),
        }

        for index, existing in enumerate(loops):
            if existing.get("topic", "").strip().lower() == key:
                loops[index] = item
                self._write(self.open_loops_file, loops)
                return item

        loops.append(item)
        self._write(self.open_loops_file, loops)
        return item

    def update_open_loop_status(self, topic: str, status: str) -> dict | None:
        """Update an open loop's status by topic."""
        loops = self._read(self.open_loops_file, [])
        key = topic.strip().lower()
        for index, existing in enumerate(loops):
            if existing.get("topic", "").strip().lower() == key:
                loops[index]["status"] = status.strip() or existing.get("status", "open")
                loops[index]["updated_at"] = self._now()
                self._write(self.open_loops_file, loops)
                return loops[index]
        return None

    def get_open_loops(self, status: str = "open", limit: int = 10) -> list:
        """Get unresolved or filtered open loops."""
        loops = self._read(self.open_loops_file, [])
        if status.strip():
            loops = [
                loop
                for loop in loops
                if loop.get("status", "").strip().lower() == status.strip().lower()
            ]
        return loops[-limit:]

    # --- Preferences ---

    def save_preference(
        self,
        category: str,
        value: str,
        context: str = "",
    ) -> dict:
        """Save or update a user preference."""
        preferences = self._read(self.preferences_file, [])
        key = (category.strip().lower(), value.strip().lower())
        item = {
            "category": category.strip(),
            "value": value.strip(),
            "context": context.strip(),
            "updated_at": self._now(),
        }

        for index, existing in enumerate(preferences):
            existing_key = (
                existing.get("category", "").strip().lower(),
                existing.get("value", "").strip().lower(),
            )
            if existing_key == key:
                preferences[index] = item
                self._write(self.preferences_file, preferences)
                return item

        preferences.append(item)
        self._write(self.preferences_file, preferences)
        return item

    def get_preferences(self, category: str = "", limit: int = 20) -> list:
        """Get saved preferences, optionally filtered by category."""
        preferences = self._read(self.preferences_file, [])
        if category.strip():
            preferences = [
                item
                for item in preferences
                if item.get("category", "").strip().lower() == category.strip().lower()
            ]
        return preferences[-limit:]

    # --- Calendar ---

    def save_calendar_item(
        self,
        title: str,
        date: str,
        time: str = "",
        details: str = "",
        status: str = "planned",
        source: str = "conversation",
    ) -> dict:
        """Save or update a calendar item."""
        items = self._read(self.calendar_file, [])
        key = (title.strip().lower(), date.strip().lower(), time.strip().lower())
        item = {
            "title": title.strip(),
            "date": date.strip(),
            "time": time.strip(),
            "details": details.strip(),
            "status": status.strip() or "planned",
            "source": source.strip() or "conversation",
            "updated_at": self._now(),
        }

        for index, existing in enumerate(items):
            existing_key = (
                existing.get("title", "").strip().lower(),
                existing.get("date", "").strip().lower(),
                existing.get("time", "").strip().lower(),
            )
            if existing_key == key:
                items[index] = item
                self._write(self.calendar_file, items)
                return item

        items.append(item)
        items.sort(key=lambda entry: (entry.get("date", ""), entry.get("time", ""), entry.get("title", "")))
        self._write(self.calendar_file, items)
        return item

    def update_calendar_item_status(
        self,
        title: str,
        status: str,
        date: str = "",
        time: str = "",
    ) -> dict | None:
        """Update a calendar item's status by title and optional date/time."""
        items = self._read(self.calendar_file, [])
        title_key = title.strip().lower()
        date_key = date.strip().lower()
        time_key = time.strip().lower()
        for index, existing in enumerate(items):
            if existing.get("title", "").strip().lower() != title_key:
                continue
            if date_key and existing.get("date", "").strip().lower() != date_key:
                continue
            if time_key and existing.get("time", "").strip().lower() != time_key:
                continue
            items[index]["status"] = status.strip() or existing.get("status", "planned")
            items[index]["updated_at"] = self._now()
            self._write(self.calendar_file, items)
            return items[index]
        return None

    def get_calendar_items(self, status: str = "", limit: int = 20) -> list:
        """Get calendar items, optionally filtered by status.

        Dated items are ordered by proximity to today so the most relevant
        recent or upcoming events surface first, while undated items stay last.
        """
        items = self._read(self.calendar_file, [])
        if status.strip():
            items = [
                item
                for item in items
                if item.get("status", "").strip().lower() == status.strip().lower()
            ]
        today = date.today()
        dated = []
        undated = []

        for item in items:
            parsed_date = self._parse_date(item.get("date", ""))
            if parsed_date is None:
                undated.append(item)
                continue
            dated.append((parsed_date, item))

        dated.sort(
            key=lambda entry: (
                abs((entry[0] - today).days),
                entry[0],
                entry[1].get("time", ""),
                entry[1].get("title", ""),
            )
        )
        undated.sort(key=lambda item: (item.get("title", ""), item.get("updated_at", "")))
        return ([item for _, item in dated] + undated)[:limit]

    # --- Favorites and recommendations ---

    def save_favorite(
        self,
        category: str,
        title: str,
        reason: str = "",
        creator: str = "",
        status: str = "saved",
    ) -> dict:
        """Save or update a recommendation or favorite."""
        favorites = self._read(self.favorites_file, [])
        key = (category.strip().lower(), title.strip().lower())
        item = {
            "category": category.strip(),
            "title": title.strip(),
            "reason": reason.strip(),
            "creator": creator.strip(),
            "status": status.strip() or "saved",
            "updated_at": self._now(),
        }

        for index, existing in enumerate(favorites):
            existing_key = (
                existing.get("category", "").strip().lower(),
                existing.get("title", "").strip().lower(),
            )
            if existing_key == key:
                favorites[index] = item
                self._write(self.favorites_file, favorites)
                return item

        favorites.append(item)
        self._write(self.favorites_file, favorites)
        return item

    def get_favorites(self, category: str = "", limit: int = 20) -> list:
        """Get favorites or recommendations."""
        favorites = self._read(self.favorites_file, [])
        if category.strip():
            favorites = [
                item
                for item in favorites
                if item.get("category", "").strip().lower() == category.strip().lower()
            ]
        return favorites[-limit:]

    # --- Wellness plans ---

    def save_gym_plan(
        self,
        name: str,
        schedule: str,
        goal: str = "",
        notes: str = "",
        status: str = "active",
    ) -> dict:
        """Save or update a gym plan."""
        plans = self._read(self.gym_file, [])
        key = name.strip().lower()
        item = {
            "name": name.strip(),
            "schedule": schedule.strip(),
            "goal": goal.strip(),
            "notes": notes.strip(),
            "status": status.strip() or "active",
            "updated_at": self._now(),
        }

        for index, existing in enumerate(plans):
            if existing.get("name", "").strip().lower() == key:
                plans[index] = item
                self._write(self.gym_file, plans)
                return item

        plans.append(item)
        self._write(self.gym_file, plans)
        return item

    def get_gym_plans(self, status: str = "", limit: int = 10) -> list:
        """Get gym plans, optionally filtered by status."""
        plans = self._read(self.gym_file, [])
        if status.strip():
            plans = [
                plan
                for plan in plans
                if plan.get("status", "").strip().lower() == status.strip().lower()
            ]
        return plans[-limit:]

    def save_food_program(
        self,
        name: str,
        goal: str,
        daily_structure: str,
        notes: str = "",
        status: str = "active",
    ) -> dict:
        """Save or update a nutrition or meal program."""
        programs = self._read(self.food_file, [])
        key = name.strip().lower()
        item = {
            "name": name.strip(),
            "goal": goal.strip(),
            "daily_structure": daily_structure.strip(),
            "notes": notes.strip(),
            "status": status.strip() or "active",
            "updated_at": self._now(),
        }

        for index, existing in enumerate(programs):
            if existing.get("name", "").strip().lower() == key:
                programs[index] = item
                self._write(self.food_file, programs)
                return item

        programs.append(item)
        self._write(self.food_file, programs)
        return item

    def get_food_programs(self, status: str = "", limit: int = 10) -> list:
        """Get nutrition programs, optionally filtered by status."""
        programs = self._read(self.food_file, [])
        if status.strip():
            programs = [
                program
                for program in programs
                if program.get("status", "").strip().lower() == status.strip().lower()
            ]
        return programs[-limit:]

    # --- Workout sessions ---

    def log_workout_session(
        self,
        plan_name: str,
        exercises: str = "",
        duration_min: int = 0,
        notes: str = "",
        energy_rating: int = 0,
    ) -> dict:
        """Log an actual completed workout session."""
        sessions = self._read(self.workout_sessions_file, [])
        item = {
            "date": date.today().isoformat(),
            "plan_name": plan_name.strip(),
            "exercises": exercises.strip(),
            "duration_min": max(0, int(duration_min)),
            "notes": notes.strip(),
            "energy_rating": max(0, min(5, int(energy_rating))),
            "timestamp": self._now(),
        }
        sessions.append(item)
        self._write(self.workout_sessions_file, sessions)
        return item

    def get_workout_sessions(self, limit: int = 10) -> list:
        """Get the most recent workout sessions."""
        return self._read(self.workout_sessions_file, [])[-limit:]

    def get_today_workout(self) -> dict | None:
        """Get today's workout session if logged."""
        today = date.today().isoformat()
        for session in reversed(self._read(self.workout_sessions_file, [])):
            if session.get("date") == today:
                return session
        return None

    # --- Body metrics ---

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
        """Log today's body metrics. Replaces any existing entry for today."""
        metrics = self._read(self.body_metrics_file, [])
        today = date.today().isoformat()
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
        for index, existing in enumerate(metrics):
            if existing.get("date") == today:
                metrics[index] = item
                self._write(self.body_metrics_file, metrics)
                return item
        metrics.append(item)
        self._write(self.body_metrics_file, metrics)
        return item

    def get_body_metrics(self, days: int = 30) -> list:
        """Get body metrics for the last N entries."""
        return self._read(self.body_metrics_file, [])[-days:]

    def get_today_body_metrics(self) -> dict | None:
        """Get today's body metrics entry if it exists."""
        today = date.today().isoformat()
        for item in reversed(self._read(self.body_metrics_file, [])):
            if item.get("date") == today:
                return item
        return None

    # --- Habits ---

    def create_or_update_habit(
        self,
        name: str,
        category: str = "health",
        target_days: str = "daily",
    ) -> dict:
        """Create a new habit or update its metadata. Preserves existing completions."""
        habits = self._read(self.habits_file, [])
        key = name.strip().lower()
        for index, existing in enumerate(habits):
            if existing.get("name", "").strip().lower() == key:
                habits[index]["category"] = category.strip() or "health"
                habits[index]["target_days"] = target_days.strip() or "daily"
                self._write(self.habits_file, habits)
                return habits[index]
        item = {
            "name": name.strip(),
            "category": category.strip() or "health",
            "target_days": target_days.strip() or "daily",
            "completions": [],
            "created_at": self._now(),
        }
        habits.append(item)
        self._write(self.habits_file, habits)
        return item

    def log_habit_completion(self, habit_name: str, date_str: str = "") -> dict:
        """Mark a habit as done. Auto-creates the habit if not found."""
        habits = self._read(self.habits_file, [])
        target_date = date_str.strip() if date_str.strip() else date.today().isoformat()
        key = habit_name.strip().lower()
        for index, habit in enumerate(habits):
            if habit.get("name", "").strip().lower() == key:
                completions = habit.get("completions", [])
                if target_date not in completions:
                    completions.append(target_date)
                    completions.sort()
                    habits[index]["completions"] = completions
                    self._write(self.habits_file, habits)
                return self._habit_snapshot(habits[index])
        item = {
            "name": habit_name.strip(),
            "category": "health",
            "target_days": "daily",
            "completions": [target_date],
            "created_at": self._now(),
        }
        habits.append(item)
        self._write(self.habits_file, habits)
        return self._habit_snapshot(item)

    def get_habits(self) -> list:
        """Get all habits with computed streak and today's completion status."""
        habits = self._read(self.habits_file, [])
        today = date.today()
        result = []
        for habit in habits:
            completions = set(habit.get("completions", []))
            done_today = today.isoformat() in completions
            result.append({**habit, "done_today": done_today, "streak": self._habit_streak(completions, today)})
        return result

    # --- Nutrition logs ---

    def log_nutrition(
        self,
        meal: str,
        description: str,
        calories_est: int = 0,
        protein_est: int = 0,
        notes: str = "",
    ) -> dict:
        """Log a meal entry for today."""
        logs = self._read(self.nutrition_file, [])
        item = {
            "date": date.today().isoformat(),
            "meal": meal.strip().lower(),
            "description": description.strip(),
            "calories_est": max(0, int(calories_est)),
            "protein_est": max(0, int(protein_est)),
            "notes": notes.strip(),
            "timestamp": self._now(),
        }
        logs.append(item)
        self._write(self.nutrition_file, logs)
        return item

    def get_nutrition_today(self) -> list:
        """Get all meal logs for today."""
        today = date.today().isoformat()
        return [e for e in self._read(self.nutrition_file, []) if e.get("date") == today]

    def get_nutrition_recent(self, limit: int = 28) -> list:
        """Get recent meal log entries."""
        return self._read(self.nutrition_file, [])[-limit:]

    # --- Water tracking ---

    def log_water(self, glasses: int = 1) -> int:
        """Add glasses to today's water count. Returns updated daily total."""
        logs = self._read(self.water_file, {})
        today = date.today().isoformat()
        logs[today] = max(0, logs.get(today, 0) + int(glasses))
        self._write(self.water_file, logs)
        return logs[today]

    def set_water(self, glasses: int) -> int:
        """Set today's water count directly."""
        logs = self._read(self.water_file, {})
        today = date.today().isoformat()
        logs[today] = max(0, int(glasses))
        self._write(self.water_file, logs)
        return logs[today]

    def get_water_today(self) -> int:
        """Get today's total water glass count."""
        return self._read(self.water_file, {}).get(date.today().isoformat(), 0)

    # --- Chapter completeness ---

    def get_chapter_completeness(self) -> dict:
        """Return item count per chapter for completeness checking."""
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
        """Return chapter display names where no data has been saved yet."""
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

    # --- Today summary ---

    def get_today_summary(self) -> dict:
        """Get a comprehensive summary of today's tracked activity."""
        today_metrics = self.get_today_body_metrics()
        today_workout = self.get_today_workout()
        today_nutrition = self.get_nutrition_today()
        habits = self.get_habits()
        water = self.get_water_today()
        habits_done = [h for h in habits if h.get("done_today")]
        habits_pending = [h for h in habits if not h.get("done_today")]
        total_calories = sum(e.get("calories_est", 0) for e in today_nutrition)
        total_protein = sum(e.get("protein_est", 0) for e in today_nutrition)
        return {
            "date": date.today().isoformat(),
            "body_metrics": today_metrics,
            "recent_body_metrics": self.get_body_metrics(days=7),
            "workout": today_workout,
            "habits_total": len(habits),
            "habits_done": len(habits_done),
            "habits_done_names": [h["name"] for h in habits_done],
            "habits_pending_names": [h["name"] for h in habits_pending],
            "water_glasses": water,
            "meals_logged": len(today_nutrition),
            "calories_est": total_calories,
            "protein_est": total_protein,
        }

    # --- App behavior ---

    def get_app_state(self) -> dict:
        """Get persisted UI and assistant app state."""
        return self._read(self.app_state_file, {})

    def set_app_state_value(self, key: str, value) -> dict:
        """Store one app state value."""
        state = self.get_app_state()
        state[key] = value
        self._write(self.app_state_file, state)
        return state

    def should_send_daily_checkin(self) -> bool:
        """Return whether June should send the daily proactive check-in."""
        state = self.get_app_state()
        return state.get("last_daily_checkin_date") != date.today().isoformat()

    def mark_daily_checkin_sent(self) -> None:
        """Persist that today's check-in has been sent."""
        self.set_app_state_value("last_daily_checkin_date", date.today().isoformat())

    def get_upcoming_notifications(self, limit: int = 5) -> list[dict]:
        """Build notification items from calendar and planning memory."""
        today = date.today()
        notifications = []

        for item in self.get_calendar_items(status="", limit=50):
            parsed_date = self._parse_date(item.get("date", ""))
            if parsed_date is None:
                continue
            status = item.get("status", "").strip().lower()
            if status in {"completed", "done", "canceled", "cancelled", "archived"}:
                continue
            days_until = (parsed_date - today).days
            if 0 <= days_until <= 14:
                notifications.append({
                    "title": item.get("title", "Event"),
                    "kind": self._infer_calendar_kind(item),
                    "when": item.get("date", ""),
                    "details": item.get("details", ""),
                    "days_until": days_until,
                })

        for loop in self.get_open_loops(status="open", limit=20):
            parsed_date = self._parse_date(loop.get("due_date", ""))
            if parsed_date is None:
                continue
            days_until = (parsed_date - today).days
            if 0 <= days_until <= 14:
                notifications.append({
                    "title": loop.get("topic", "Open loop"),
                    "kind": "plan",
                    "when": loop.get("due_date", ""),
                    "details": loop.get("next_step", ""),
                    "days_until": days_until,
                })

        notifications.sort(key=lambda item: (item["days_until"], item["when"], item["title"]))
        return notifications[:limit]

    # --- Summary ---

    def get_progress_snapshot(self) -> dict:
        """Summarize the user's recent assistant activity."""
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
        active_goals = [goal for goal in goals if goal.get("status") == "active"]
        unresolved_loops = [
            loop for loop in open_loops if loop.get("status", "open") == "open"
        ]
        habits_done_today = len([h for h in habits if h.get("done_today")])

        return {
            "mood_count": len(moods),
            "latest_mood": moods[-1]["mood"] if moods else "",
            "journal_count": len(journal),
            "relationship_count": len(relationships),
            "goal_count": len(goals),
            "active_goal_count": len(active_goals),
            "open_loop_count": len(unresolved_loops),
            "preference_count": len(preferences),
            "calendar_count": len(calendar_items),
            "favorite_count": len(favorites),
            "gym_plan_count": len(gym_plans),
            "food_program_count": len(food_programs),
            "habit_count": len(habits),
            "habits_done_today": habits_done_today,
            "water_today": self.get_water_today(),
            "workout_session_count": len(self.get_workout_sessions(limit=50)),
        }

    # --- Internal helpers ---

    def _read(self, path: Path, default):
        if path.exists():
            with open(path) as file:
                raw = file.read()
            if not raw.strip():
                return default
            try:
                return json.loads(raw)
            except JSONDecodeError:
                recovered = self._recover_json(raw, default)
                self._write(path, recovered)
                return recovered
        return default

    def _write(self, path: Path, data) -> None:
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        with open(temp_path, "w") as file:
            json.dump(data, file, indent=2)
        temp_path.replace(path)

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _recover_json(self, raw: str, default):
        decoder = json.JSONDecoder()
        cleaned = raw.lstrip()
        if not cleaned:
            return default
        try:
            parsed, _index = decoder.raw_decode(cleaned)
        except JSONDecodeError:
            return default
        if isinstance(parsed, type(default)):
            return parsed
        return default

    def _parse_date(self, value: str) -> date | None:
        try:
            return date.fromisoformat(value.strip())
        except (TypeError, ValueError, AttributeError):
            return None

    def _infer_calendar_kind(self, item: dict) -> str:
        haystack = " ".join(
            str(item.get(field, "")).lower()
            for field in ("title", "details", "source", "status")
        )
        if "birthday" in haystack:
            return "birthday"
        if "trip" in haystack or "travel" in haystack or "flight" in haystack:
            return "trip"
        if "date" in haystack or "anniversary" in haystack:
            return "dating"
        return "calendar"

    def _habit_streak(self, completions: set[str], start_date: date | None = None) -> int:
        """Count the current streak ending at start_date or today."""
        from datetime import timedelta

        check = start_date or date.today()
        streak = 0
        while check.isoformat() in completions:
            streak += 1
            check = check - timedelta(days=1)
        return streak

    def _habit_snapshot(self, habit: dict) -> dict:
        """Return a habit record with computed streak metadata."""
        completions = set(habit.get("completions", []))
        today = date.today()
        return {
            **habit,
            "done_today": today.isoformat() in completions,
            "streak": self._habit_streak(completions, today),
        }
