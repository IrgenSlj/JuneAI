"""JuneAI memory system.

Stores conversation history, mood logs, journal entries, and structured
relationship planning data as JSON files locally.
"""

import json
from datetime import datetime
from pathlib import Path

from .config import MEMORY_DIR


class Memory:
    """Manages all persistent storage for a single user."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.dir = Path(MEMORY_DIR)
        self.dir.mkdir(exist_ok=True)

        # Each user gets their own files
        self.chat_file = self.dir / f"{user_id}_chat.json"
        self.mood_file = self.dir / f"{user_id}_moods.json"
        self.journal_file = self.dir / f"{user_id}_journal.json"
        self.relationship_file = self.dir / f"{user_id}_relationships.json"
        self.goals_file = self.dir / f"{user_id}_goals.json"
        self.open_loops_file = self.dir / f"{user_id}_open_loops.json"

    # --- Chat history ---

    def save_message(self, role: str, content: str) -> None:
        """Append a message to chat history (keeps last 50)."""
        history = self.load_chat()
        history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        if len(history) > 50:
            history = history[-50:]
        self._write(self.chat_file, history)

    def load_chat(self) -> list:
        """Load full chat history."""
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
            "mood": mood,
            "note": note,
            "timestamp": datetime.now().isoformat(),
        }
        moods.append(entry)
        self._write(self.mood_file, moods)
        return entry

    def get_mood_history(self, limit: int = 10) -> list:
        """Get the most recent mood entries."""
        return self._read(self.mood_file, [])[-limit:]

    # --- Journal ---

    def save_journal(self, entry: str) -> dict:
        """Save a journal or therapy note."""
        journal = self._read(self.journal_file, [])
        item = {
            "entry": entry,
            "timestamp": datetime.now().isoformat(),
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
        now = datetime.now().isoformat()
        item = {
            "person": person.strip(),
            "relationship": relationship.strip(),
            "summary": summary.strip(),
            "user_needs": user_needs.strip(),
            "cautions": cautions.strip(),
            "updated_at": now,
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
        now = datetime.now().isoformat()
        item = {
            "title": title.strip(),
            "category": category.strip() or "personal",
            "target_date": target_date.strip(),
            "next_step": next_step.strip(),
            "status": status.strip() or "active",
            "updated_at": now,
        }

        for index, existing in enumerate(goals):
            if existing.get("title", "").strip().lower() == key:
                goals[index] = item
                self._write(self.goals_file, goals)
                return item

        goals.append(item)
        self._write(self.goals_file, goals)
        return item

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
        now = datetime.now().isoformat()
        item = {
            "topic": topic.strip(),
            "next_step": next_step.strip(),
            "due_date": due_date.strip(),
            "status": status.strip() or "open",
            "updated_at": now,
        }

        for index, existing in enumerate(loops):
            if existing.get("topic", "").strip().lower() == key:
                loops[index] = item
                self._write(self.open_loops_file, loops)
                return item

        loops.append(item)
        self._write(self.open_loops_file, loops)
        return item

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

    def get_progress_snapshot(self) -> dict:
        """Summarize the user's recent activity across memory types."""
        moods = self.get_mood_history(10)
        journal = self.get_journal(5)
        goals = self.get_goals(limit=20)
        open_loops = self.get_open_loops(status="", limit=20)
        relationships = self.get_relationship_profiles()
        active_goals = [goal for goal in goals if goal.get("status") == "active"]
        unresolved_loops = [
            loop for loop in open_loops if loop.get("status", "open") == "open"
        ]

        return {
            "mood_count": len(moods),
            "latest_mood": moods[-1]["mood"] if moods else "",
            "journal_count": len(journal),
            "relationship_count": len(relationships),
            "goal_count": len(goals),
            "active_goal_count": len(active_goals),
            "open_loop_count": len(unresolved_loops),
        }

    # --- Internal helpers ---

    def _read(self, path: Path, default):
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return default

    def _write(self, path: Path, data) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
