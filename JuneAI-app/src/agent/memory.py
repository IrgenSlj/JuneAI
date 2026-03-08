"""JuneAI memory system.

Stores conversation history, mood logs, and journal entries
as JSON files locally. Simple, transparent, easy to understand.
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

    # --- Internal helpers ---

    def _read(self, path: Path, default):
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return default

    def _write(self, path: Path, data) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
