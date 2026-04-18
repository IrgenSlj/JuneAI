"""June daily skill — journaling, moods, goals, and open loops via MCP."""

from __future__ import annotations

from june_brain.memory import Memory
from june_brain.skills.server import MCPStdioServer

server = MCPStdioServer(name="june-daily", version="0.1.0")


@server.tool(
    name="log_mood",
    description="Log the user's emotional state.",
    input_schema={
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "mood": {"type": "string"},
            "note": {"type": "string", "default": ""},
        },
        "required": ["user_id", "mood"],
    },
)
def log_mood(user_id: str, mood: str, note: str = "") -> str:
    Memory(user_id).log_mood(mood, note)
    return f"Noted your mood as '{mood}'."


@server.tool(
    name="save_journal_entry",
    description="Save a reflection or conversation note to the journal.",
    input_schema={
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "entry": {"type": "string"},
        },
        "required": ["user_id", "entry"],
    },
)
def save_journal_entry(user_id: str, entry: str) -> str:
    Memory(user_id).save_journal(entry)
    return "Saved to your journal."


@server.tool(
    name="track_goal",
    description="Create or update a goal with category, next step, and target date.",
    input_schema={
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "title": {"type": "string"},
            "category": {"type": "string", "default": "personal"},
            "target_date": {"type": "string", "default": ""},
            "next_step": {"type": "string", "default": ""},
            "status": {"type": "string", "default": "active"},
        },
        "required": ["user_id", "title"],
    },
)
def track_goal(
    user_id: str,
    title: str,
    category: str = "personal",
    target_date: str = "",
    next_step: str = "",
    status: str = "active",
) -> str:
    goal = Memory(user_id).save_goal(
        title=title,
        category=category,
        target_date=target_date,
        next_step=next_step,
        status=status,
    )
    return f"Added '{goal['title']}' to your goals."


@server.tool(
    name="save_open_loop",
    description="Track an unresolved issue, follow-up, or decision.",
    input_schema={
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "topic": {"type": "string"},
            "next_step": {"type": "string", "default": ""},
            "due_date": {"type": "string", "default": ""},
            "status": {"type": "string", "default": "open"},
        },
        "required": ["user_id", "topic"],
    },
)
def save_open_loop(
    user_id: str, topic: str, next_step: str = "", due_date: str = "", status: str = "open"
) -> str:
    item = Memory(user_id).save_open_loop(
        topic=topic, next_step=next_step, due_date=due_date, status=status
    )
    return f"Noted '{item['topic']}' as an open loop."


def main() -> None:
    server.run()
