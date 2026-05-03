"""June daily skill — journaling, moods, goals, and open loops via MCP.

Writes route through ``MemoryManager.write`` so each save lands in SQLite
*and* gets paraphrased into the vector store. Recall picks up these
paraphrases the same way it picks up extract-derived facts, so a goal
or journal entry written by this skill surfaces in subsequent turns
without the user's question having to overlap the structured fields.
"""

from __future__ import annotations

from june_brain.memory import MemoryManager
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
    result = MemoryManager(user_id).write(
        {"kind": "mood", "fields": {"mood": mood, "note": note}},
        source="skill:daily:log_mood",
    )
    if not result.get("written"):
        return "Couldn't note that mood."
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
    result = MemoryManager(user_id).write(
        {"kind": "journal", "fields": {"entry": entry}},
        source="skill:daily:save_journal_entry",
    )
    if not result.get("written"):
        return "Couldn't save that journal entry."
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
    result = MemoryManager(user_id).write(
        {
            "kind": "goal",
            "fields": {
                "title": title,
                "category": category,
                "target_date": target_date,
                "next_step": next_step,
                "status": status,
            },
        },
        source="skill:daily:track_goal",
    )
    if not result.get("written"):
        return "Couldn't save that goal."
    return f"Added '{title}' to your goals."


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
    result = MemoryManager(user_id).write(
        {
            "kind": "open_loop",
            "fields": {
                "topic": topic,
                "next_step": next_step,
                "due_date": due_date,
                "status": status,
            },
        },
        source="skill:daily:save_open_loop",
    )
    if not result.get("written"):
        return "Couldn't save that open loop."
    return f"Noted '{topic}' as an open loop."


def main() -> None:
    server.run()
