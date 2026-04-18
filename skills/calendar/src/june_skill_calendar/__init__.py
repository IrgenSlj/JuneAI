"""June calendar skill — save, list, and update calendar items via MCP."""

from __future__ import annotations

from june_brain.memory import Memory
from june_brain.skills.server import MCPStdioServer

server = MCPStdioServer(name="june-calendar", version="0.1.0")


@server.tool(
    name="save_calendar_item",
    description="Save a calendar item: event, appointment, reminder, or birthday.",
    input_schema={
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "title": {"type": "string"},
            "date": {"type": "string", "description": "ISO date YYYY-MM-DD."},
            "time": {"type": "string", "default": ""},
            "details": {"type": "string", "default": ""},
            "status": {"type": "string", "default": "planned"},
            "source": {"type": "string", "default": "conversation"},
        },
        "required": ["user_id", "title", "date"],
    },
)
def save_calendar_item(
    user_id: str,
    title: str,
    date: str,
    time: str = "",
    details: str = "",
    status: str = "planned",
    source: str = "conversation",
) -> str:
    item = Memory(user_id).save_calendar_item(
        title=title, date=date, time=time, details=details, status=status, source=source
    )
    return f"Saved '{item['title']}' on {item['date']}."


@server.tool(
    name="list_calendar_items",
    description="List saved calendar items, optionally filtered by status.",
    input_schema={
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "status": {"type": "string", "default": ""},
        },
        "required": ["user_id"],
    },
)
def list_calendar_items(user_id: str, status: str = "") -> str:
    items = Memory(user_id).get_calendar_items(status=status)
    if not items:
        return "No calendar items."
    lines = []
    for item in items:
        line = f"- {item['date']}"
        if item.get("time"):
            line += f" {item['time']}"
        line += f": {item['title']}"
        if item.get("details"):
            line += f" | {item['details']}"
        if item.get("status"):
            line += f" [{item['status']}]"
        lines.append(line)
    return "Calendar:\n" + "\n".join(lines)


@server.tool(
    name="update_calendar_item_status",
    description="Update a calendar item's status (done, cancelled, planned, etc.).",
    input_schema={
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "title": {"type": "string"},
            "status": {"type": "string"},
            "date": {"type": "string", "default": ""},
            "time": {"type": "string", "default": ""},
        },
        "required": ["user_id", "title", "status"],
    },
)
def update_calendar_item_status(
    user_id: str, title: str, status: str, date: str = "", time: str = ""
) -> str:
    item = Memory(user_id).update_calendar_item_status(
        title=title, status=status, date=date, time=time
    )
    if not item:
        return f"No item matched '{title}'."
    return f"Updated '{item['title']}' to {item['status']}."


def main() -> None:
    server.run()
