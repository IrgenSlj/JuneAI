"""Session-state helpers for the JuneAI app shell."""

from __future__ import annotations

from typing import Any

from agent.memory import Memory
from agent.skills import DEFAULT_SKILL

from .layout import normalize_rail_view
from .rendering import default_ui_state


def default_tool_stats() -> dict[str, Any]:
    """Return the default tool stats payload."""
    return {"requested": 0, "succeeded": 0, "failed": 0, "last_calls": []}


def sync_selected_chapter(session_state: Any, chapter_key: str = "") -> str:
    """Keep local chapter selection aligned with UI state."""
    normalized = chapter_key.strip().lower()
    session_state["selected_chapter"] = normalized
    session_state["ui_state"]["selected_chapter"] = normalized
    return normalized


def sync_shell_flags(session_state: Any) -> tuple[str, bool]:
    """Mirror shell flags from ui_state onto top-level session state."""
    selected = str(session_state["ui_state"].get("selected_chapter", "")).strip().lower()
    visible = bool(session_state["ui_state"].get("show_right_panel", True))
    session_state["selected_chapter"] = selected
    session_state["show_right_panel"] = visible
    return selected, visible


def shell_state_defaults(
    user_id: str,
    *,
    messages: tuple[Any, ...] | list[Any] = (),
    activity_log: tuple[str, ...] | list[str] = (),
    ui_state: dict[str, Any] | None = None,
    tool_stats: dict[str, Any] | None = None,
    rail_view: str = "today",
) -> dict[str, Any]:
    """Build a normalized shell-state payload."""
    merged_ui = default_ui_state()
    if ui_state:
        merged_ui.update({key: value for key, value in ui_state.items() if key in merged_ui or key in {"selected_chapter", "show_right_panel"}})
    merged_ui["layout"] = str(merged_ui.get("layout", "split")).strip().lower()
    merged_ui["selected_chapter"] = str(merged_ui.get("selected_chapter", "")).strip().lower()
    merged_ui["show_right_panel"] = bool(merged_ui.get("show_right_panel", True))
    merged_ui["checklist_items"] = [str(item) for item in merged_ui.get("checklist_items", [])]

    normalized_tool_stats = default_tool_stats()
    if tool_stats:
        normalized_tool_stats.update(
            {
                "requested": int(tool_stats.get("requested", 0)),
                "succeeded": int(tool_stats.get("succeeded", 0)),
                "failed": int(tool_stats.get("failed", 0)),
                "last_calls": [dict(item) for item in tool_stats.get("last_calls", [])],
            }
        )

    return {
        "messages": list(messages),
        "last_user_id": user_id,
        "activity_log": list(activity_log),
        "ui_state": merged_ui,
        "live_response": "",
        "final_state": None,
        "is_generating": False,
        "pending_prompt": "",
        "active_skill_key": DEFAULT_SKILL,
        "selected_chapter": merged_ui["selected_chapter"],
        "tool_stats": normalized_tool_stats,
        "show_right_panel": merged_ui["show_right_panel"],
        "rail_view": normalize_rail_view(rail_view),
    }


def initialize_session_state(session_state: Any, user_id: str) -> None:
    """Initialize the app shell state if keys are missing."""
    defaults = shell_state_defaults(user_id)
    if "messages" not in session_state:
        defaults["messages"] = Memory(user_id).load_chat_messages()
    for key, value in defaults.items():
        if key not in session_state:
            session_state[key] = value


def reset_shell_state(
    session_state: Any,
    user_id: str,
    *,
    messages: tuple[Any, ...] | list[Any] | None = None,
    ui_state: dict[str, Any] | None = None,
    tool_stats: dict[str, Any] | None = None,
    rail_view: str = "today",
) -> dict[str, Any]:
    """Reset shell state and return the updated mapping."""
    default_messages = Memory(user_id).load_chat_messages() if messages is None else list(messages)
    state = shell_state_defaults(
        user_id,
        messages=tuple(default_messages),
        ui_state=ui_state,
        tool_stats=tool_stats,
        rail_view=rail_view,
    )
    session_state.update(state)
    return dict(session_state)


def reset_session_state(session_state: Any, user_id: str) -> None:
    """Reset the app shell state for a clear chat or profile switch."""
    reset_shell_state(session_state, user_id)


__all__ = [
    "default_tool_stats",
    "initialize_session_state",
    "reset_session_state",
    "reset_shell_state",
    "shell_state_defaults",
    "sync_selected_chapter",
    "sync_shell_flags",
]
