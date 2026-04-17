"""Runtime and shell helpers for the Streamlit app entrypoint."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from agent.memory import Memory

from .rendering import extract_text, render_activity, render_workspace, transcript_html


def append_activity(session_state: Any, message: str) -> None:
    """Append a log entry to the in-session activity feed."""
    session_state.activity_log.append(message)


def build_turn_save_summary(last_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Build a compact in-chat summary of what June saved this turn."""
    label_map = {
        "save_goal": "Goal saved",
        "save_open_loop": "Open loop saved",
        "save_calendar_item": "Calendar item saved",
        "save_relationship_profile": "Relationship note saved",
        "save_user_preference": "Preference saved",
        "save_favorite_recommendation": "Favorite saved",
        "save_journal_entry": "Journal saved",
        "log_body_metrics": "Body check-in saved",
        "log_workout_session": "Workout saved",
        "log_nutrition": "Meal log saved",
        "log_water": "Water updated",
        "log_habit_completion": "Habit completion saved",
        "track_goal": "Goal saved",
        "create_habit": "Habit saved",
    }
    chapter_map = {
        "save_goal": ("plans", "Plans"),
        "track_goal": ("plans", "Plans"),
        "save_open_loop": ("plans", "Plans"),
        "save_calendar_item": ("calendar", "Calendar"),
        "save_relationship_profile": ("family", "Family"),
        "log_body_metrics": ("body", "Body"),
        "log_workout_session": ("gym", "Gym"),
        "log_nutrition": ("food", "Food"),
        "log_water": ("water", "Water"),
        "log_habit_completion": ("habits", "Habits"),
        "create_habit": ("habits", "Habits"),
    }
    items: list[str] = []
    actions: list[dict[str, str]] = []
    for call in last_calls:
        if call.get("status") != "success":
            continue
        tool_name = str(call.get("name", "")).strip()
        if not tool_name.startswith(("save_", "log_", "track_", "create_")):
            continue
        preview = str(call.get("preview", "")).strip() or "Saved to memory."
        label = label_map.get(tool_name, tool_name.replace("_", " ").title())
        items.append(f"{label}: {preview}")
        chapter = chapter_map.get(tool_name)
        if chapter and not any(item.get("chapter_key") == chapter[0] for item in actions):
            actions.append({"chapter_key": chapter[0], "chapter_label": chapter[1]})
    if not items:
        return None
    return {
        "type": "save_summary",
        "label": "What June saved",
        "items": items[:5],
        "actions": actions[:3],
    }


def current_local_time() -> datetime:
    """Return the current local time with timezone awareness."""
    return datetime.now().astimezone()


def current_part_of_day(now: datetime) -> str:
    """Return the coarse part-of-day label for UI copy."""
    if 5 <= now.hour < 12:
        return "morning"
    if 12 <= now.hour < 17:
        return "afternoon"
    if 17 <= now.hour < 22:
        return "evening"
    return "night"


def phrase_bucket(now: datetime) -> str:
    """Return a stable 15-minute bucket key for rotating copy."""
    return f"{now.strftime('%Y-%m-%d')}-{now.hour:02d}-{now.minute // 15}"


def generate_sidebar_phrase(now: datetime) -> str:
    """Generate a deterministic sidebar phrase for the current time bucket."""
    openings = [
        "Health begins with gentle attention.",
        "A good life grows from small kindnesses to the body.",
        "Energy is built in quiet disciplines.",
        "Strength lasts longer when it is renewed by care.",
        "Well-being lives between motion, rest, and meaning.",
        "A clear day starts with one honest choice.",
    ]
    middles = [
        "Let today favour steady movement.",
        "Protect your peace like a serious habit.",
        "Feed the body with rhythm.",
        "Choose what leaves you lighter by tonight.",
        "Walk toward what strengthens you quietly.",
        "Keep your routines human and durable.",
    ]
    seed = now.timetuple().tm_yday + (now.hour * 4) + (now.minute // 15)
    return f"{openings[seed % len(openings)]} {middles[(seed * 3 + now.weekday()) % len(middles)]}"


def get_rotating_sidebar_phrase(memory: Memory, now: datetime) -> str:
    """Return the persisted phrase for the active time bucket."""
    bucket = phrase_bucket(now)
    state = memory.get_app_state()
    if state.get("sidebar_phrase_bucket") != bucket:
        phrase = generate_sidebar_phrase(now)
        memory.set_app_state_value("sidebar_phrase_text", phrase)
        memory.set_app_state_value("sidebar_phrase_bucket", bucket)
        return phrase
    return str(state.get("sidebar_phrase_text", generate_sidebar_phrase(now)))


def handle_stream_chunk(
    mode: str,
    data: Any,
    *,
    session_state: Any,
    transcript_placeholder: Any,
    workspace_placeholder: Any,
    activity_placeholder: Any,
    default_runtime_label: str,
    default_runtime_model: str,
    on_sync_selected_chapter: Callable[[Any, str], str],
    on_render_scroll_to_latest: Callable[[], None],
) -> None:
    """Apply one streamed agent event to Streamlit session state and placeholders."""
    if mode == "custom":
        event = data or {}
        if event.get("event") == "chat_started":
            append_activity(session_state, f"route | {event.get('skill')}")
            append_activity(
                session_state,
                "runtime | "
                + f"{event.get('runtime_label', session_state.get('current_runtime_label', default_runtime_label))}"
                + f" | {event.get('runtime_model', session_state.get('current_runtime_model', default_runtime_model))}",
            )
            session_state.active_tool_names = []
        elif event.get("event") == "tool_calls_requested":
            tools = event.get("tools", [])
            session_state.active_tool_names = tools
            append_activity(session_state, "tool request | " + ", ".join(tools))
        elif event.get("event") == "tool_results":
            session_state.active_tool_names = []
            summary = event.get("summary", {})
            append_activity(
                session_state,
                "tool results | "
                + f"{summary.get('succeeded', 0)} ok / {summary.get('failed', 0)} failed total",
            )
            for call in event.get("calls", []):
                append_activity(
                    session_state,
                    f"tool {call.get('status', 'unknown')} | "
                    f"{call.get('name', '?')} | {call.get('preview', '')}",
                )
        elif event.get("event") == "response_completed":
            append_activity(session_state, "response | direct answer")
        activity_placeholder.markdown(render_activity(session_state.activity_log), unsafe_allow_html=True)
        return

    if mode == "messages":
        message, _metadata = data
        if isinstance(message, AIMessageChunk):
            token_text = extract_text(message.content)
            if token_text:
                session_state.live_response += token_text
                if transcript_placeholder is not None:
                    transcript_placeholder.markdown(
                        transcript_html(session_state.messages, session_state.live_response),
                        unsafe_allow_html=True,
                    )
                on_render_scroll_to_latest()
            for chunk in getattr(message, "tool_call_chunks", []) or []:
                name = chunk.get("name")
                if name:
                    append_activity(session_state, f"planning | {name}")
        activity_placeholder.markdown(render_activity(session_state.activity_log), unsafe_allow_html=True)
        return

    if mode == "updates":
        for node_name, payload in (data or {}).items():
            append_activity(session_state, f"node | {node_name}")
            if isinstance(payload, dict):
                if "ui_state" in payload:
                    session_state.ui_state = payload["ui_state"]
                    on_sync_selected_chapter(session_state, session_state.ui_state.get("selected_chapter", ""))
                    if not session_state.selected_chapter:
                        workspace_placeholder.markdown(
                            render_workspace(session_state.ui_state, include_header=False),
                            unsafe_allow_html=True,
                        )
                for message in payload.get("messages", []):
                    if isinstance(message, ToolMessage):
                        append_activity(session_state, f"tool | {extract_text(message.content)}")
                    elif isinstance(message, AIMessage):
                        for tool_call in getattr(message, "tool_calls", []) or []:
                            append_activity(
                                session_state,
                                f"tool args | {tool_call.get('name')} {tool_call.get('args')}",
                            )
        activity_placeholder.markdown(render_activity(session_state.activity_log), unsafe_allow_html=True)
        return

    if mode == "values" and isinstance(data, dict):
        session_state.final_state = data
        if "tool_stats" in data:
            session_state.tool_stats = data["tool_stats"]
        if "ui_state" in data:
            session_state.ui_state = data["ui_state"]
            on_sync_selected_chapter(session_state, session_state.ui_state.get("selected_chapter", ""))
            if not session_state.selected_chapter:
                workspace_placeholder.markdown(
                    render_workspace(session_state.ui_state, include_header=False),
                    unsafe_allow_html=True,
                )
