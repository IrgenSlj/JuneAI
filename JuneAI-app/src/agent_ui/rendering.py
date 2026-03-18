"""Reusable HTML rendering helpers for the JuneAI Streamlit app."""

from __future__ import annotations

import html
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage

from agent.memory import Memory
from agent.tools import DEFAULT_UI_STATE

from .chapters import chapter_items, chapter_subtitle


def default_ui_state() -> dict:
    return {
        "layout": DEFAULT_UI_STATE["layout"],
        "selected_chapter": DEFAULT_UI_STATE["selected_chapter"],
        "focus_title": DEFAULT_UI_STATE["focus_title"],
        "focus_body": DEFAULT_UI_STATE["focus_body"],
        "checklist_title": DEFAULT_UI_STATE["checklist_title"],
        "checklist_items": list(DEFAULT_UI_STATE["checklist_items"]),
        "notice": DEFAULT_UI_STATE["notice"],
    }


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def habit_ring_svg(done: bool, size: int = 26) -> str:
    r = 10
    c = size // 2
    circumference = round(2 * 3.14159 * r, 2)
    offset = 0 if done else circumference
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
        f'style="display:block;flex-shrink:0;">'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" '
        f'stroke="rgba(22,20,16,0.10)" stroke-width="2"/>'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" '
        f'stroke="#0f5f4a" stroke-width="2" '
        f'stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" '
        f'transform="rotate(-90 {c} {c})" '
        f'style="transition:stroke-dashoffset 0.55s cubic-bezier(0.34,1.56,0.64,1);"/>'
        f'</svg>'
    )


def water_dots_html(count: int, goal: int = 8) -> str:
    dots = []
    for i in range(goal):
        css = "filled" if i < count else "empty"
        delay = f"animation-delay:{i * 0.04}s;" if i < count else ""
        dots.append(f'<span class="june-water-dot {css}" style="{delay}"></span>')
    return f'<div class="june-water-track">{"".join(dots)}</div>'


def energy_dots_html(value: int, max_val: int = 5) -> str:
    dots = "".join(
        f'<span class="june-metric-dot {"active" if i < value else "inactive"}"></span>'
        for i in range(max_val)
    )
    return f'<div class="june-metric-dots">{dots}</div>'


def transcript_html(messages: list, live_response: str = "") -> str:
    blocks = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            blocks.append(
                '<div class="june-message june-message-user">'
                '<div class="june-message-label">You</div>'
                f"{html.escape(extract_text(msg.content))}"
                "</div>"
            )
        elif isinstance(msg, AIMessage) and msg.content:
            blocks.append(
                '<div class="june-message june-message-assistant">'
                '<div class="june-message-label">June</div>'
                f"{html.escape(extract_text(msg.content))}"
                "</div>"
            )
    if live_response:
        blocks.append(
            '<div class="june-message june-message-assistant">'
            '<div class="june-message-label">June</div>'
            f"{html.escape(live_response)}"
            "</div>"
        )
    return (
        '<div class="june-transcript" id="june-transcript">'
        + "".join(blocks)
        + '<div id="june-transcript-end"></div>'
        + "</div>"
        + """<script>
        setTimeout(() => {
            const doc = window.parent.document;
            const t = doc.getElementById("june-transcript");
            const end = doc.getElementById("june-transcript-end");
            if (t) t.scrollTop = t.scrollHeight;
            if (end) end.scrollIntoView({ block: "end" });
        }, 0);
        </script>"""
    )


def render_list(items: list[tuple[str, str]]) -> str:
    if not items:
        return '<div class="june-item-meta">Nothing saved yet.</div>'
    return '<div class="june-list">' + "".join(
        f'<div class="june-item">'
        f'<div class="june-item-title">{html.escape(title)}</div>'
        f'<div class="june-item-meta">{html.escape(meta)}</div>'
        f'</div>'
        for title, meta in items
    ) + "</div>"


def render_workspace(ui_state: dict, include_header: bool = True) -> str:
    checklist_items = ui_state.get("checklist_items", [])
    checklist = (
        "".join(f"<li>{html.escape(item)}</li>" for item in checklist_items)
        if checklist_items
        else "<li>No pinned actions.</li>"
    )
    if include_header:
        header = (
            f'<div class="june-label">Workspace</div>'
            f'<h3 class="june-title">{html.escape(ui_state.get("focus_title", "Workspace"))}</h3>'
        )
    else:
        header = f'<h3 class="june-title">{html.escape(ui_state.get("focus_title", "Workspace"))}</h3>'

    return header + (
        f'<div class="june-item-meta">{html.escape(ui_state.get("focus_body", ""))}</div>'
        f'<div class="june-label" style="margin-top:0.6rem;">'
        f'{html.escape(ui_state.get("checklist_title", "Next steps"))}</div>'
        f'<div class="june-item-meta"><ul>{checklist}</ul></div>'
        + (
            f'<div class="june-item-meta" style="margin-top:0.5rem;">'
            f'{html.escape(ui_state.get("notice", ""))}</div>'
            if ui_state.get("notice")
            else ""
        )
    )


def render_memory_focus(memory: Memory, chapter_key: str) -> str:
    items = chapter_items(memory, chapter_key)
    return f'<div class="june-subtitle">{html.escape(chapter_subtitle(chapter_key))}</div>{render_list(items)}'


def render_activity(activity_log: list[str]) -> str:
    if not activity_log:
        return '<div class="june-item-meta">No activity yet.</div>'
    return '<div class="june-list">' + "".join(
        f'<div class="june-item">'
        f'<div class="june-item-meta">{html.escape(line)}</div>'
        f'</div>'
        for line in activity_log[-10:]
    ) + "</div>"


def render_notifications(memory: Memory) -> str:
    notifications = memory.get_upcoming_notifications(limit=5)
    items = []
    for item in notifications:
        prefix = "today" if item["days_until"] == 0 else f"in {item['days_until']}d"
        items.append(
            (
                item["title"],
                f"{item['kind']} · {item['when']} · {prefix}"
                f"{' | ' + item['details'] if item.get('details') else ''}",
            )
        )
    return render_list(items)


def render_capture_health(memory: Memory, activity_log: list[str]) -> str:
    chapter_counts = {
        "Agenda": len(memory.get_calendar_items(limit=100)),
        "Habits": len(memory.get_habits()),
        "Body": len(memory.get_body_metrics(days=30)),
        "Workout Sessions": len(memory.get_workout_sessions(limit=100)),
        "Nutrition": len(memory.get_nutrition_recent(limit=100)),
        "Water": 1 if memory.get_water_today() else 0,
        "Birthdays": len(chapter_items(memory, "birthdays")),
        "Trips": len(chapter_items(memory, "trips")),
        "Gym": len(memory.get_gym_plans(limit=100)),
        "Food": len(memory.get_food_programs(limit=100)),
        "Plans": len(memory.get_goals(status="", limit=100)) + len(memory.get_open_loops(status="", limit=100)),
        "Dating": len(chapter_items(memory, "dating")),
        "Family": len(chapter_items(memory, "family")),
    }
    cards = "".join(
        f'<div class="june-stat-card">'
        f'<div class="june-stat-label">{html.escape(label)}</div>'
        f'<div class="june-stat-value">{count}</div>'
        f'</div>'
        for label, count in chapter_counts.items()
    )
    recent_saves = [
        line for line in activity_log[-30:]
        if "save_" in line or "Saved " in line or "saved " in line
    ][-5:]
    save_html = (
        render_list([("Capture", line) for line in recent_saves])
        if recent_saves
        else '<div class="june-item-meta">No recent captures this session.</div>'
    )
    return f'<div class="june-stat-grid">{cards}</div><div style="margin-top:0.6rem;"></div>' + save_html
