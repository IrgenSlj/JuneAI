"""Reusable HTML rendering helpers for the JuneAI Streamlit app."""

from __future__ import annotations

import html
from typing import Any

from agent.memory import Memory
from agent.tools import DEFAULT_UI_STATE

from .chapters import chapter_items, chapter_subtitle
from .onboarding import (
    chapter_onboarding_plan,
    notification_onboarding_plan,
    workspace_onboarding_plan,
)
from .shell import (
    empty_state_html,
    list_html,
    onboarding_card_html,
    stat_card_html,
    stat_grid_html,
)
from .transcript import extract_text, render_save_summary_html as _render_save_summary_html, render_transcript_html

__all__ = [
    "default_ui_state",
    "energy_dots_html",
    "extract_text",
    "habit_ring_svg",
    "render_activity",
    "render_capture_health",
    "render_list",
    "render_memory_focus",
    "render_notifications",
    "render_save_summary_html",
    "render_trust_panel",
    "render_workspace",
    "transcript_html",
    "water_dots_html",
]


def default_ui_state() -> dict[str, Any]:
    return {
        "layout": DEFAULT_UI_STATE["layout"],
        "show_right_panel": DEFAULT_UI_STATE["show_right_panel"],
        "selected_chapter": DEFAULT_UI_STATE["selected_chapter"],
        "focus_title": DEFAULT_UI_STATE["focus_title"],
        "focus_body": DEFAULT_UI_STATE["focus_body"],
        "checklist_title": DEFAULT_UI_STATE["checklist_title"],
        "checklist_items": list(DEFAULT_UI_STATE["checklist_items"]),
        "notice": DEFAULT_UI_STATE["notice"],
    }


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


def transcript_html(messages: list[Any], live_response: str = "", extra_messages: list[Any] | None = None) -> str:
    return render_transcript_html(messages, live_response=live_response, extra_messages=extra_messages)


def render_save_summary_html(
    title: object,
    items: list[tuple[object, object]],
    *,
    note: object = "",
    collapse_threshold: int = 900,
) -> str:
    """Render a compact inline summary of what June saved from a turn."""
    return _render_save_summary_html(
        title,
        items,
        note=note,
        collapse_threshold=collapse_threshold,
    )


def render_list(items: list[tuple[str, str]]) -> str:
    return list_html(items)


def render_workspace(ui_state: dict[str, Any], include_header: bool = True) -> str:
    title = ui_state.get("focus_title") or "Workspace"
    body = ui_state.get("focus_body") or ""
    checklist_items = [str(item) for item in ui_state.get("checklist_items", []) if str(item)]
    checklist_title = ui_state.get("checklist_title") or "Next steps"
    notice = ui_state.get("notice") or ""
    header = (
        f'<div class="june-label">Workspace</div>'
        f'<h3 class="june-title">{html.escape(title)}</h3>'
        if include_header
        else f'<h3 class="june-title">{html.escape(title)}</h3>'
    )
    if not body and not checklist_items and not notice:
        plan = workspace_onboarding_plan(ui_state)
        return header + onboarding_card_html(
            **plan.as_kwargs(),
        )

    parts = [header]
    if body:
        parts.append(f'<div class="june-item-meta">{html.escape(body)}</div>')
    parts.append(f'<div class="june-label" style="margin-top:0.6rem;">{html.escape(checklist_title)}</div>')
    parts.append(
        '<div class="june-item-meta"><ul>'
        + "".join(f"<li>{html.escape(item)}</li>" for item in checklist_items)
        + ("</ul></div>" if checklist_items else "<li>No pinned actions.</li></ul></div>")
    )
    if notice:
        parts.append(f'<div class="june-item-meta" style="margin-top:0.5rem;">{html.escape(notice)}</div>')
    return "".join(parts)


def render_memory_focus(memory: Memory, chapter_key: str) -> str:
    items = chapter_items(memory, chapter_key)
    subtitle = chapter_subtitle(chapter_key)
    if not items:
        plan = chapter_onboarding_plan(memory, chapter_key)
        return (
            f'<div class="june-subtitle">{html.escape(subtitle)}</div>'
            + empty_state_html(**plan.as_kwargs())
        )
    return f'<div class="june-subtitle">{html.escape(subtitle)}</div>{list_html(items)}'


def render_activity(activity_log: list[str]) -> str:
    if not activity_log:
        return empty_state_html(
            title="Assistant activity",
            body="June's recent actions, tool calls, and routing choices will appear here.",
            tips=("This stays secondary to the main conversation.",),
            eyebrow="Activity",
        )
    return list_html([("", line) for line in activity_log[-10:]])


def _section_html(title: str, items: list[tuple[str, str]], empty_title: str, empty_body: str) -> str:
    if not items:
        return empty_state_html(
            title=empty_title,
            body=empty_body,
            tips=("The surface will fill as June saves more context.",),
            eyebrow=title,
        )
    return f'<div class="june-label" style="margin-top:0.6rem;">{html.escape(title)}</div>' + render_list(items)


def render_trust_panel(memory: Memory, activity_log: list[str]) -> str:
    """Render a user-facing summary of what June saved and recent assistant actions."""
    from .panels import build_trust_panel_model

    model = build_trust_panel_model(memory, activity_log)
    saved_items = [(item.title, item.copy) for item in model.what_june_saved]
    action_items = [(item.title, item.copy) for item in model.recent_assistant_actions]
    health_count_items = [
        (label, str(count))
        for label, count in model.capture_health_counts.items()
        if count > 0
    ][:6]
    parts = [
        '<div class="june-label">Trust</div>',
        f'<h3 class="june-title">{html.escape(model.title)}</h3>',
        f'<div class="june-item-meta">{html.escape(model.caption)}</div>',
        _section_html("What June saved", saved_items, "Nothing saved yet.", "June will show the latest saved items here."),
        _section_html("Recent assistant actions", action_items, "No assistant actions yet.", "June's recent routes, tool calls, and responses will appear here."),
        _section_html("Light health check", health_count_items, "No captures yet.", "The health check becomes useful once June has a few saved items."),
    ]
    return "".join(parts)


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
    if not items:
        return empty_state_html(**notification_onboarding_plan(memory).as_kwargs())
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
    recent_saves = [
        line for line in activity_log[-30:]
        if "save_" in line or "Saved " in line or "saved " in line
    ][-5:]
    save_html = (
        render_list([("", line) for line in recent_saves])
        if recent_saves
        else '<div class="june-item-meta">No recent captures this session.</div>'
    )
    return stat_grid_html([stat_card_html(label, count) for label, count in chapter_counts.items()]) + '<div style="margin-top:0.6rem;"></div>' + save_html
