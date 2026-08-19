"""Derived context summaries for June's backend intelligence layer."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from .memory import Memory

JsonDict = dict[str, Any]

_COMPLETED_STATUSES = {"completed", "done", "canceled", "cancelled", "archived"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_text(value: Any) -> str:
    return str(value).strip()


def _parse_date(value: Any) -> date | None:
    text = _as_text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _active_metric_value(entry: Mapping[str, Any], key: str) -> float | None:
    value = _as_float(entry.get(key), 0.0)
    return value if value > 0 else None


def _recent_average(entries: list[Mapping[str, Any]], key: str) -> float | None:
    values = []
    for entry in entries:
        value = _active_metric_value(entry, key)
        if value is not None:
            values.append(value)
    return _average(values)


def _format_metric(value: float | None, suffix: str = "", precision: int = 1) -> str:
    if value is None:
        return "n/a"
    if float(value).is_integer():
        return f"{int(value)}{suffix}"
    return f"{value:.{precision}f}{suffix}"


def build_active_commitments_summary(memory: Memory) -> JsonDict:
    """Build a structured summary of active commitments across chapters."""
    today = date.today()
    calendar_items = memory.get_calendar_items(limit=50)
    goals = memory.get_goals(status="active", limit=20)
    open_loops = memory.get_open_loops(status="open", limit=20)
    habits = memory.get_habits()

    upcoming_calendar: list[JsonDict] = []
    active_goal_items: list[JsonDict] = []
    open_loop_items: list[JsonDict] = []
    pending_habit_items: list[JsonDict] = []

    for item in calendar_items:
        parsed_date = _parse_date(item.get("date"))
        if parsed_date is None:
            continue
        if _as_text(item.get("status")).lower() in _COMPLETED_STATUSES:
            continue
        days_until = (parsed_date - today).days
        if days_until <= 14:
            upcoming_calendar.append(
                {
                    "title": item.get("title", ""),
                    "date": item.get("date", ""),
                    "time": item.get("time", ""),
                    "details": item.get("details", ""),
                    "days_until": days_until,
                    "status": item.get("status", ""),
                }
            )

    for goal in goals:
        parsed_date = _parse_date(goal.get("target_date"))
        active_goal_items.append(
            {
                "title": goal.get("title", ""),
                "category": goal.get("category", ""),
                "status": goal.get("status", ""),
                "target_date": goal.get("target_date", ""),
                "next_step": goal.get("next_step", ""),
                "days_until": (parsed_date - today).days if parsed_date is not None else None,
            }
        )

    for loop in open_loops:
        parsed_date = _parse_date(loop.get("due_date"))
        if parsed_date is not None and (parsed_date - today).days > 14:
            continue
        open_loop_items.append(
            {
                "topic": loop.get("topic", ""),
                "status": loop.get("status", ""),
                "due_date": loop.get("due_date", ""),
                "next_step": loop.get("next_step", ""),
                "days_until": (parsed_date - today).days if parsed_date is not None else None,
            }
        )

    for habit in habits:
        if habit.get("done_today"):
            continue
        pending_habit_items.append(
            {
                "name": habit.get("name", ""),
                "category": habit.get("category", ""),
                "target_days": habit.get("target_days", ""),
                "streak": habit.get("streak", 0),
            }
        )

    load_score = min(
        100,
        len(upcoming_calendar) * 4
        + len(active_goal_items) * 3
        + len(open_loop_items) * 3
        + len(pending_habit_items) * 2,
    )
    if load_score <= 8:
        load_label = "light"
    elif load_score <= 18:
        load_label = "moderate"
    elif load_score <= 32:
        load_label = "busy"
    else:
        load_label = "loaded"

    next_actions: list[str] = []
    if upcoming_calendar:
        next_actions.extend(
            f"{item['title']} on {item['date']}"
            + (f" at {item['time']}" if item.get("time") else "")
            for item in upcoming_calendar[:3]
        )
    for goal in active_goal_items:
        if goal.get("next_step"):
            goal_text = f"{goal['title']}: {goal['next_step']}"
            if goal.get("target_date"):
                goal_text += f" (target {goal['target_date']})"
            next_actions.append(goal_text)
    for loop in open_loop_items:
        loop_text = f"{loop['topic']}"
        if loop.get("next_step"):
            loop_text += f": {loop['next_step']}"
        if loop.get("due_date"):
            loop_text += f" (due {loop['due_date']})"
        next_actions.append(loop_text)
    for habit in pending_habit_items[:3]:
        next_actions.append(f"{habit['name']} today")

    signals: list[str] = []
    if upcoming_calendar:
        signals.append(f"{len(upcoming_calendar)} calendar item(s) due within 14 days")
    if active_goal_items:
        signals.append(f"{len(active_goal_items)} active goal(s)")
    if open_loop_items:
        signals.append(f"{len(open_loop_items)} open loop(s)")
    if pending_habit_items:
        signals.append(f"{len(pending_habit_items)} habit(s) pending today")
    if not signals:
        signals.append("No active commitments need attention right now.")

    return {
        "date": today.isoformat(),
        "load_score": load_score,
        "load_label": load_label,
        "counts": {
            "calendar_due_soon": len(upcoming_calendar),
            "active_goals": len(active_goal_items),
            "open_loops": len(open_loop_items),
            "pending_habits": len(pending_habit_items),
        },
        "calendar_due_soon": upcoming_calendar[:5],
        "active_goals": active_goal_items[:5],
        "open_loops": open_loop_items[:5],
        "pending_habits": pending_habit_items[:5],
        "next_actions": next_actions[:5],
        "signals": signals,
    }


def format_active_commitments_summary(summary: Mapping[str, Any]) -> str:
    """Render the active commitments snapshot as a concise string."""
    lines = [
        f"Active commitments for {summary.get('date', '')}:",
        f"- Load: {summary.get('load_label', 'unknown')} ({summary.get('load_score', 0)}/100)",
    ]

    counts = summary.get("counts", {})
    lines.append(
        "- Counts: "
        f"{counts.get('calendar_due_soon', 0)} calendar due soon, "
        f"{counts.get('active_goals', 0)} active goals, "
        f"{counts.get('open_loops', 0)} open loops, "
        f"{counts.get('pending_habits', 0)} habits pending"
    )

    calendar_items = summary.get("calendar_due_soon", [])
    if calendar_items:
        lines.append("- Calendar:")
        for item in calendar_items[:3]:
            line = f"  - {item.get('title', '')} on {item.get('date', '')}"
            if item.get("time"):
                line += f" at {item['time']}"
            if item.get("days_until") is not None:
                line += f" ({item['days_until']}d)"
            lines.append(line)

    goals = summary.get("active_goals", [])
    if goals:
        lines.append("- Goals:")
        for goal in goals[:3]:
            line = f"  - {goal.get('title', '')}"
            if goal.get("next_step"):
                line += f" | Next: {goal['next_step']}"
            if goal.get("target_date"):
                line += f" | Target: {goal['target_date']}"
            lines.append(line)

    open_loops = summary.get("open_loops", [])
    if open_loops:
        lines.append("- Open loops:")
        for loop in open_loops[:3]:
            line = f"  - {loop.get('topic', '')}"
            if loop.get("next_step"):
                line += f" | Next: {loop['next_step']}"
            if loop.get("due_date"):
                line += f" | Due: {loop['due_date']}"
            lines.append(line)

    pending_habits = summary.get("pending_habits", [])
    if pending_habits:
        lines.append("- Habits pending today:")
        for habit in pending_habits[:3]:
            line = f"  - {habit.get('name', '')}"
            if habit.get("category"):
                line += f" ({habit['category']})"
            if habit.get("streak"):
                line += f" | Streak {habit['streak']}"
            lines.append(line)

    next_actions = summary.get("next_actions", [])
    if next_actions:
        lines.append("- Next actions:")
        for action in next_actions[:5]:
            lines.append(f"  - {action}")

    signals = summary.get("signals", [])
    if signals:
        lines.append("- Signals: " + "; ".join(str(signal) for signal in signals))

    return "\n".join(lines)
