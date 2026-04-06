"""Proactive pattern detection for JuneAI.

Scans memory across chapters and returns a list of observations June can weave
naturally into conversation — without being asked.

June notices:
- Workout gaps (not trained in N days)
- Sustained low energy (below threshold for N consecutive days)
- Stale active goals (no updates in N days)
- Broken habit streaks
- Upcoming birthdays (within 7 days)
- Tomorrow's calendar items
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from .memory import Memory


@dataclass(frozen=True)
class PatternInsight:
    """A single observation June may surface in conversation."""

    category: str  # workout | energy | goal | habit | birthday | calendar
    observation: str  # short natural-language string


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_date(value: Any) -> date | None:
    text = str(value).strip() if value else ""
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _workout_gap(memory: Memory, today: date) -> PatternInsight | None:
    """Detect if the user hasn't logged a workout recently."""
    sessions = memory.get_workout_sessions(limit=1)
    if not sessions:
        # No workouts ever — only surface if habits/gym plan exists (shows intent)
        habits = memory.get_habits()
        gym_plans = memory.get_gym_plans()
        if not habits and not gym_plans:
            return None
        return PatternInsight(
            category="workout",
            observation="No workout sessions have been logged yet.",
        )
    last = sessions[0]
    last_date = _parse_date(last.get("date"))
    if last_date is None:
        return None
    gap = (today - last_date).days
    if gap >= 4:
        label = last.get("plan_name", "workout") or "workout"
        return PatternInsight(
            category="workout",
            observation=f"No workout logged in {gap} days — last was {label} on {last_date.isoformat()}.",
        )
    return None


def _low_energy_streak(memory: Memory, today: date, threshold: int = 3) -> PatternInsight | None:
    """Detect consecutive days where energy was logged below 3/5."""
    metrics = memory.get_body_metrics(days=7)
    if len(metrics) < threshold:
        return None

    # metrics are ordered newest-first from SQLite; check most recent N days
    low_streak = 0
    for entry in metrics[:threshold]:
        energy = _as_int(entry.get("energy"), 0)
        if 0 < energy <= 2:
            low_streak += 1
        else:
            break

    if low_streak >= threshold:
        return PatternInsight(
            category="energy",
            observation=f"Energy has been logged at 2/5 or below for {low_streak} days in a row.",
        )
    return None


def _stale_goals(memory: Memory, today: date, stale_days: int = 14) -> list[PatternInsight]:
    """Detect active goals with no recent next-step updates."""
    goals = memory.get_goals(status="active", limit=10)
    insights: list[PatternInsight] = []
    for goal in goals:
        updated_raw = goal.get("updated_at") or goal.get("created_at", "")
        updated = _parse_date(updated_raw)
        if updated is None:
            continue
        age = (today - updated).days
        if age >= stale_days:
            title = goal.get("title", "a goal")
            insights.append(PatternInsight(
                category="goal",
                observation=f"Goal '{title}' has had no updates in {age} days.",
            ))
    return insights[:2]  # surface at most 2 stale goals


def _broken_habit_streaks(memory: Memory) -> list[PatternInsight]:
    """Detect habits whose streaks were recently broken (streak was positive, now 0)."""
    habits = memory.get_habits()
    insights: list[PatternInsight] = []
    for habit in habits:
        streak = _as_int(habit.get("streak"), 0)
        name = habit.get("name", "")
        done_today = bool(habit.get("done_today"))
        # A habit with zero streak but active is recently broken (or never started)
        # Only flag it if the habit has a history of being active (active=True)
        if not done_today and streak == 0 and habit.get("active", True):
            insights.append(PatternInsight(
                category="habit",
                observation=f"Habit '{name}' streak is at zero — worth picking back up today.",
            ))
    return insights[:2]  # surface at most 2


def _upcoming_birthdays(memory: Memory, today: date, window_days: int = 7) -> list[PatternInsight]:
    """Detect birthdays or anniversaries coming up within the window."""
    items = memory.get_calendar_items(limit=100)
    insights: list[PatternInsight] = []
    for item in items:
        title = (item.get("title") or "").lower()
        if "birthday" not in title and "anniversary" not in title:
            continue
        item_date = _parse_date(item.get("date"))
        if item_date is None:
            continue
        # Compare month-day only — recurring each year
        this_year = item_date.replace(year=today.year)
        if this_year < today:
            this_year = item_date.replace(year=today.year + 1)
        days_until = (this_year - today).days
        if 0 <= days_until <= window_days:
            display_title = item.get("title", "")
            if days_until == 0:
                when = "today"
            elif days_until == 1:
                when = "tomorrow"
            else:
                when = f"in {days_until} days"
            insights.append(PatternInsight(
                category="birthday",
                observation=f"{display_title} is {when} ({this_year.isoformat()}).",
            ))
    return insights


def _tomorrows_calendar(memory: Memory, today: date) -> list[PatternInsight]:
    """Surface calendar items due tomorrow."""
    tomorrow = today + timedelta(days=1)
    items = memory.get_calendar_items(limit=50)
    insights: list[PatternInsight] = []
    for item in items:
        item_date = _parse_date(item.get("date"))
        if item_date != tomorrow:
            continue
        status = (item.get("status") or "").lower()
        if status in {"done", "completed", "cancelled", "canceled"}:
            continue
        title = item.get("title", "event")
        time_str = item.get("time", "")
        when = f"at {time_str}" if time_str else "tomorrow"
        insights.append(PatternInsight(
            category="calendar",
            observation=f"{title} is {when}.",
        ))
    return insights[:3]


def detect_patterns(memory: Memory) -> list[PatternInsight]:
    """Run all pattern detectors and return a deduplicated list of insights.

    Keep the list short — at most 5 insights injected per turn.
    """
    today = date.today()
    insights: list[PatternInsight] = []

    gap = _workout_gap(memory, today)
    if gap:
        insights.append(gap)

    energy = _low_energy_streak(memory, today)
    if energy:
        insights.append(energy)

    insights.extend(_stale_goals(memory, today))
    insights.extend(_broken_habit_streaks(memory))
    insights.extend(_upcoming_birthdays(memory, today))
    insights.extend(_tomorrows_calendar(memory, today))

    return insights[:5]


def format_patterns_for_prompt(insights: list[PatternInsight]) -> str:
    """Format pattern insights as a short block for injection into the system prompt."""
    if not insights:
        return ""
    lines = ["JUNE'S OBSERVATIONS (weave these in naturally — do not recite as a list):"]
    for insight in insights:
        lines.append(f"- {insight.observation}")
    return "\n".join(lines)
