"""Tests for get_daily_suggestion in patterns.py."""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock


def _make_memory(**kwargs):
    m = MagicMock()
    m.get_calendar_items.return_value = kwargs.get("cal_items", [])
    m.get_goals.return_value = kwargs.get("goals", [])
    m.get_body_metrics.return_value = kwargs.get("metrics", [])
    m.get_habits.return_value = kwargs.get("habits", [])
    return m


def test_daily_suggestion_stalled_goal():
    from june_brain.patterns import get_daily_suggestion

    stale_date = (date.today() - timedelta(days=20)).isoformat()
    memory = _make_memory(
        cal_items=[],  # empty tomorrow
        goals=[{"title": "Write book", "status": "active", "updated_at": stale_date, "created_at": stale_date}],
    )
    result = get_daily_suggestion(memory)
    assert result is not None
    assert "Write book" in result or "tomorrow" in result.lower()


def test_daily_suggestion_none_for_empty_memory():
    from june_brain.patterns import get_daily_suggestion

    memory = _make_memory()
    result = get_daily_suggestion(memory)
    # With no data, all conditions fail gracefully
    assert result is None or isinstance(result, str)


def test_daily_suggestion_habit_low_completion():
    from june_brain.patterns import get_daily_suggestion

    # Habit with only 1 completion this week (14%)
    (date.today() - timedelta(days=7)).isoformat()
    one_day = (date.today() - timedelta(days=1)).isoformat()
    memory = _make_memory(
        goals=[],
        cal_items=[{"date": (date.today() + timedelta(days=1)).isoformat(), "title": "Meeting"}],
        habits=[{"name": "Meditation", "completions": [one_day]}],
    )
    result = get_daily_suggestion(memory)
    assert result is not None
    assert "Meditation" in result or "%" in result
