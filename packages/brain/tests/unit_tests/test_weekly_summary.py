"""Tests for generate_weekly_summary tool."""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch


def _make_memory(
    sessions=None, habits=None, goals=None, metrics=None,
    cal_items=None, moods=None,
):
    m = MagicMock()
    m.get_workout_sessions.return_value = sessions or []
    m.get_habits.return_value = habits or []
    m.get_goals.return_value = goals or []
    m.get_body_metrics.return_value = metrics or []
    m.get_calendar_items.return_value = cal_items or []
    m.get_mood_history.return_value = moods or []
    m.save_journal.return_value = {}
    return m


def _run_summary(memory):
    """Run generate_weekly_summary with an injected mock memory."""
    from june_brain.tools import generate_weekly_summary

    with patch("june_brain.tools._memory_for_state", return_value=memory):
        return generate_weekly_summary.invoke({"state": {"messages": [], "user_id": "test"}})


def test_weekly_summary_returns_nonempty_for_populated_memory():
    today = date.today()
    yesterday = (today - timedelta(days=1)).isoformat()

    memory = _make_memory(
        sessions=[{"date": yesterday, "plan_name": "Push", "energy_rating": 4}],
        habits=[{"name": "Water", "completions": [(today - timedelta(days=i)).isoformat() for i in range(7)]}],
        goals=[{"title": "Run 5k", "status": "active", "updated_at": yesterday}],
        metrics=[{"date": yesterday, "sleep_hours": 7.5, "energy": 3, "stress": 2}],
        moods=[{"mood": "good", "date": yesterday}],
    )
    result = _run_summary(memory)
    assert isinstance(result, str)
    assert len(result) > 50
    assert "Weekly Review" in result


def test_weekly_summary_saves_journal_entry():
    memory = _make_memory()
    _run_summary(memory)
    memory.save_journal.assert_called_once()
    call_args = memory.save_journal.call_args[0][0]
    assert "Weekly Review" in call_args


def test_weekly_summary_graceful_for_empty_memory():
    memory = _make_memory()
    result = _run_summary(memory)
    assert isinstance(result, str)
    assert "Weekly Review" in result
    assert "No sessions" in result or "No habits" in result or "No active goals" in result
