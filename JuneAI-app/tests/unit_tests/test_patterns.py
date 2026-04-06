"""Tests for Session 4: proactive pattern detection."""
import sqlite3
from datetime import date, timedelta
from unittest.mock import patch

import pytest

from agent.memory import Memory
from agent.patterns import (
    PatternInsight,
    detect_patterns,
    format_patterns_for_prompt,
    _workout_gap,
    _low_energy_streak,
    _stale_goals,
    _upcoming_birthdays,
    _tomorrows_calendar,
)


@pytest.fixture
def mem(tmp_path):
    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        yield Memory(user_id="pat")


def _insert_workout(mem: Memory, days_ago: int, plan_name: str = "Push day") -> None:
    """Insert a workout session with a past date directly into SQLite."""
    d = (date.today() - timedelta(days=days_ago)).isoformat()
    with sqlite3.connect(mem._db_path) as conn:
        conn.execute(
            "INSERT INTO workout_sessions "
            "(user_id,date,plan_name,exercises,duration_min,notes,energy_rating,timestamp) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("pat", d, plan_name, "bench, OHP", 45, "", 4, d + "T10:00:00"),
        )


def _insert_body_metrics(mem: Memory, days_ago: int, energy: int) -> None:
    """Insert body metrics for a past date directly into SQLite."""
    d = (date.today() - timedelta(days=days_ago)).isoformat()
    with sqlite3.connect(mem._db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO body_metrics "
            "(user_id,date,weight_kg,sleep_hours,sleep_quality,energy,stress,soreness,"
            "resting_hr,steps,notes,timestamp) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ("pat", d, 0, 0, 0, energy, 0, 0, 0, 0, "", d + "T08:00:00"),
        )


def test_no_patterns_on_empty_memory(mem):
    insights = detect_patterns(mem)
    assert isinstance(insights, list)
    assert len(insights) <= 5


def test_workout_gap_fires_after_four_days(mem):
    _insert_workout(mem, days_ago=5)
    insight = _workout_gap(mem, date.today())
    assert insight is not None
    assert insight.category == "workout"
    assert "5 days" in insight.observation


def test_workout_gap_does_not_fire_when_recent(mem):
    _insert_workout(mem, days_ago=1)
    insight = _workout_gap(mem, date.today())
    assert insight is None


def test_low_energy_streak_fires_on_three_consecutive_low_days(mem):
    for i in range(3):
        _insert_body_metrics(mem, days_ago=i, energy=2)
    insight = _low_energy_streak(mem, date.today())
    assert insight is not None
    assert insight.category == "energy"


def test_low_energy_streak_does_not_fire_when_energy_ok(mem):
    for i in range(3):
        _insert_body_metrics(mem, days_ago=i, energy=4)
    insight = _low_energy_streak(mem, date.today())
    assert insight is None


def test_stale_goals_fires_for_old_active_goal(mem):
    mem.save_goal(
        title="Launch product",
        category="work",
        target_date="",
        next_step="finish MVP",
        status="active",
    )
    stale_date = (date.today() - timedelta(days=20)).isoformat()
    with sqlite3.connect(mem._db_path) as conn:
        conn.execute(
            "UPDATE goals SET updated_at = ? WHERE user_id = ?",
            (stale_date + "T00:00:00", "pat"),
        )
    insights = _stale_goals(mem, date.today())
    assert len(insights) >= 1
    assert any("Launch product" in i.observation for i in insights)


def test_upcoming_birthdays_fires_within_window(mem):
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    mem.save_calendar_item(
        title="Mom's birthday",
        date=tomorrow,
        time="",
        details="birthday",
        status="upcoming",
    )
    insights = _upcoming_birthdays(mem, date.today(), window_days=7)
    assert len(insights) >= 1
    assert any("Mom's birthday" in i.observation for i in insights)


def test_upcoming_birthdays_does_not_fire_outside_window(mem):
    far_future = (date.today() + timedelta(days=30)).isoformat()
    mem.save_calendar_item(
        title="Brother's birthday",
        date=far_future,
        time="",
        details="birthday",
        status="upcoming",
    )
    insights = _upcoming_birthdays(mem, date.today(), window_days=7)
    assert all("Brother's birthday" not in i.observation for i in insights)


def test_tomorrows_calendar_fires(mem):
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    mem.save_calendar_item(
        title="Dentist appointment",
        date=tomorrow,
        time="10:00",
        details="checkup",
        status="upcoming",
    )
    insights = _tomorrows_calendar(mem, date.today())
    assert len(insights) >= 1
    assert any("Dentist appointment" in i.observation for i in insights)


def test_detect_patterns_returns_at_most_five(mem):
    _insert_workout(mem, days_ago=5)
    for i in range(3):
        _insert_body_metrics(mem, days_ago=i, energy=1)
    for j in range(3):
        d = (date.today() + timedelta(days=j + 1)).isoformat()
        mem.save_calendar_item(
            title=f"Meeting {j}", date=d, time="09:00",
            details="", status="upcoming",
        )
    insights = detect_patterns(mem)
    assert len(insights) <= 5


def test_format_patterns_for_prompt_empty():
    assert format_patterns_for_prompt([]) == ""


def test_format_patterns_for_prompt_non_empty():
    insights = [
        PatternInsight(category="workout", observation="No workout in 5 days."),
        PatternInsight(category="energy", observation="Energy has been low for 3 days."),
    ]
    result = format_patterns_for_prompt(insights)
    assert "JUNE'S OBSERVATIONS" in result
    assert "No workout in 5 days." in result
    assert "Energy has been low for 3 days." in result
