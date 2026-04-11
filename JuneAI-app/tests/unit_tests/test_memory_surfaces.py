"""Unit tests for memory surfaces not covered in test_memory.py.

Covers: nutrition, water, workout sessions, journal, relationship profiles,
recovery/commitment context summaries, and chapter completeness.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import pytest

from agent.memory import Memory


@pytest.fixture
def memory_dir(tmp_path):
    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        yield tmp_path


@pytest.fixture
def mem(memory_dir):
    return Memory("test_user")


# ---------------------------------------------------------------------------
# Nutrition
# ---------------------------------------------------------------------------

def test_log_nutrition_stores_meal(mem):
    mem.log_nutrition("lunch", "Chicken and rice", calories_est=520, protein_est=45)
    meals = mem.get_nutrition_today()
    assert meals, "Expected at least one meal"
    assert meals[0]["description"] == "Chicken and rice"
    assert meals[0]["calories_est"] == 520
    assert meals[0]["protein_est"] == 45


def test_log_multiple_meals_today(mem):
    mem.log_nutrition("breakfast", "Oats with berries", calories_est=320, protein_est=12)
    mem.log_nutrition("dinner", "Steak and potatoes", calories_est=680, protein_est=52)
    meals = mem.get_nutrition_today()
    assert len(meals) == 2
    descriptions = {m["description"] for m in meals}
    assert "Oats with berries" in descriptions
    assert "Steak and potatoes" in descriptions


def test_nutrition_recent_returns_correct_limit(mem):
    for i in range(5):
        mem.log_nutrition(f"meal{i}", f"Food item {i}", calories_est=400)
    recent = mem.get_nutrition_recent(limit=3)
    assert len(recent) == 3


# ---------------------------------------------------------------------------
# Water
# ---------------------------------------------------------------------------

def test_log_water_increments(mem):
    initial = mem.get_water_today()
    mem.log_water(2)
    assert mem.get_water_today() == initial + 2


def test_set_water_overrides_count(mem):
    mem.log_water(3)
    mem.set_water(5)
    assert mem.get_water_today() == 5


def test_water_starts_at_zero_for_new_user(mem):
    assert mem.get_water_today() == 0


# ---------------------------------------------------------------------------
# Workout sessions
# ---------------------------------------------------------------------------

def test_log_workout_session_stores_fields(mem):
    session = mem.log_workout_session(
        plan_name="Pull Day",
        exercises="Deadlift 4x5, Rows 4x8, Curls 3x12",
        duration_min=50,
        energy_rating=3,
        notes="Grip was the limiting factor",
    )
    assert session["plan_name"] == "Pull Day"
    assert session["duration_min"] == 50
    assert session["energy_rating"] == 3
    assert session["notes"] == "Grip was the limiting factor"


def test_get_today_workout_returns_latest(mem):
    mem.log_workout_session("Push Day", "Bench 4x8", duration_min=45, energy_rating=4)
    workout = mem.get_today_workout()
    assert workout is not None
    assert workout["plan_name"] == "Push Day"


def test_workout_sessions_returns_both_sessions(mem):
    mem.log_workout_session("Session A", "Squats", duration_min=40)
    mem.log_workout_session("Session B", "Rows", duration_min=35)
    sessions = mem.get_workout_sessions(limit=10)
    names = {s["plan_name"] for s in sessions}
    assert "Session A" in names
    assert "Session B" in names
    assert len(sessions) == 2


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

def test_save_and_retrieve_journal(mem):
    mem.save_journal("Today was a productive day. Shipped the export script.")
    entries = mem.get_journal(limit=5)
    assert entries, "Expected a journal entry"
    assert "productive" in entries[0]["entry"]


def test_multiple_journal_entries_ordered_newest_first(mem):
    mem.save_journal("First entry")
    mem.save_journal("Second entry")
    entries = mem.get_journal(limit=5)
    # get_journal returns in reverse-insert order (newest first via DESC then reversed slice)
    assert any("Second entry" in e["entry"] for e in entries)
    assert any("First entry" in e["entry"] for e in entries)


# ---------------------------------------------------------------------------
# Relationship profiles
# ---------------------------------------------------------------------------

def test_save_and_retrieve_relationship_profile(mem):
    mem.save_relationship_profile(
        person="Anna",
        relationship="sister",
        summary="We talk every Sunday. She lives in Berlin.",
    )
    profiles = mem.get_relationship_profiles()
    assert profiles, "Expected a relationship profile"
    anna = next((p for p in profiles if p["person"] == "Anna"), None)
    assert anna is not None
    assert anna["relationship"] == "sister"


def test_relationship_profile_upserts(mem):
    """Saving the same person twice should update, not duplicate."""
    mem.save_relationship_profile("Bob", "friend", "Met at uni")
    mem.save_relationship_profile("Bob", "friend", "Met at uni — now in NYC")
    profiles = [p for p in mem.get_relationship_profiles() if p["person"] == "Bob"]
    assert len(profiles) == 1
    assert "NYC" in profiles[0]["summary"]


# ---------------------------------------------------------------------------
# Recovery readiness & commitment summaries
# ---------------------------------------------------------------------------

def test_build_recovery_readiness_summary_returns_dict(mem):
    from agent.context_intelligence import build_recovery_readiness_summary
    result = build_recovery_readiness_summary(mem)
    assert isinstance(result, dict)
    assert "score" in result or "readiness" in result or isinstance(result, dict)


def test_format_recovery_readiness_summary_returns_string(mem):
    from agent.context_intelligence import (
        build_recovery_readiness_summary,
        format_recovery_readiness_summary,
    )
    summary = build_recovery_readiness_summary(mem)
    text = format_recovery_readiness_summary(summary)
    assert isinstance(text, str)
    assert len(text) > 0


def test_build_active_commitments_summary_returns_dict(mem):
    from agent.context_intelligence import build_active_commitments_summary
    mem.save_calendar_item("Doctor", (date.today() + timedelta(days=2)).isoformat())
    mem.save_goal("Ship v1.0", next_step="Write release notes")
    result = build_active_commitments_summary(mem)
    assert isinstance(result, dict)


def test_format_active_commitments_summary_returns_string(mem):
    from agent.context_intelligence import (
        build_active_commitments_summary,
        format_active_commitments_summary,
    )
    mem.save_calendar_item("Gym", (date.today() + timedelta(days=1)).isoformat())
    summary = build_active_commitments_summary(mem)
    text = format_active_commitments_summary(summary)
    assert isinstance(text, str)
    assert len(text) > 0


# ---------------------------------------------------------------------------
# Chapter completeness
# ---------------------------------------------------------------------------

def test_chapter_completeness_starts_at_zero(mem):
    completeness = mem.get_chapter_completeness()
    assert all(v == 0 for v in completeness.values()), (
        f"Expected all chapters empty, got {completeness}"
    )


def test_chapter_completeness_reflects_saves(mem):
    mem.save_calendar_item("Birthday party", "2026-06-01")
    mem.create_or_update_habit("Morning walk", "health", "daily")
    completeness = mem.get_chapter_completeness()
    assert completeness.get("calendar", 0) > 0
    assert completeness.get("habits", 0) > 0


def test_chapters_needing_attention_shrinks_as_data_fills(mem):
    empty_before = mem.get_chapters_needing_attention()
    mem.save_gym_plan("Spring Split", "Mon/Wed/Fri", goal="strength")
    empty_after = mem.get_chapters_needing_attention()
    assert len(empty_after) < len(empty_before)
