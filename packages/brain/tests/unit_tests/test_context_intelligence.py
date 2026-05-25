"""Tests for derived context intelligence summaries."""

from datetime import date, timedelta
from unittest.mock import patch

import pytest
from june_brain.context_intelligence import (
    build_active_commitments_summary,
    build_recovery_readiness_summary,
    format_active_commitments_summary,
    format_recovery_readiness_summary,
)
from june_brain.memory import Memory
from june_brain.skills import build_system_prompt
from june_brain.tools import (
    JUNE_TOOLS,
    get_active_commitments_summary,
    get_recovery_readiness_summary,
)


@pytest.fixture
def memory_dir(tmp_path):
    """Patch the memory directory for each test."""
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        yield


@pytest.fixture
def mem(memory_dir):
    """Memory instance backed by a temporary directory."""
    return Memory("test_user")


@pytest.fixture
def tool_state():
    return {
        "messages": [],
        "user_id": "test_user",
        "skill": "assistant",
        "ui_state": {
            "layout": "split",
            "selected_chapter": "",
            "focus_title": "Workspace",
            "focus_body": "",
            "checklist_title": "Next steps",
            "checklist_items": [],
            "notice": "",
        },
    }


def test_recovery_summary_uses_recent_body_and_daily_signals(mem):
    mem.log_body_metrics(
        weight_kg=82.4,
        sleep_hours=7.5,
        sleep_quality=4,
        energy=4,
        stress=1,
        soreness=1,
        resting_hr=56,
        steps=9876,
        notes="Recovered well.",
    )
    mem.log_workout_session(
        plan_name="Upper",
        exercises="Bench, row, press",
        duration_min=45,
        energy_rating=4,
    )
    mem.log_nutrition("breakfast", "eggs and oats", calories_est=450, protein_est=30)
    mem.log_nutrition("dinner", "chicken and rice", calories_est=700, protein_est=50)
    mem.log_water(6)
    mem.create_or_update_habit("Walk", target_days="daily")
    mem.log_habit_completion("Walk")
    mem.create_or_update_habit("Read", target_days="daily")

    summary = build_recovery_readiness_summary(mem)
    text = format_recovery_readiness_summary(summary)

    assert summary["readiness_label"] == "ready"
    assert summary["readiness_score"] >= 75
    assert summary["body_source"] == "today"
    assert summary["water_glasses"] == 6
    assert summary["habits_done"] == 1
    assert summary["habits_pending"] == ["Read"]
    assert "Recovery readiness" in text
    assert "sleep 7.5h" in text
    assert "Water: 6 glasses" in text
    assert "Habits: 1/2 done" in text


def test_commitments_summary_unifies_calendar_goals_loops_and_habits(mem):
    today = date.today()
    mem.save_calendar_item("Dentist", (today + timedelta(days=2)).isoformat(), details="Checkup")
    mem.save_goal(
        "Write proposal",
        target_date=(today + timedelta(days=3)).isoformat(),
        next_step="Draft outline",
    )
    mem.save_open_loop(
        "Book train",
        due_date=(today + timedelta(days=1)).isoformat(),
        next_step="Choose morning train",
    )
    mem.create_or_update_habit("Meditate", target_days="daily")
    mem.create_or_update_habit("Walk", target_days="daily")
    mem.log_habit_completion("Walk")

    summary = build_active_commitments_summary(mem)
    text = format_active_commitments_summary(summary)

    assert summary["load_label"] == "moderate"
    assert summary["counts"] == {
        "calendar_due_soon": 1,
        "active_goals": 1,
        "open_loops": 1,
        "pending_habits": 1,
    }
    assert any("Dentist" in action for action in summary["next_actions"])
    assert any("Write proposal" in action for action in summary["next_actions"])
    assert any("Book train" in action for action in summary["next_actions"])
    assert any("Meditate" in action for action in summary["next_actions"])
    assert "Active commitments" in text
    assert "Calendar:" in text
    assert "Goals:" in text
    assert "Open loops:" in text
    assert "Habits pending today:" in text


def test_tools_and_prompt_reference_new_summaries(mem, tool_state):
    mem.log_body_metrics(sleep_hours=7.0, energy=4, stress=1, soreness=1, steps=8000)
    mem.save_calendar_item("Standup", (date.today() + timedelta(days=1)).isoformat())
    mem.save_goal("Ship draft", next_step="Finish outline")

    tool_names = {tool.name for tool in JUNE_TOOLS}
    assert "get_recovery_readiness_summary" in tool_names
    assert "get_active_commitments_summary" in tool_names

    recovery_text = get_recovery_readiness_summary.func(state=tool_state)
    commitments_text = get_active_commitments_summary.func(state=tool_state)
    prompt = build_system_prompt("assistant")

    assert "Recovery readiness" in recovery_text
    assert "Active commitments" in commitments_text
    assert "get_recovery_readiness_summary" in prompt
    assert "get_active_commitments_summary" in prompt
