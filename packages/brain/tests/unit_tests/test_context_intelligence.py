"""Tests for derived context intelligence summaries.

The tool-wiring half is gone: `get_active_commitments_summary` and
`get_personal_context` read goals, open loops and gym tables whose writers were
deleted in D.5a, so they advertised knowledge June could no longer acquire. The
module itself is still exercised here until D.5b removes it.
"""

from datetime import date, timedelta
from unittest.mock import patch

import pytest
from june_brain.context_intelligence import (
    build_active_commitments_summary,
    format_active_commitments_summary,
)
from june_brain.memory import Memory


@pytest.fixture
def memory_dir(tmp_path):
    """Patch the memory directory for each test."""
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        yield


@pytest.fixture
def mem(memory_dir):
    """Memory instance backed by a temporary directory."""
    return Memory("test_user")


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
