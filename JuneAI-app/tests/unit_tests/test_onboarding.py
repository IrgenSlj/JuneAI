from unittest.mock import patch

from agent.memory import Memory
from agent_ui.onboarding import (
    chapter_onboarding_plan,
    first_run_profile_summary,
    first_run_setup_summary,
    workspace_onboarding_plan,
)


def test_workspace_onboarding_plan_reflects_current_ui_state() -> None:
    plan = workspace_onboarding_plan(
        {
            "layout": "split",
            "selected_chapter": "plans",
            "checklist_items": ["Add next step", "Capture follow-up"],
            "notice": "",
        }
    )

    assert plan.title == "Nothing pinned yet."
    assert "pin the active focus" in plan.body
    assert plan.note == "2 pinned action(s) already captured."
    assert "Capture a note" in plan.actions
    assert "Set a next step" in plan.actions


def test_chapter_onboarding_plan_is_actionable(tmp_path) -> None:
    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        memory = Memory("onboarding_chapter_user")

    plan = chapter_onboarding_plan(memory, "body")

    assert plan.title == "Nothing saved yet."
    assert "Log sleep" in plan.body
    assert "Log body" in plan.actions
    assert "Body" in plan.eyebrow
    assert "body" in plan.note.lower()


def test_first_run_summaries_distinguish_empty_and_populated_memory(tmp_path) -> None:
    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        memory = Memory("first_run_user")
        empty_profile = first_run_profile_summary(memory)
        empty_setup = first_run_setup_summary(memory)

        memory.save_goal("Ship onboarding helpers", next_step="Wire the empty states")
        memory.save_calendar_item("Launch review", "2026-04-02", details="Review onboarding copy")
        memory.log_body_metrics(
            weight_kg=73.1,
            sleep_hours=7.2,
            sleep_quality=4,
            energy=4,
            stress=2,
            soreness=1,
            resting_hr=55,
            steps=9200,
            notes="Solid baseline day.",
        )

        populated_profile = first_run_profile_summary(memory)
        populated_setup = first_run_setup_summary(memory)

    assert empty_profile.has_data is False
    assert "starting fresh" in empty_profile.headline.lower()
    assert empty_profile.activation_level == "quiet"
    assert empty_profile.chrome_hint == "minimal"
    assert empty_profile.recommended_next_action == "Add one goal"
    assert empty_profile.stages[0].title == "Seed"
    assert "one concrete save" in empty_profile.staged_steps[0].lower()
    assert "Add one goal" in empty_setup.next_actions
    assert "starter saves" in empty_setup.headline.lower()

    assert populated_profile.has_data is True
    assert "already knows" in populated_profile.headline.lower()
    assert populated_profile.activation_level == "growing"
    assert populated_profile.primary_focus == "Calendar"
    assert populated_profile.chrome_hint in {"minimal", "calm"}
    assert populated_profile.recommended_next_action.startswith("Fill ")
    assert populated_profile.stages[0].title == "Confirm"
    assert "Calendar" in populated_profile.active_surfaces
    assert "Body Metrics" in populated_profile.active_surfaces
    assert "Calendar" not in populated_profile.missing_surfaces
    assert "personalize follow-up" in populated_setup.headline.lower()
