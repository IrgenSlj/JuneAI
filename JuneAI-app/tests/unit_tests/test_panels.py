from datetime import date, timedelta
from unittest.mock import patch

from agent.memory import Memory
from agent_ui.panels import (
    build_debug_panel_model,
    build_memory_panel_model,
    build_setup_progress_model,
    build_today_panel_model,
    build_trust_panel_model,
    build_workspace_panel_model,
)


def _make_memory(tmp_path, user_id: str = "panel_user") -> Memory:
    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        return Memory(user_id)


def test_build_setup_progress_model_tracks_core_surfaces(tmp_path) -> None:
    memory = _make_memory(tmp_path, "setup_progress_user")

    empty_model = build_setup_progress_model(memory)

    memory.save_goal("Ship onboarding helpers", next_step="Wire the new panels")
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
    memory.create_or_update_habit("Walk")
    memory.save_relationship_profile("Ava", "family", "Sister")
    memory.save_calendar_item("Anna birthday", "2026-04-14", details="Birthday reminder")

    populated_model = build_setup_progress_model(memory)

    assert empty_model.is_complete is False
    assert empty_model.ready_count == 0
    assert "Calendar" in empty_model.missing_titles
    assert empty_model.setup_steps
    assert empty_model.summary.activation_level == "quiet"
    assert empty_model.recommended_next_action == "Add one goal"
    assert empty_model.stages[0].title == "Seed"
    assert empty_model.chrome.mode == "minimal"
    assert empty_model.chrome.primary_surface == "Today"

    assert populated_model.is_complete is True
    assert populated_model.has_data is True
    assert populated_model.ready_count == populated_model.total_count == 6
    assert populated_model.missing_titles == ()
    assert populated_model.next_actions
    assert populated_model.summary.activation_level == "established"
    assert populated_model.recommended_next_action in populated_model.next_actions
    assert populated_model.stages[0].title == "Confirm"
    assert populated_model.chrome.density == "calm"


def test_build_today_panel_model_exposes_actionable_summary(tmp_path) -> None:
    memory = _make_memory(tmp_path, "today_panel_user")
    memory.save_goal("Ship onboarding helpers", next_step="Wire the new panels")
    memory.save_open_loop("Confirm UI copy", next_step="Check the empty states", due_date=(date.today() + timedelta(days=3)).isoformat())
    memory.save_calendar_item("Launch review", date.today().isoformat(), details="Review onboarding copy")
    memory.create_or_update_habit("Walk")
    memory.log_habit_completion("Walk")
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
    memory.log_water(6)
    memory.log_workout_session("Push day", duration_min=45, energy_rating=4)
    memory.log_nutrition("breakfast", "oats and berries", calories_est=420, protein_est=24)

    model = build_today_panel_model(memory, {"calendar_count": 1})

    assert model.title == "Daily operating view"
    assert model.headline == model.today_summary["headline"]
    assert model.subheadline.startswith("Established mode · Next:")
    assert [metric.label for metric in model.kpis] == ["Agenda", "Load", "Habits", "Water"]
    assert model.today_summary["headline"] != "No tracked activity yet."
    assert model.today_summary["by_key"]["body"]["value"] != "No body log yet"
    assert model.readiness_summary["readiness_label"] in {"ready", "steady", "recovering", "low"}
    assert any(item.title == "Launch review" for item in model.next_up)
    assert any(line.title == "Ship onboarding helpers" for line in model.priority_stack)
    assert model.setup.ready_count >= 1
    assert model.setup.recommended_next_action
    assert model.setup.stages
    assert model.chrome.primary_surface == "Today"
    assert [section.key for section in model.sections] == ["today", "readiness", "setup", "priority"]
    assert model.sections[0].density == "calm"
    assert model.sections[2].note == model.setup.headline
    assert model.sections[2].items[0].title in {"Calendar", "Gym Schedule", "Food Schedule", "Trips", "Plans", "Habits", "Body Metrics", "Workout Sessions", "Nutrition", "Water", "Dating/Love", "Family", "Birthdays"}


def test_build_memory_panel_model_returns_chapter_metadata(tmp_path) -> None:
    memory = _make_memory(tmp_path, "memory_panel_user")
    memory.save_goal("Ship onboarding helpers", next_step="Wire the new panels")
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

    model = build_memory_panel_model(memory, "plans")

    assert model.selected_label == "Plans"
    assert model.kicker_copy.startswith("Open: Plans")
    assert len(model.chapter_cards) >= 6
    assert model.selected_card is not None
    assert model.selected_card["key"] == "plans"
    assert any(card["key"] == "body" for card in model.chapter_cards)
    assert model.chrome.primary_surface == "Memory"


def test_build_workspace_panel_model_handles_empty_and_populated_states() -> None:
    empty_model = build_workspace_panel_model(
        {
            "focus_title": "Workspace",
            "focus_body": "",
            "checklist_title": "Next steps",
            "checklist_items": [],
            "notice": "",
        }
    )
    populated_model = build_workspace_panel_model(
        {
            "focus_title": "Workspace",
            "focus_body": "Current focus: ship the rail split.",
            "checklist_title": "Next steps",
            "checklist_items": ["Ship panels", "Add tests"],
            "notice": "Pinned from the conversation.",
        }
    )

    assert empty_model.is_empty is True
    assert empty_model.onboarding is not None
    assert empty_model.onboarding.title == "Nothing pinned yet."
    assert empty_model.chrome.mode == "minimal"
    assert populated_model.is_empty is False
    assert populated_model.onboarding is None
    assert populated_model.checklist_items == ("Ship panels", "Add tests")
    assert "ship the rail split" in populated_model.focus_body
    assert populated_model.chrome.mode == "calm"


def test_build_debug_panel_model_surfaces_recent_events_and_activity(tmp_path) -> None:
    memory = _make_memory(tmp_path, "debug_panel_user")
    memory.save_goal("Ship onboarding helpers", next_step="Wire the new panels")
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
    memory.log_water(1)
    memory.record_save_event("goal", "Ship onboarding helpers")

    model = build_debug_panel_model(
        memory,
        [
            "Saved goal | Ship onboarding helpers",
            "body | detailed check-in saved",
            "layout | split",
        ],
    )
    trust_model = build_trust_panel_model(
        memory,
        [
            "Saved goal | Ship onboarding helpers",
            "body | detailed check-in saved",
            "layout | split",
        ],
    )

    assert model.title == "Saved context"
    assert model.caption.startswith("Review what June stored")
    assert trust_model.what_june_saved
    assert trust_model.recent_assistant_actions
    assert len(model.recent_events) >= 1
    assert model.recent_activity == (
        "Saved goal | Ship onboarding helpers",
        "body | detailed check-in saved",
        "layout | split",
    )
    assert model.recent_saves == ("Saved goal | Ship onboarding helpers", "body | detailed check-in saved")
    assert model.capture_health_counts["Plans"] >= 1
    assert model.capture_health_counts["Body"] >= 1
    assert model.capture_health_counts["Water"] == 1
    assert model.chrome.primary_surface == "Trust"
    assert model.chrome.surface_budget == 1
