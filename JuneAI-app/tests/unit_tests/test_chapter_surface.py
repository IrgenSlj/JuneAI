from unittest.mock import patch

from agent.memory import Memory
from agent_ui.chapter_surface import (
    CHAPTERS,
    chapter_status,
    chapter_status_cards,
    chapter_statuses,
    today_memory_summary,
)


def test_chapter_statuses_include_freshness_attention_preview_and_last_updated(tmp_path) -> None:
    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        memory = Memory("surface_user")
        memory.save_goal("Ship chapter metadata", next_step="Add card summaries")
        memory.save_open_loop("Review status cards", next_step="Check freshness")
        memory.log_body_metrics(
            weight_kg=72.4,
            sleep_hours=7.5,
            sleep_quality=4,
            energy=4,
            stress=2,
            soreness=1,
            resting_hr=54,
            steps=10320,
            notes="Recovered well.",
        )
        memory.log_water(3)

    statuses = {status.key: status for status in chapter_statuses(memory)}
    plans = statuses["plans"]
    body = statuses["body"]
    water = chapter_status(memory, "water")
    cards = chapter_status_cards(memory)
    with patch("agent.memory.MEMORY_DIR", str(tmp_path / "empty")):
        empty_water = Memory("empty_water_user")
        empty_status = chapter_status(empty_water, "water")

    assert [key for key, _label in CHAPTERS] == list(statuses.keys())
    assert len(cards) == len(CHAPTERS)
    assert cards[4]["key"] == "plans"
    assert plans.item_count == 2
    assert plans.freshness == "today"
    assert plans.attention == "watch"
    assert "1 goal" in plans.preview
    assert "1 open loop" in plans.preview
    assert "Next: Add card summaries" in plans.preview
    assert plans.last_updated
    assert body.freshness == "today"
    assert "sleep 7.5h" in body.preview
    assert body.last_updated
    assert water.freshness == "today"
    assert water.attention == "watch"
    assert "3 glasses today" in water.preview
    assert empty_status.freshness == "empty"
    assert empty_status.attention == "needs_attention"
    assert "No entries yet" in empty_status.preview or "No water" in empty_status.preview


def test_today_memory_summary_builds_ui_card_data(tmp_path) -> None:
    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        memory = Memory("summary_user")
        memory.create_or_update_habit("Read", category="focus", target_days="daily")
        memory.log_habit_completion("Read")
        memory.log_body_metrics(
            weight_kg=71.9,
            sleep_hours=8.0,
            sleep_quality=4,
            energy=4,
            stress=1,
            soreness=1,
            resting_hr=52,
            steps=11200,
            notes="Good recovery.",
        )
        memory.log_workout_session(
            plan_name="Push Day",
            exercises="Bench, incline, dips",
            duration_min=55,
            notes="Kept the pace controlled.",
            energy_rating=4,
        )
        memory.log_nutrition("lunch", "Chicken rice bowl", calories_est=640, protein_est=42)
        memory.log_water(5)
        memory.save_calendar_item(
            "Trip to the mountains",
            "2026-04-03",
            details="Weekend getaway",
        )

    summary = today_memory_summary(memory)
    cards = summary["by_key"]

    assert summary["headline"]
    assert cards["body"]["value"].startswith("sleep 8.0h")
    assert cards["habits"]["value"] == "1/1 done"
    assert "Complete: Read" in cards["habits"]["preview"]
    assert cards["workout"]["value"] == "Push Day"
    assert "55 min" in cards["workout"]["preview"]
    assert cards["nutrition"]["value"] == "1 meal"
    assert "~640 kcal" in cards["nutrition"]["preview"]
    assert cards["water"]["value"] == "5 glasses"
    assert "to goal" in cards["water"]["preview"]
    assert cards["reminders"]["value"] == "1 upcoming"
    assert "Trip to the mountains" in cards["reminders"]["preview"]
    assert len(summary["cards"]) == 6
