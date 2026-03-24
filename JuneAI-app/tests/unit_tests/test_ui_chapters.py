from unittest.mock import patch

from agent.memory import Memory
from agent_ui.chapters import CHAPTERS, chapter_items
from agent_ui.rendering import render_capture_health


def test_chapter_registry_includes_new_wellness_surfaces() -> None:
    keys = {key for key, _label in CHAPTERS}
    assert {"habits", "body", "workouts", "nutrition", "water"}.issubset(keys)


def test_new_chapter_surfaces_render_from_memory(tmp_path) -> None:
    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        memory = Memory("ui_test_user")
        memory.create_or_update_habit("Read", category="focus", target_days="daily")
        memory.log_habit_completion("Read")
        memory.log_body_metrics(
            weight_kg=72.4,
            sleep_hours=7.5,
            sleep_quality=4,
            energy=4,
            stress=2,
            soreness=1,
            resting_hr=54,
            steps=10320,
            notes="Legs still heavy from yesterday",
        )
        memory.log_workout_session(plan_name="Push Day", exercises="Bench, incline, dips", duration_min=55)
        memory.log_nutrition("lunch", "Chicken rice bowl", calories_est=640, protein_est=42)
        memory.log_water(3)

    assert chapter_items(memory, "habits")
    body_items = chapter_items(memory, "body")
    assert body_items
    assert "resting HR 54" in body_items[0][1]
    assert "steps 10320" in body_items[0][1]
    assert chapter_items(memory, "workouts")
    assert chapter_items(memory, "nutrition")
    assert chapter_items(memory, "water")

    health = render_capture_health(memory, ["Saved habit Read"])
    assert "Workout Sessions" in health
    assert "Nutrition" in health
    assert "Water" in health
