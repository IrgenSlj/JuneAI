from datetime import datetime, timezone

from june_brain.skills import infer_skill_from_text
from june_brain.skills import build_system_prompt


def test_infer_planner_skill() -> None:
    assert infer_skill_from_text("Schedule a dentist appointment for next Friday") == "planner"


def test_infer_wellness_skill() -> None:
    assert infer_skill_from_text("Build me a gym plan and a high protein meal structure") == "wellness"


def test_infer_curator_skill() -> None:
    assert infer_skill_from_text("Recommend a movie like Past Lives and save it") == "curator"


def test_infer_default_skill() -> None:
    assert infer_skill_from_text("Help me think through a difficult decision") == "assistant"


def test_system_prompt_contains_temporal_context() -> None:
    now = datetime(2026, 3, 11, 18, 45, tzinfo=timezone.utc)
    prompt = build_system_prompt("assistant", now=now)
    assert "Local date: 2026-03-11" in prompt
    assert "Day of year: 70" in prompt
    assert "Part of day: evening" in prompt
