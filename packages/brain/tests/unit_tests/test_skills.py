"""The assembled system prompt.

The four persona-inference tests that used to live here went with the personas
(D.5a). They asserted that "gym plan and a high protein meal structure" routed
to a Wellness Architect — a v1 persona nothing in production ever selected,
kept green and looking maintained by exactly these tests.
"""

from datetime import datetime, timezone

from june_brain.skills import DEFAULT_SKILL, build_system_prompt


def test_system_prompt_contains_temporal_context() -> None:
    now = datetime(2026, 3, 11, 18, 45, tzinfo=timezone.utc)
    prompt = build_system_prompt("assistant", now=now)
    assert "Local date: 2026-03-11" in prompt
    assert "Day of year: 70" in prompt
    assert "Part of day: evening" in prompt


def test_an_unknown_skill_key_falls_back_to_the_default() -> None:
    """The scheduler's one caller passes "default", which is not a key."""
    assert build_system_prompt("default") == build_system_prompt(DEFAULT_SKILL)


def test_the_prompt_names_the_memory_surface() -> None:
    prompt = build_system_prompt(DEFAULT_SKILL)
    for name in ("remember", "forget", "list_promises", "update_promise"):
        assert name in prompt, f"the prompt never tells the model about {name}"
