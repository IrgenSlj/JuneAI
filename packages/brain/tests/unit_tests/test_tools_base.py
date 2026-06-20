"""Tests for the dependency-free tool abstraction (tools_base)."""

from __future__ import annotations

from typing import Annotated

from june_brain.tools_base import Inject, Tool, tool


def test_tool_exposes_name_description_and_args() -> None:
    @tool
    def log_mood(mood: str, note: str = "") -> str:
        """Log the user's mood."""
        return f"mood={mood} note={note}"

    assert isinstance(log_mood, Tool)
    assert log_mood.name == "log_mood"
    assert log_mood.description == "Log the user's mood."
    assert set(log_mood.args.keys()) == {"mood", "note"}
    assert log_mood.args["mood"]["required"] is True
    assert log_mood.args["note"]["required"] is False


def test_injected_params_excluded_from_args_and_filled_at_invoke() -> None:
    seen: dict = {}

    @tool
    def save_thing(
        title: str,
        state: Annotated[dict | None, Inject] = None,
    ) -> str:
        """Save a thing."""
        seen["state"] = state
        return "ok"

    # The injected param is hidden from the advertised schema.
    assert "state" not in save_thing.args
    assert save_thing.injected == ("state",)

    # The model only supplies `title`; state defaults to None.
    assert save_thing.invoke({"title": "x"}) == "ok"
    assert seen["state"] is None

    # The dispatcher may inject state alongside the model args.
    save_thing.invoke({"title": "x", "state": {"user_id": "alice"}})
    assert seen["state"] == {"user_id": "alice"}


def test_invoke_ignores_unknown_keys_and_respects_defaults() -> None:
    @tool
    def greet(name: str, greeting: str = "Hi") -> str:
        """Greet someone."""
        return f"{greeting}, {name}"

    # Unknown keys are dropped; missing optional uses the default.
    assert greet.invoke({"name": "Sam", "bogus": 1}) == "Hi, Sam"
    assert greet.invoke({"name": "Sam", "greeting": "Yo"}) == "Yo, Sam"
