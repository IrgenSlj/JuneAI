"""Tests for the tool-call coercion + normalization helpers in graph.py.

Local models (Gemma, smaller Mixtrals) emit tool calls in surprising
shapes. The two helpers exist to absorb that variation before the
ToolNode sees the call. These tests pin the contract so a refactor can't
silently regress a shape we already accept.
"""

from __future__ import annotations

from june_brain.graph import _coerce_tool_calls, _normalize_tool_call


# ---------------------------------------------------------------------- coerce


def test_coerce_handles_openai_style_tool_calls_list() -> None:
    """OpenAI-compatible shape: top-level dict with a `tool_calls` array."""
    payload = {
        "tool_calls": [
            {"name": "log_water", "args": {"glasses": 2}},
            {"name": "log_mood", "args": {"mood": "ok"}},
        ]
    }
    calls = _coerce_tool_calls(payload)
    assert calls == [("log_water", {"glasses": 2}), ("log_mood", {"mood": "ok"})]


def test_coerce_handles_function_block_shape() -> None:
    """Some models nest under `function: {name, arguments}` (GPT-style)."""
    payload = [
        {
            "function": {"name": "log_water", "arguments": {"glasses": 3}},
        }
    ]
    assert _coerce_tool_calls(payload) == [("log_water", {"glasses": 3})]


def test_coerce_parses_stringified_arguments_json() -> None:
    """Local models often emit `arguments` as a JSON-encoded string."""
    payload = {
        "tool_calls": [
            {"name": "log_mood", "arguments": '{"mood": "great"}'},
        ]
    }
    assert _coerce_tool_calls(payload) == [("log_mood", {"mood": "great"})]


def test_coerce_drops_calls_without_a_name() -> None:
    payload = {"tool_calls": [{"args": {"x": 1}}, {"name": "ok", "args": {}}]}
    assert _coerce_tool_calls(payload) == [("ok", {})]


def test_coerce_accepts_alternate_top_level_keys() -> None:
    """`calls` and `tools` are accepted aliases for `tool_calls`."""
    assert _coerce_tool_calls({"calls": [{"name": "x", "args": {}}]}) == [("x", {})]
    assert _coerce_tool_calls({"tools": [{"name": "y", "args": {}}]}) == [("y", {})]


def test_coerce_treats_bare_dict_as_single_call() -> None:
    """If the payload looks like a single call dict, accept it as one."""
    assert _coerce_tool_calls({"name": "log_mood", "args": {"mood": "ok"}}) == [
        ("log_mood", {"mood": "ok"})
    ]


def test_coerce_returns_empty_for_garbage() -> None:
    assert _coerce_tool_calls(None) == []
    assert _coerce_tool_calls(42) == []
    assert _coerce_tool_calls("not a tool call") == []
    # A list of non-dicts yields nothing.
    assert _coerce_tool_calls([1, 2, 3]) == []


def test_coerce_picks_tool_name_when_name_is_missing() -> None:
    """Some Gemma outputs use `tool_name` instead of `name`."""
    payload = {"tool_calls": [{"tool_name": "log_water", "args": {"glasses": 1}}]}
    assert _coerce_tool_calls(payload) == [("log_water", {"glasses": 1})]


# -------------------------------------------------------------------- normalize


def test_normalize_aliases_save_goal_to_track_goal() -> None:
    name, args = _normalize_tool_call("save_goal", {"title": "Run 5k"})
    assert name == "track_goal"
    assert args["title"] == "Run 5k"
    assert args["status"] == "active"  # defaults filled in


def test_normalize_calendar_collects_synonyms_into_canonical_shape() -> None:
    """The model may use `event`/`day`/`note` instead of title/date/details."""
    name, args = _normalize_tool_call(
        "save_reminder",
        {"event": "Dentist", "day": "2026-06-01", "note": "bring xrays", "at": "09:00"},
    )
    assert name == "save_calendar_item"
    assert args == {
        "title": "Dentist",
        "date": "2026-06-01",
        "time": "09:00",
        "details": "bring xrays",
    }


def test_normalize_track_goal_fills_required_defaults() -> None:
    name, args = _normalize_tool_call("track_goal", {"goal": "lose 5kg"})
    assert name == "track_goal"
    # `goal` is a recognized synonym for `title`.
    assert args["title"] == "lose 5kg"
    assert args["category"] == "personal"
    assert args["status"] == "active"


def test_normalize_passes_unknown_tool_through_untouched() -> None:
    name, args = _normalize_tool_call("not_a_known_tool", {"foo": 1})
    assert name == "not_a_known_tool"
    assert args == {"foo": 1}


def test_normalize_handles_none_args() -> None:
    """A model may emit no args; the helper must coerce to defaults, not crash."""
    name, args = _normalize_tool_call("log_water", None)  # type: ignore[arg-type]
    assert name == "log_water"
    # log_water has a sensible default of 1 glass when no count is given.
    assert args == {"glasses": 1}


def test_normalize_unknown_tool_handles_none_args() -> None:
    name, args = _normalize_tool_call("definitely_not_a_tool", None)  # type: ignore[arg-type]
    assert name == "definitely_not_a_tool"
    assert args == {}
