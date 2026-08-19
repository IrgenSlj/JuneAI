"""Tests for Session 1 reliability fixes."""
from unittest.mock import patch

import pytest


def test_extract_json_payload_does_not_use_eval():
    """_extract_json_payload must not use ast.literal_eval."""
    import inspect

    from june_brain.loop import agent_helpers

    source = inspect.getsource(agent_helpers._extract_json_payload)
    assert "literal_eval" not in source
    assert "ast.literal_eval" not in source


def test_ast_not_imported_in_agent_helpers():
    """The ast module must not be imported in agent_helpers."""
    import inspect

    from june_brain.loop import agent_helpers

    source = inspect.getsource(agent_helpers)
    assert "import ast" not in source


def test_extract_json_payload_handles_clean_json():
    from june_brain.loop.agent_helpers import _extract_json_payload

    result = _extract_json_payload('{"name": "log_mood", "args": {"mood": "good"}}')
    assert result == {"name": "log_mood", "args": {"mood": "good"}}


def test_extract_json_payload_returns_none_on_garbage():
    from june_brain.loop.agent_helpers import _extract_json_payload

    result = _extract_json_payload("this is not json at all %%%")
    assert result is None


def test_extract_json_payload_handles_embedded_json():
    """Extracts JSON even when surrounded by prose."""
    from june_brain.loop.agent_helpers import _extract_json_payload

    result = _extract_json_payload('Sure, here is the call: {"name": "log_water", "args": {}} Done.')
    assert result == {"name": "log_water", "args": {}}


def test_empty_structured_reads_return_empty_lists(tmp_path):
    """A fresh user's structured reads return [], not None and not a crash.

    Previously asserted through get_mood_history, which went with the health
    cluster (D.5b). The property is about the empty-store path, so it now uses
    the structured stores that still have readers.
    """
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        from june_brain.memory import Memory
        mem = Memory("fresh_user")
        results = [mem.get_journal(), mem.get_goals(), mem.get_open_loops(),
                   mem.get_preferences(), mem.get_relationship_profiles()]

    assert results == [[], [], [], [], []]


def test_config_gemini_raises_without_api_key():
    """_resolve_runtime_config_for_preset raises ValueError when gemini preset has no key."""
    import os

    from june_brain.config import RUNTIME_PRESETS, _resolve_runtime_config_for_preset

    preset = RUNTIME_PRESETS["gemini"]
    saved_gemini = os.environ.pop("GEMINI_API_KEY", None)
    saved_llm = os.environ.pop("LLM_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            _resolve_runtime_config_for_preset(preset)
    finally:
        if saved_gemini is not None:
            os.environ["GEMINI_API_KEY"] = saved_gemini
        if saved_llm is not None:
            os.environ["LLM_API_KEY"] = saved_llm


def test_a_prose_answer_does_not_log_the_user_s_content(caplog) -> None:
    """Two papercuts in one line, both on the ordinary path.

    `_extract_json_payload` runs on every turn to test for a tool call, so
    "no JSON here" is what a normal answer looks like — and a markdown list
    reaches the failure branch, because `[` and `]` parse as an array opener.
    It logged that at WARNING, with 120 characters of the model's reply, which
    is the answer given to the user and carries whatever was recalled into the
    turn. The ledger already follows "shape, never content" for injection
    reports; so does this now.
    """
    import logging

    from june_brain.loop.agent_helpers import _extract_json_payload

    answer = (
        "You have the following open promises:\n"
        "- [0f506320] Find a dentist and book a check-up\n"
        "- [6a1b2c3d] Renew the passport"
    )
    with caplog.at_level(logging.DEBUG):
        assert _extract_json_payload(answer) is None

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
        "an ordinary prose answer logged at WARNING or above"
    )
    assert "dentist" not in caplog.text and "passport" not in caplog.text, (
        "the model's answer text reached the log"
    )
