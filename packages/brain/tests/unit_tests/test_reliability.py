"""Tests for Session 1 reliability fixes."""
import logging
from unittest.mock import patch

import pytest


def test_startup_error_exported_from_graph():
    """graph module exports startup_error at module level."""
    import june_brain.graph as g

    # startup_error is either None (agent started OK) or a string (error message)
    assert hasattr(g, "startup_error")
    assert g.startup_error is None or isinstance(g.startup_error, str)


def test_extract_json_payload_does_not_use_eval():
    """_extract_json_payload must not use ast.literal_eval."""
    import inspect

    import june_brain.graph as g

    source = inspect.getsource(g._extract_json_payload)
    assert "literal_eval" not in source
    assert "ast.literal_eval" not in source


def test_ast_not_imported_in_graph():
    """The ast module must not be imported in agent.graph."""
    import inspect

    import june_brain.graph as g

    source = inspect.getsource(g)
    assert "import ast" not in source


def test_extract_json_payload_handles_clean_json():
    from june_brain.graph import _extract_json_payload

    result = _extract_json_payload('{"name": "log_mood", "args": {"mood": "good"}}')
    assert result == {"name": "log_mood", "args": {"mood": "good"}}


def test_extract_json_payload_returns_none_on_garbage():
    from june_brain.graph import _extract_json_payload

    result = _extract_json_payload("this is not json at all %%%")
    assert result is None


def test_extract_json_payload_handles_embedded_json():
    """Extracts JSON even when surrounded by prose."""
    from june_brain.graph import _extract_json_payload

    result = _extract_json_payload('Sure, here is the call: {"name": "log_water", "args": {}} Done.')
    assert result == {"name": "log_water", "args": {}}


def test_memory_context_cache_returns_same_value_within_ttl(tmp_path):
    """Second call within TTL must not re-read memory files."""
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        from june_brain.memory import Memory
        from june_brain.graph import _build_memory_context, invalidate_context_cache

        invalidate_context_cache("cache_test_user")
        mem = Memory("cache_test_user")

        call_count = {"n": 0}
        original_get_today_summary = mem.get_today_summary

        def counting_get_today_summary():
            call_count["n"] += 1
            return original_get_today_summary()

        # Patch Memory so _build_memory_context uses our counting instance
        with patch("june_brain.graph.Memory") as MockMemory:
            mock_instance = MockMemory.return_value
            mock_instance.get_today_summary.return_value = {}
            mock_instance.get_body_metrics.return_value = []
            mock_instance.get_open_loops.return_value = []
            mock_instance.get_goals.return_value = []
            mock_instance.get_calendar_items.return_value = []

            # Also patch the context_intelligence functions to avoid real reads
            with patch("june_brain.graph.build_recovery_readiness_summary", return_value={}), \
                 patch("june_brain.graph.format_recovery_readiness_summary", return_value=""), \
                 patch("june_brain.graph.build_active_commitments_summary", return_value={}), \
                 patch("june_brain.graph.format_active_commitments_summary", return_value=""):

                invalidate_context_cache("cache_test_user")

                _build_memory_context("cache_test_user")
                first_call_count = mock_instance.get_today_summary.call_count

                # Second call — within TTL — must NOT call get_today_summary again
                _build_memory_context("cache_test_user")
                assert mock_instance.get_today_summary.call_count == first_call_count, (
                    "get_today_summary was called again within the cache TTL"
                )

                invalidate_context_cache("cache_test_user")


def test_invalidate_context_cache_clears_entry(tmp_path):
    """invalidate_context_cache removes the cached entry so the next call re-reads."""
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        from june_brain.graph import _build_memory_context, invalidate_context_cache

        with patch("june_brain.graph.Memory") as MockMemory, \
             patch("june_brain.graph.build_recovery_readiness_summary", return_value={}), \
             patch("june_brain.graph.format_recovery_readiness_summary", return_value=""), \
             patch("june_brain.graph.build_active_commitments_summary", return_value={}), \
             patch("june_brain.graph.format_active_commitments_summary", return_value=""):

            mock_instance = MockMemory.return_value
            mock_instance.get_today_summary.return_value = {}

            invalidate_context_cache("inv_test_user")
            _build_memory_context("inv_test_user")
            after_first = mock_instance.get_today_summary.call_count

            invalidate_context_cache("inv_test_user")
            _build_memory_context("inv_test_user")
            after_second = mock_instance.get_today_summary.call_count

            assert after_second == after_first + 1, (
                "After invalidation, get_today_summary should have been called once more"
            )

            invalidate_context_cache("inv_test_user")


def test_empty_mood_history_returns_empty_list(tmp_path):
    """get_mood_history returns [] when no moods have been logged."""
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        from june_brain.memory import Memory
        mem = Memory("fresh_user")
        result = mem.get_mood_history()

    assert result == []


def test_config_anthropic_raises_without_api_key():
    """_resolve_runtime_config_for_preset raises ValueError when anthropic preset has no key."""
    import os
    from june_brain.config import RUNTIME_PRESETS, _resolve_runtime_config_for_preset

    preset = RUNTIME_PRESETS["claude_high"]
    # Remove any API keys from the environment for this test
    env_patch = {
        "ANTHROPIC_API_KEY": "",
        "LLM_API_KEY": "",
    }
    with patch.dict(os.environ, env_patch, clear=False):
        # Temporarily unset the vars (patch.dict with empty strings still passes _env_text checks)
        import os as _os
        saved_anthropic = _os.environ.pop("ANTHROPIC_API_KEY", None)
        saved_llm = _os.environ.pop("LLM_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                _resolve_runtime_config_for_preset(preset)
        finally:
            if saved_anthropic is not None:
                _os.environ["ANTHROPIC_API_KEY"] = saved_anthropic
            if saved_llm is not None:
                _os.environ["LLM_API_KEY"] = saved_llm
