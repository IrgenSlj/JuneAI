"""Tests for runtime-aware tool selection.

Per ADR 0002 the product ships with two presets only: gemma (local) and
gemini (cloud). Gemma runs the trimmed tool list to keep small-model
prompts manageable; Gemini uses the full set.
"""
from june_brain.config import (
    RUNTIME_PRESETS,
    detect_tool_strategy,
    resolve_runtime_config,
)
from june_brain.graph import _select_tools_for_runtime
from june_brain.tools import JUNE_TOOLS, JUNE_TOOLS_GEMMA


def test_gemma_preset_exists():
    assert "gemma" in RUNTIME_PRESETS
    preset = RUNTIME_PRESETS["gemma"]
    assert preset.tool_strategy == "native"
    assert preset.default_model == "gemma4:e4b"


def test_gemini_preset_exists():
    assert "gemini" in RUNTIME_PRESETS
    preset = RUNTIME_PRESETS["gemini"]
    assert preset.tool_strategy == "native"
    assert preset.default_model.startswith("gemini")


def test_detect_tool_strategy_is_native_for_any_model():
    # Both supported families use native OpenAI-compatible tool calling.
    assert detect_tool_strategy("gemma4:e4b") == "native"
    assert detect_tool_strategy("gemini-2.0-flash") == "native"


def test_gemma_tools_are_subset_of_full_tools():
    full_names = {t.name for t in JUNE_TOOLS}
    gemma_names = {t.name for t in JUNE_TOOLS_GEMMA}
    assert gemma_names.issubset(full_names)


def test_gemma_tools_is_a_trimmed_subset():
    assert len(JUNE_TOOLS_GEMMA) < len(JUNE_TOOLS)


def test_graph_uses_gemma_tool_subset_for_gemma_runtime():
    selected = _select_tools_for_runtime(resolve_runtime_config("gemma"))
    assert {tool.name for tool in selected} == {tool.name for tool in JUNE_TOOLS_GEMMA}


def test_graph_uses_full_tools_for_gemini_runtime(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    selected = _select_tools_for_runtime(resolve_runtime_config("gemini"))
    assert {tool.name for tool in selected} == {tool.name for tool in JUNE_TOOLS}
