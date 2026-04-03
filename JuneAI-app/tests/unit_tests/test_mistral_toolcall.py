"""Tests for Session 2: Mistral tool calling reliability."""
from agent.config import (
    RUNTIME_PRESETS,
    detect_tool_strategy,
    resolve_runtime_config,
)
from agent.tools import JUNE_TOOLS, JUNE_TOOLS_CORE


def test_local_mistral_7b_preset_exists():
    assert "local_mistral_7b" in RUNTIME_PRESETS
    preset = RUNTIME_PRESETS["local_mistral_7b"]
    assert preset.tool_strategy == "native"
    assert "7b-instruct-v0.3" in preset.default_model


def test_detect_tool_strategy_native_for_v03():
    assert detect_tool_strategy("mistral:7b-instruct-v0.3") == "native"


def test_detect_tool_strategy_native_for_claude():
    assert detect_tool_strategy("claude-sonnet-4-6") == "native"


def test_detect_tool_strategy_recovery_for_3b():
    assert detect_tool_strategy("mistral:3b") == "recovery"


def test_detect_tool_strategy_recovery_for_unknown():
    assert detect_tool_strategy("llama3:8b") == "recovery"


def test_june_tools_core_has_twenty_tools():
    assert len(JUNE_TOOLS_CORE) == 20


def test_june_tools_core_is_subset_of_june_tools():
    full_names = {t.name for t in JUNE_TOOLS}
    core_names = {t.name for t in JUNE_TOOLS_CORE}
    assert core_names.issubset(full_names)


def test_june_tools_full_is_larger_than_core():
    assert len(JUNE_TOOLS) > len(JUNE_TOOLS_CORE)
