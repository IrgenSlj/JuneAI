"""Tests for runtime and privacy inspection helpers."""

from unittest.mock import patch

from june_brain.config import resolve_runtime_config
from june_brain.runtime_privacy import (
    build_runtime_privacy_status,
    build_runtime_preset_switch_preview,
    format_runtime_privacy_status,
    format_runtime_preset_switch_plan,
)
from june_brain.tools import (
    get_runtime_privacy_status,
    preview_runtime_preset_switch,
    switch_runtime_preset,
)


def test_local_runtime_is_reported_as_local():
    with patch.dict(
        "os.environ",
        {
            "MODEL_PRESET": "local_mistral_3b",
            "LOCAL_SMALL_MODEL_NAME": "mistral-local-3b",
            "LLM_BASE_URL": "http://127.0.0.1:11434/v1",
        },
        clear=True,
    ):
        runtime = resolve_runtime_config()
        status = build_runtime_privacy_status(runtime)
        text = format_runtime_privacy_status(status)
        tool_text = get_runtime_privacy_status.func()

    assert runtime.mode == "local"
    assert runtime.is_local is True
    assert runtime.is_api is False
    assert runtime.privacy_label == "local-only"
    assert status["local_first"] is True
    assert status["mode"] == "local"
    assert "Local inference stays on this machine." in status["summary"]
    assert "Mode: local LLM" in text
    assert "Runtime privacy status" in tool_text


def test_api_runtime_is_reported_as_api():
    with patch.dict(
        "os.environ",
        {
            "MODEL_PRESET": "claude_high",
            "MODEL_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "test-key",
            "CLAUDE_MODEL_NAME": "claude-test",
        },
        clear=True,
    ):
        runtime = resolve_runtime_config()
        status = build_runtime_privacy_status(runtime)

    assert runtime.mode == "api"
    assert runtime.is_local is False
    assert runtime.is_api is True
    assert runtime.privacy_label == "api-assisted"
    assert status["local_first"] is False
    assert status["mode"] == "api"
    assert status["provider"] == "anthropic"
    assert "remote API" in status["summary"]
    assert "Anthropic" in status["privacy_boundary"]


def test_remote_openai_compatible_endpoint_is_treated_as_api():
    with patch.dict(
        "os.environ",
        {
            "MODEL_PRESET": "local_mistral_3b",
            "LOCAL_SMALL_MODEL_NAME": "mistral-local-3b",
            "LLM_BASE_URL": "https://api.example.com/v1",
        },
        clear=True,
    ):
        runtime = resolve_runtime_config()
        status = build_runtime_privacy_status(runtime)

    assert runtime.mode == "api"
    assert runtime.is_local is False
    assert runtime.privacy_label == "api-assisted"
    assert status["provider"] == "openai_compatible"
    assert status["endpoint_label"] == "https://api.example.com/v1"


def test_previewing_local_to_api_switch_shows_privacy_warning():
    with patch.dict(
        "os.environ",
        {
            "MODEL_PRESET": "local_mistral_8b",
            "LOCAL_LARGE_MODEL_NAME": "mistral-local-8b",
            "MODEL_PROVIDER": "openai_compatible",
            "LLM_BASE_URL": "http://127.0.0.1:11434/v1",
            "MODEL_NAME": "mistral-local-8b",
            "ANTHROPIC_API_KEY": "test-key",
        },
        clear=True,
    ):
        preview = build_runtime_preset_switch_preview("claude_high")
        text = format_runtime_preset_switch_plan(preview)
        tool_text = preview_runtime_preset_switch.func("claude_high")

    assert preview["schema_version"] == 1
    assert preview["requires_confirmation"] is True
    assert preview["applied"] is False
    assert preview["target"]["mode"] == "api"
    assert any("remote provider" in warning for warning in preview["warnings"])
    assert "Runtime switch preview" in text
    assert "Target mode: api" in text
    assert "Runtime switch preview" in tool_text


def test_switch_runtime_preset_requires_confirmation_for_local_to_api():
    with patch.dict(
        "os.environ",
        {
            "MODEL_PRESET": "local_mistral_8b",
            "LOCAL_LARGE_MODEL_NAME": "mistral-local-8b",
            "MODEL_PROVIDER": "openai_compatible",
            "LLM_BASE_URL": "http://127.0.0.1:11434/v1",
            "MODEL_NAME": "mistral-local-8b",
            "ANTHROPIC_API_KEY": "test-key",
        },
        clear=True,
    ):
        result = switch_runtime_preset.func("claude_high")
        runtime = resolve_runtime_config()

    assert runtime.preset_key == "local_mistral_8b"
    assert runtime.is_local is True
    assert runtime.provider == "openai_compatible"
    assert "Confirm the privacy warning" in result
    assert "Runtime switch preview" in result


def test_switch_runtime_preset_applies_when_confirmed():
    with patch.dict(
        "os.environ",
        {
            "MODEL_PRESET": "local_mistral_8b",
            "LOCAL_LARGE_MODEL_NAME": "mistral-local-8b",
            "MODEL_PROVIDER": "openai_compatible",
            "LLM_BASE_URL": "http://127.0.0.1:11434/v1",
            "MODEL_NAME": "mistral-local-8b",
            "ANTHROPIC_API_KEY": "test-key",
        },
        clear=True,
    ):
        result = switch_runtime_preset.func("claude_high", confirm_api_transition=True)
        runtime = resolve_runtime_config()

    assert runtime.preset_key == "claude_high"
    assert runtime.is_api is True
    assert runtime.provider == "anthropic"
    assert runtime.model == "claude-sonnet-4-6"
    assert "Runtime environment updated" in result
    assert "Confirmed: yes" in result
