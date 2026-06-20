from unittest.mock import patch

from june_brain.config import resolve_runtime_config, runtime_preset_options


def test_resolve_runtime_config_for_gemma():
    with patch.dict(
        "os.environ",
        {
            "MODEL_PROVIDER": "gemma",
            "GEMMA_MODEL": "gemma4:custom",
            "OLLAMA_BASE_URL": "http://127.0.0.1:11434/v1",
        },
        clear=False,
    ):
        runtime = resolve_runtime_config()

    assert runtime.provider == "openai_compatible"
    assert runtime.label == "Gemma 4 (local)"
    assert runtime.model == "gemma4:custom"
    assert runtime.base_url == "http://127.0.0.1:11434/v1"
    assert runtime.is_local is True


def test_resolve_runtime_config_for_gemini():
    with patch.dict(
        "os.environ",
        {
            "MODEL_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key",
            "GEMINI_MODEL": "gemini-2.0-flash",
        },
        clear=False,
    ):
        runtime = resolve_runtime_config()

    assert runtime.provider == "openai_compatible"
    assert runtime.label == "Gemini (cloud)"
    assert runtime.model == "gemini-2.0-flash"
    assert runtime.api_key == "test-key"
    assert runtime.is_api is True


def test_resolve_runtime_config_accepts_explicit_preset_key():
    with patch.dict("os.environ", {}, clear=False):
        runtime = resolve_runtime_config("gemma")

    assert runtime.preset_key == "gemma"
    assert runtime.label == "Gemma 4 (local)"


def test_runtime_preset_options_expose_known_presets():
    option_keys = [preset.key for preset in runtime_preset_options()]

    assert "gemma" in option_keys
    assert "gemini" in option_keys
    assert len(option_keys) == 2

