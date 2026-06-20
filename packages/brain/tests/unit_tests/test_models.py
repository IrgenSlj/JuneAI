from unittest.mock import patch

from june_brain.config import resolve_runtime_config, runtime_preset_options
from june_brain.models import build_chat_model
from pydantic import SecretStr


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


def test_build_chat_model_uses_current_openai_signature():
    runtime = resolve_runtime_config()
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch("june_brain.models.ChatOpenAI", FakeChatOpenAI):
        build_chat_model(runtime)

    assert captured["model"] == runtime.model
    assert isinstance(captured["api_key"], SecretStr)
    assert captured["api_key"].get_secret_value() == runtime.api_key
    assert captured["base_url"] == runtime.base_url
    assert captured["max_completion_tokens"] == runtime.max_tokens
    assert captured["streaming"] is True
    assert captured["timeout"] == 120


def test_build_chat_model_pins_keep_alive_for_local():
    runtime = resolve_runtime_config("gemma")
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch("june_brain.models.ChatOpenAI", FakeChatOpenAI):
        build_chat_model(runtime)

    # Local Ollama gets keep_alive=-1 so the model stays resident between turns.
    assert captured.get("extra_body") == {"keep_alive": -1}


def test_build_chat_model_omits_keep_alive_for_cloud():
    with patch.dict(
        "os.environ",
        {"MODEL_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"},
        clear=False,
    ):
        runtime = resolve_runtime_config()
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch("june_brain.models.ChatOpenAI", FakeChatOpenAI):
        build_chat_model(runtime)

    # Cloud providers must not receive Ollama's keep_alive field.
    assert "extra_body" not in captured
