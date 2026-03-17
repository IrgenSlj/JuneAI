"""Configuration and runtime profiles for JuneAI."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

MEMORY_DIR = os.getenv("MEMORY_DIR", ".june_memory")


@dataclass(frozen=True)
class RuntimePreset:
    """A named model/runtime profile."""

    key: str
    label: str
    provider: str
    model_env_var: str
    default_model: str
    default_base_url: str
    default_api_key: str
    temperature: float
    max_tokens: int
    tool_strategy: str


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved model/runtime configuration."""

    preset_key: str
    label: str
    provider: str
    model: str
    api_key: str
    base_url: str
    temperature: float
    max_tokens: int
    tool_strategy: str

    @property
    def is_local(self) -> bool:
        return self.provider == "openai_compatible"


RUNTIME_PRESETS: dict[str, RuntimePreset] = {
    "local_mistral_3b": RuntimePreset(
        key="local_mistral_3b",
        label="Local Mistral 3B",
        provider="openai_compatible",
        model_env_var="LOCAL_SMALL_MODEL_NAME",
        default_model="mistral",
        default_base_url="http://localhost:11434/v1",
        default_api_key="ollama",
        temperature=0.2,
        max_tokens=700,
        tool_strategy="strict_json_short_turns",
    ),
    "local_mistral_8b": RuntimePreset(
        key="local_mistral_8b",
        label="Local Mistral 8B",
        provider="openai_compatible",
        model_env_var="LOCAL_LARGE_MODEL_NAME",
        default_model="mistral-nemo",
        default_base_url="http://localhost:11434/v1",
        default_api_key="ollama",
        temperature=0.2,
        max_tokens=900,
        tool_strategy="strict_json_short_turns",
    ),
    "claude_high": RuntimePreset(
        key="claude_high",
        label="Claude High Performance",
        provider="anthropic",
        model_env_var="CLAUDE_MODEL_NAME",
        default_model="claude-3-5-sonnet-latest",
        default_base_url="",
        default_api_key="",
        temperature=0.35,
        max_tokens=1200,
        tool_strategy="balanced_reasoning",
    ),
}

DEFAULT_RUNTIME_PRESET = "local_mistral_8b"


def resolve_runtime_config() -> RuntimeConfig:
    """Resolve the active runtime from environment variables."""

    preset_key = os.getenv("MODEL_PRESET", DEFAULT_RUNTIME_PRESET).strip() or DEFAULT_RUNTIME_PRESET
    preset = RUNTIME_PRESETS.get(preset_key, RUNTIME_PRESETS[DEFAULT_RUNTIME_PRESET])

    provider = os.getenv("MODEL_PROVIDER", preset.provider).strip() or preset.provider
    model = os.getenv(preset.model_env_var, "").strip() or os.getenv("MODEL_NAME", "").strip() or preset.default_model

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip()
        base_url = os.getenv("LLM_BASE_URL", "").strip()
    else:
        api_key = os.getenv("LLM_API_KEY", "").strip() or preset.default_api_key
        base_url = os.getenv("LLM_BASE_URL", "").strip() or preset.default_base_url

    temperature = float(os.getenv("MODEL_TEMPERATURE", str(preset.temperature)))
    max_tokens = int(os.getenv("MODEL_MAX_TOKENS", str(preset.max_tokens)))
    tool_strategy = os.getenv("MODEL_TOOL_STRATEGY", preset.tool_strategy).strip() or preset.tool_strategy

    return RuntimeConfig(
        preset_key=preset.key,
        label=preset.label,
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        tool_strategy=tool_strategy,
    )
