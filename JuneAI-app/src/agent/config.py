"""Configuration and runtime profiles for JuneAI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

MEMORY_DIR = os.getenv("MEMORY_DIR", ".june_memory")
LOCAL_LOOPBACK_HOSTNAMES = {"localhost", "127.0.0.1", "::1"}


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
        return resolve_runtime_mode(self.provider, self.base_url) == "local"

    @property
    def is_api(self) -> bool:
        return not self.is_local

    @property
    def mode(self) -> str:
        """Return the runtime transport mode: local or api."""
        return resolve_runtime_mode(self.provider, self.base_url)

    @property
    def privacy_label(self) -> str:
        """Return a compact privacy label for UI display."""
        return "local-only" if self.is_local else "api-assisted"

    @property
    def privacy_boundary(self) -> str:
        """Return a short description of where inference happens."""
        if self.is_local:
            return "Inference stays on this machine."
        if self.provider == "anthropic":
            return "Inference is sent to Anthropic's API."
        if self.provider == "openai_compatible" and self.base_url:
            return f"Inference is sent to {self.base_url}."
        return "Inference is sent to a remote API."


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
    "local_mistral_7b": RuntimePreset(
        key="local_mistral_7b",
        label="Mistral 7B (local)",
        provider="openai_compatible",
        model_env_var="LOCAL_LARGE_MODEL_NAME",
        default_model="mistral:7b-instruct-v0.3",
        default_base_url="http://localhost:11434/v1",
        default_api_key="ollama",
        temperature=0.3,
        max_tokens=4096,
        tool_strategy="native",
    ),
    "local_gemma_4": RuntimePreset(
        key="local_gemma_4",
        label="Gemma 4 (local)",
        provider="openai_compatible",
        model_env_var="LOCAL_GEMMA_MODEL_NAME",
        default_model="gemma4",
        default_base_url="http://localhost:11434/v1",
        default_api_key="ollama",
        temperature=1.0,
        max_tokens=4096,
        tool_strategy="native",
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

DEFAULT_RUNTIME_PRESET = "local_gemma_4"


def detect_tool_strategy(model_name: str) -> str:
    """Return 'native' for models known to support OpenAI-style function calling, else 'recovery'."""
    model_lower = model_name.lower()
    native_patterns = (
        "gemma4",
        "gemma-4",
        "7b-instruct-v0.3",
        "mistral-nemo",
        "mistral-small",
        "mistral-large",
        "mixtral",
        "claude",
        "gpt-4",
        "gpt-3.5-turbo",
        "gemini",
    )
    return "native" if any(p in model_lower for p in native_patterns) else "recovery"


def resolve_runtime_config(preset_key: str | None = None) -> RuntimeConfig:
    """Resolve the active runtime from environment variables or an explicit preset."""
    preset_key = _env_text("MODEL_PRESET", DEFAULT_RUNTIME_PRESET) if preset_key is None else preset_key.strip() or DEFAULT_RUNTIME_PRESET
    preset = RUNTIME_PRESETS.get(preset_key, RUNTIME_PRESETS[DEFAULT_RUNTIME_PRESET])

    return _resolve_runtime_config_for_preset(preset)


def _env_text(name: str, default: str = "") -> str:
    """Read an environment variable as a trimmed string."""
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped or default


def _resolve_runtime_config_for_preset(preset: RuntimePreset) -> RuntimeConfig:
    """Resolve a runtime config for a specific preset using current environment overrides."""

    provider = _env_text("MODEL_PROVIDER", preset.provider)
    model = _env_text(preset.model_env_var) or _env_text("MODEL_NAME") or preset.default_model

    if provider == "anthropic":
        api_key = _env_text("ANTHROPIC_API_KEY") or _env_text("LLM_API_KEY") or preset.default_api_key
        base_url = _env_text("LLM_BASE_URL")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required for the 'claude_high' preset but was not set. "
                "Set it in your .env file."
            )
    else:
        api_key = _env_text("LLM_API_KEY") or preset.default_api_key
        base_url = _env_text("LLM_BASE_URL") or preset.default_base_url

    temperature = float(_env_text("MODEL_TEMPERATURE", str(preset.temperature)))
    max_tokens = int(_env_text("MODEL_MAX_TOKENS", str(preset.max_tokens)))
    _env_tool_strategy = _env_text("MODEL_TOOL_STRATEGY")
    tool_strategy = _env_tool_strategy or preset.tool_strategy or detect_tool_strategy(model)

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


def runtime_preset_options() -> tuple[RuntimePreset, ...]:
    """Return the available runtime presets in display order."""
    return tuple(RUNTIME_PRESETS.values())


def build_runtime_preset_switch_plan(
    preset_key: str,
    current: RuntimeConfig | None = None,
) -> dict[str, Any]:
    """Preview a runtime preset switch without mutating the environment."""
    requested_key = preset_key.strip() or DEFAULT_RUNTIME_PRESET
    preset = RUNTIME_PRESETS.get(requested_key)
    warnings: list[str] = []
    if preset is None:
        warnings.append(
            f"Unknown runtime preset '{preset_key}'. Falling back to '{DEFAULT_RUNTIME_PRESET}'."
        )
        preset = RUNTIME_PRESETS[DEFAULT_RUNTIME_PRESET]

    current_runtime = current or resolve_runtime_config()
    target_runtime = _resolve_target_runtime_for_preset(preset)
    requires_confirmation = current_runtime.is_local and target_runtime.is_api

    if requires_confirmation:
        warnings.append(
            "Switching from local inference to API mode will send prompts to a remote provider."
        )
    if target_runtime.is_api and not target_runtime.api_key:
        warnings.append(
            f"{target_runtime.label} requires an API key before it can be used safely."
        )

    return {
        "schema_version": 1,
        "applied": False,
        "requested_preset_key": requested_key,
        "resolved_preset_key": preset.key,
        "requires_confirmation": requires_confirmation,
        "warnings": warnings,
        "current": {
            "preset_key": current_runtime.preset_key,
            "mode": current_runtime.mode,
            "privacy_label": current_runtime.privacy_label,
            "provider": current_runtime.provider,
            "label": current_runtime.label,
            "model": current_runtime.model,
            "base_url": current_runtime.base_url,
        },
        "target": {
            "preset_key": target_runtime.preset_key,
            "mode": target_runtime.mode,
            "privacy_label": target_runtime.privacy_label,
            "provider": target_runtime.provider,
            "label": target_runtime.label,
            "model": target_runtime.model,
            "base_url": target_runtime.base_url,
        },
        "env_patch": {
            "MODEL_PRESET": preset.key,
            "MODEL_PROVIDER": target_runtime.provider,
            "MODEL_NAME": target_runtime.model,
            preset.model_env_var: target_runtime.model,
            "MODEL_TEMPERATURE": str(target_runtime.temperature),
            "MODEL_MAX_TOKENS": str(target_runtime.max_tokens),
            "MODEL_TOOL_STRATEGY": target_runtime.tool_strategy,
            "LLM_BASE_URL": target_runtime.base_url,
        },
    }


def _resolve_target_runtime_for_preset(preset: RuntimePreset) -> RuntimeConfig:
    """Resolve the target runtime for a preset switch."""
    model = _env_text(preset.model_env_var) or preset.default_model

    if preset.provider == "anthropic":
        api_key = _env_text("ANTHROPIC_API_KEY") or _env_text("LLM_API_KEY")
        base_url = ""
    else:
        api_key = _env_text("LLM_API_KEY") or preset.default_api_key
        base_url = _env_text("LLM_BASE_URL") or preset.default_base_url

    temperature = float(_env_text("MODEL_TEMPERATURE", str(preset.temperature)))
    max_tokens = int(_env_text("MODEL_MAX_TOKENS", str(preset.max_tokens)))
    tool_strategy = _env_text("MODEL_TOOL_STRATEGY", preset.tool_strategy)

    return RuntimeConfig(
        preset_key=preset.key,
        label=preset.label,
        provider=preset.provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        tool_strategy=tool_strategy,
    )


def apply_runtime_preset_switch(
    preset_key: str,
    *,
    confirm_api_transition: bool = False,
) -> dict[str, Any]:
    """Apply a runtime preset switch, blocking local-to-API transitions unless confirmed."""
    plan = build_runtime_preset_switch_plan(preset_key)
    target_mode = str(plan["target"]["mode"])
    if plan["requires_confirmation"] and not confirm_api_transition:
        plan["warnings"] = [
            *plan["warnings"],
            "Confirm the privacy warning before switching away from local inference.",
        ]
        return plan
    if target_mode == "api":
        target_preset = RUNTIME_PRESETS.get(str(plan["resolved_preset_key"]), RUNTIME_PRESETS[DEFAULT_RUNTIME_PRESET])
        target_runtime = _resolve_target_runtime_for_preset(target_preset)
        if not target_runtime.api_key:
            plan["warnings"] = [
                *plan["warnings"],
                f"{target_runtime.label} needs ANTHROPIC_API_KEY or LLM_API_KEY before it can run.",
            ]
            return plan

    target_preset = RUNTIME_PRESETS.get(str(plan["resolved_preset_key"]), RUNTIME_PRESETS[DEFAULT_RUNTIME_PRESET])
    target_runtime = _resolve_target_runtime_for_preset(target_preset)
    os.environ["MODEL_PRESET"] = target_preset.key
    os.environ["MODEL_PROVIDER"] = target_runtime.provider
    os.environ["MODEL_NAME"] = target_runtime.model
    os.environ[target_preset.model_env_var] = target_runtime.model
    os.environ["MODEL_TEMPERATURE"] = str(target_runtime.temperature)
    os.environ["MODEL_MAX_TOKENS"] = str(target_runtime.max_tokens)
    os.environ["MODEL_TOOL_STRATEGY"] = target_runtime.tool_strategy
    os.environ["LLM_BASE_URL"] = target_runtime.base_url
    if target_runtime.provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = target_runtime.api_key
    else:
        os.environ["LLM_API_KEY"] = target_runtime.api_key

    plan["applied"] = True
    plan["current"] = {
        "preset_key": target_runtime.preset_key,
        "mode": target_runtime.mode,
        "privacy_label": target_runtime.privacy_label,
        "provider": target_runtime.provider,
        "label": target_runtime.label,
        "model": target_runtime.model,
        "base_url": target_runtime.base_url,
    }
    plan["warnings"] = [
        *plan["warnings"],
        "Runtime environment updated for the current process.",
    ]
    return plan


def is_loopback_base_url(base_url: str) -> bool:
    """Return True when a base URL points to localhost or another loopback host."""
    parsed = urlparse(base_url.strip())
    host = (parsed.hostname or "").lower()
    return host in LOCAL_LOOPBACK_HOSTNAMES


def resolve_runtime_mode(provider: str, base_url: str) -> str:
    """Classify the runtime transport as local or api."""
    normalized_provider = provider.strip().lower()
    if normalized_provider == "anthropic":
        return "api"
    if normalized_provider == "openai_compatible":
        return "local" if is_loopback_base_url(base_url) else "api"
    return "api"
