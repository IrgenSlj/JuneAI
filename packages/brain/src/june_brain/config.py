"""Configuration and runtime profiles for June.

Per ADR 0002, June supports exactly two model providers: Gemma 4 (local via
Ollama) and Gemini (cloud via Google's OpenAI-compatible endpoint). Both are
reached through the model-specific provider layer (ADR 0017).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

def _default_data_dir() -> str:
    """Return the platform-appropriate directory for June's user data."""
    if sys.platform == "darwin":
        return str(Path.home() / "Library" / "Application Support" / "June")
    if sys.platform == "win32":
        base = os.getenv("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return str(Path(base) / "June")
    base = os.getenv("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return str(Path(base) / "June")


MEMORY_DIR = os.getenv("JUNE_DATA_DIR") or os.getenv("MEMORY_DIR") or _default_data_dir()
LOCAL_LOOPBACK_HOSTNAMES = {"localhost", "127.0.0.1", "::1"}

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


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
    prompt_style: str = "openai_compatible"


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
    prompt_style: str = "openai_compatible"

    @property
    def is_local(self) -> bool:
        return resolve_runtime_mode(self.provider, self.base_url) == "local"

    @property
    def is_api(self) -> bool:
        return not self.is_local

    @property
    def mode(self) -> str:
        return resolve_runtime_mode(self.provider, self.base_url)

    @property
    def privacy_label(self) -> str:
        return "local-only" if self.is_local else "api-assisted"

    @property
    def privacy_boundary(self) -> str:
        if self.is_local:
            return "Inference stays on this machine."
        if self.base_url:
            return f"Inference is sent to {self.base_url}."
        return "Inference is sent to a remote API."


RUNTIME_PRESETS: dict[str, RuntimePreset] = {
    "gemma": RuntimePreset(
        key="gemma",
        label="Gemma 4 (local)",
        provider="openai_compatible",
        model_env_var="GEMMA_MODEL",
        # Smallest Gemma 4 by default: fastest cold start and lowest memory, so
        # June runs well on modest machines out of the box. Users can point
        # GEMMA_MODEL at a larger tag (e4b/26b/31b) for more capability.
        default_model="gemma4:e2b",
        default_base_url="http://localhost:11434/v1",
        default_api_key="ollama",
        temperature=0.4,
        max_tokens=4096,
        tool_strategy="native",
        prompt_style="gemma",
    ),
    "gemini": RuntimePreset(
        key="gemini",
        label="Gemini (cloud)",
        provider="openai_compatible",
        model_env_var="GEMINI_MODEL",
        default_model="gemini-2.0-flash",
        default_base_url=GEMINI_OPENAI_BASE_URL,
        default_api_key="",
        temperature=0.4,
        max_tokens=4096,
        tool_strategy="native",
        prompt_style="gemma",
    ),
}

DEFAULT_RUNTIME_PRESET = "gemma"


def detect_tool_strategy(model_name: str) -> str:
    """Both supported families use native OpenAI-compatible tool calling."""
    return "native"


def resolve_runtime_config(preset_key: str | None = None) -> RuntimeConfig:
    """Resolve the active runtime from env vars or an explicit preset key."""
    if preset_key is None:
        preset_key = _env_text("MODEL_PROVIDER") or _env_text("MODEL_PRESET") or DEFAULT_RUNTIME_PRESET
    else:
        preset_key = preset_key.strip() or DEFAULT_RUNTIME_PRESET

    preset = RUNTIME_PRESETS.get(preset_key, RUNTIME_PRESETS[DEFAULT_RUNTIME_PRESET])
    return _resolve_runtime_config_for_preset(preset)


def _env_text(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped or default


def _resolve_api_key_for_preset(preset: RuntimePreset) -> str:
    """Read the appropriate API key env var for the given preset."""
    if preset.key == "gemini":
        return _env_text("GEMINI_API_KEY") or _env_text("LLM_API_KEY") or preset.default_api_key
    return _env_text("LLM_API_KEY") or preset.default_api_key


def _resolve_runtime_config_for_preset(preset: RuntimePreset) -> RuntimeConfig:
    """Resolve a runtime config for a specific preset using current environment overrides."""
    model = _env_text(preset.model_env_var) or _env_text("MODEL_NAME") or preset.default_model
    api_key = _resolve_api_key_for_preset(preset)

    if preset.key == "gemma":
        base_url = _env_text("OLLAMA_BASE_URL") or _env_text("LLM_BASE_URL") or preset.default_base_url
    else:
        base_url = _env_text("LLM_BASE_URL") or preset.default_base_url

    if preset.key == "gemini" and not api_key:
        raise ValueError(
            "GEMINI_API_KEY is required for the 'gemini' preset but was not set. "
            "Get a key at https://aistudio.google.com and set it in your .env."
        )

    if not _is_valid_http_url(base_url):
        raise ValueError(
            f"Invalid base_url for preset '{preset.key}': {base_url!r}. "
            "Expected an http:// or https:// URL with a host."
        )

    temperature = float(_env_text("MODEL_TEMPERATURE", str(preset.temperature)))
    max_tokens = int(_env_text("MODEL_MAX_TOKENS", str(preset.max_tokens)))
    tool_strategy = _env_text("MODEL_TOOL_STRATEGY") or preset.tool_strategy

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
        prompt_style=preset.prompt_style,
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
        "current": _runtime_summary(current_runtime),
        "target": _runtime_summary(target_runtime),
        "env_patch": {
            "MODEL_PROVIDER": preset.key,
            "MODEL_NAME": target_runtime.model,
            preset.model_env_var: target_runtime.model,
            "MODEL_TEMPERATURE": str(target_runtime.temperature),
            "MODEL_MAX_TOKENS": str(target_runtime.max_tokens),
            "LLM_BASE_URL": target_runtime.base_url,
        },
    }


def _resolve_target_runtime_for_preset(preset: RuntimePreset) -> RuntimeConfig:
    """Resolve the target runtime for a preset switch (tolerates missing keys)."""
    model = _env_text(preset.model_env_var) or preset.default_model
    api_key = _resolve_api_key_for_preset(preset)
    if preset.key == "gemma":
        base_url = _env_text("OLLAMA_BASE_URL") or _env_text("LLM_BASE_URL") or preset.default_base_url
    else:
        base_url = _env_text("LLM_BASE_URL") or preset.default_base_url
    temperature = float(_env_text("MODEL_TEMPERATURE", str(preset.temperature)))
    max_tokens = int(_env_text("MODEL_MAX_TOKENS", str(preset.max_tokens)))
    tool_strategy = _env_text("MODEL_TOOL_STRATEGY") or preset.tool_strategy

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
        prompt_style=preset.prompt_style,
    )


def _runtime_summary(runtime: RuntimeConfig) -> dict[str, Any]:
    return {
        "preset_key": runtime.preset_key,
        "mode": runtime.mode,
        "privacy_label": runtime.privacy_label,
        "provider": runtime.provider,
        "label": runtime.label,
        "model": runtime.model,
        "base_url": runtime.base_url,
    }


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

    target_preset = RUNTIME_PRESETS.get(
        str(plan["resolved_preset_key"]),
        RUNTIME_PRESETS[DEFAULT_RUNTIME_PRESET],
    )
    target_runtime = _resolve_target_runtime_for_preset(target_preset)
    if target_mode == "api" and not target_runtime.api_key:
        plan["warnings"] = [
            *plan["warnings"],
            f"{target_runtime.label} needs GEMINI_API_KEY or LLM_API_KEY before it can run.",
        ]
        return plan

    os.environ["MODEL_PROVIDER"] = target_preset.key
    os.environ["MODEL_NAME"] = target_runtime.model
    os.environ[target_preset.model_env_var] = target_runtime.model
    os.environ["MODEL_TEMPERATURE"] = str(target_runtime.temperature)
    os.environ["MODEL_MAX_TOKENS"] = str(target_runtime.max_tokens)
    if target_preset.key == "gemini":
        os.environ["GEMINI_API_KEY"] = target_runtime.api_key
        os.environ["LLM_BASE_URL"] = target_runtime.base_url
    else:
        os.environ["LLM_API_KEY"] = target_runtime.api_key
        os.environ["OLLAMA_BASE_URL"] = target_runtime.base_url

    plan["applied"] = True
    plan["current"] = _runtime_summary(target_runtime)
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
    if normalized_provider == "openai_compatible":
        return "local" if is_loopback_base_url(base_url) else "api"
    return "api"


def _is_valid_http_url(value: str) -> bool:
    """Return True when ``value`` is an http(s) URL with a host."""
    if not value or not value.strip():
        return False
    try:
        parsed = urlparse(value.strip())
    except (TypeError, ValueError):
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)
