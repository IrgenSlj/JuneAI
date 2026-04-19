"""Persistent user configuration for June.

The runtime resolver in ``config.py`` reads from environment variables. That's
fine for developers and CI, but a first-run user who picks a provider in the
setup screen needs that choice to survive a restart. This module adds a
side-channel: a small JSON file at ``JUNE_DATA_DIR/config.json`` that is
loaded once at import time and overlaid onto ``os.environ`` before any runtime
resolution happens.

Precedence (lowest to highest):
    1. Preset defaults (from ``RUNTIME_PRESETS`` in ``config.py``).
    2. ``config.json`` on disk.
    3. Actual environment variables (including values from ``.env``).

Env-first lets developers override persisted choices without editing the file,
which keeps the test and dev loop fast.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from .config import MEMORY_DIR

CONFIG_FILENAME: Final = "config.json"


@dataclass
class StoredConfig:
    """User-facing settings that survive restarts."""

    provider: str | None = None
    gemma_model: str | None = None
    gemini_model: str | None = None
    gemini_api_key: str | None = None
    ollama_base_url: str | None = None
    extras: dict[str, str] = field(default_factory=dict)

    def to_env_patch(self) -> dict[str, str]:
        """Map stored values onto the env vars that ``config.py`` reads."""
        patch: dict[str, str] = {}
        if self.provider:
            patch["MODEL_PROVIDER"] = self.provider
        if self.gemma_model:
            patch["GEMMA_MODEL"] = self.gemma_model
        if self.gemini_model:
            patch["GEMINI_MODEL"] = self.gemini_model
        if self.gemini_api_key:
            patch["GEMINI_API_KEY"] = self.gemini_api_key
        if self.ollama_base_url:
            patch["OLLAMA_BASE_URL"] = self.ollama_base_url
        patch.update(self.extras)
        return patch


def config_path() -> Path:
    """Resolve the config.json path lazily so tests that patch MEMORY_DIR work."""
    return Path(MEMORY_DIR).expanduser() / CONFIG_FILENAME


def load_stored_config() -> StoredConfig:
    """Read config.json if present. Missing or malformed files return defaults."""
    path = config_path()
    if not path.exists():
        return StoredConfig()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return StoredConfig()

    if not isinstance(raw, dict):
        return StoredConfig()

    known = {"provider", "gemma_model", "gemini_model", "gemini_api_key", "ollama_base_url"}
    extras = {
        str(k): str(v)
        for k, v in raw.items()
        if k not in known and isinstance(v, (str, int, float, bool))
    }

    return StoredConfig(
        provider=_as_str(raw.get("provider")),
        gemma_model=_as_str(raw.get("gemma_model")),
        gemini_model=_as_str(raw.get("gemini_model")),
        gemini_api_key=_as_str(raw.get("gemini_api_key")),
        ollama_base_url=_as_str(raw.get("ollama_base_url")),
        extras=extras,
    )


def save_stored_config(config: StoredConfig) -> Path:
    """Write config.json atomically with mode 0600 so keys don't leak to other users."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, str] = {}
    if config.provider:
        payload["provider"] = config.provider
    if config.gemma_model:
        payload["gemma_model"] = config.gemma_model
    if config.gemini_model:
        payload["gemini_model"] = config.gemini_model
    if config.gemini_api_key:
        payload["gemini_api_key"] = config.gemini_api_key
    if config.ollama_base_url:
        payload["ollama_base_url"] = config.ollama_base_url
    payload.update(config.extras)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
    tmp.replace(path)
    return path


def apply_stored_config_to_env() -> StoredConfig:
    """Overlay persisted settings onto os.environ without clobbering existing env vars."""
    stored = load_stored_config()
    for key, value in stored.to_env_patch().items():
        os.environ.setdefault(key, value)
    return stored


def is_configured(stored: StoredConfig | None = None) -> bool:
    """A setup is complete once a provider is chosen and its minimum inputs are present."""
    stored = stored if stored is not None else load_stored_config()
    provider = (stored.provider or os.getenv("MODEL_PROVIDER", "")).strip().lower()
    if provider == "gemini":
        return bool(stored.gemini_api_key or os.getenv("GEMINI_API_KEY"))
    if provider == "gemma":
        return True
    return False


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
