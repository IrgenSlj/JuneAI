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
from .secret_store import (
    SecretLocation,
    delete_secret,
    keyring_available,
    load_secret,
    save_secret,
)

CONFIG_FILENAME: Final = "config.json"
GEMINI_KEY_NAME: Final = "gemini_api_key"


@dataclass
class StoredConfig:
    """User-facing settings that survive restarts."""

    provider: str | None = None
    user_name: str | None = None
    gemma_model: str | None = None
    gemini_model: str | None = None
    gemini_api_key: str | None = None
    ollama_base_url: str | None = None
    privacy_dial: str | None = None
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
    """Read config.json if present. Missing or malformed files return defaults.

    The Gemini API key is pulled from the OS credential store first; only if
    no keyring backend is present do we fall back to the ``gemini_api_key``
    field in the JSON file.
    """
    path = config_path()
    raw: dict[str, object] = {}
    if path.exists():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            raw = parsed

    known = {
        "provider",
        "user_name",
        "gemma_model",
        "gemini_model",
        "gemini_api_key",
        "ollama_base_url",
        "privacy_dial",
    }
    extras = {
        str(k): str(v)
        for k, v in raw.items()
        if k not in known and isinstance(v, (str, int, float, bool))
    }

    file_gemini_key = _as_str(raw.get("gemini_api_key"))
    env_gemini_key = _as_str(os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY"))
    provider = _active_provider(raw)
    gemini_key = env_gemini_key
    if gemini_key is None and provider == "gemini":
        gemini_key = load_secret(GEMINI_KEY_NAME) or file_gemini_key
    elif file_gemini_key is not None:
        # Local Gemma startup must not unlock/probe the OS keychain just to find
        # an optional cloud key. A file-backed key is already available without
        # prompting and can still be shown as present in settings.
        gemini_key = file_gemini_key

    return StoredConfig(
        provider=_as_str(raw.get("provider")),
        user_name=_as_str(raw.get("user_name")),
        gemma_model=_as_str(raw.get("gemma_model")),
        gemini_model=_as_str(raw.get("gemini_model")),
        gemini_api_key=gemini_key,
        ollama_base_url=_as_str(raw.get("ollama_base_url")),
        privacy_dial=_as_str(raw.get("privacy_dial")),
        extras=extras,
    )


def save_stored_config(config: StoredConfig) -> Path:
    """Write config.json atomically with mode 0600.

    The Gemini API key is routed through the OS credential store when one is
    available; only if that fails does it land in the file. In either case the
    JSON blob keeps non-secret fields (provider, model tags, base URL).
    """
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    key_location: SecretLocation = "none"
    if config.gemini_api_key:
        key_location = save_secret(GEMINI_KEY_NAME, config.gemini_api_key)
    elif keyring_available():
        # Ensure a prior keyring entry doesn't outlive an explicit clear.
        delete_secret(GEMINI_KEY_NAME)

    payload: dict[str, str] = {}
    if config.provider:
        payload["provider"] = config.provider
    if config.user_name:
        payload["user_name"] = config.user_name
    if config.gemma_model:
        payload["gemma_model"] = config.gemma_model
    if config.gemini_model:
        payload["gemini_model"] = config.gemini_model
    if config.gemini_api_key and key_location != "keyring":
        payload["gemini_api_key"] = config.gemini_api_key
    if config.ollama_base_url:
        payload["ollama_base_url"] = config.ollama_base_url
    if config.privacy_dial:
        payload["privacy_dial"] = config.privacy_dial
    payload.update(config.extras)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
    tmp.replace(path)
    return path


def forget_gemini_key() -> SecretLocation:
    """Remove the Gemini key from both the keyring and the JSON fallback.

    Returns where the key had been living so the UI can report whether a
    keyring or file entry was removed. ``"none"`` means there was nothing to
    delete. Callers should also scrub the environment via ``os.environ.pop``
    if the current process might still hold the old value in memory.
    """
    location: SecretLocation = "none"
    if delete_secret(GEMINI_KEY_NAME):
        location = "keyring"

    path = config_path()
    if path.exists():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict) and "gemini_api_key" in parsed:
            parsed.pop("gemini_api_key", None)
            path.write_text(json.dumps(parsed, indent=2, sort_keys=True), encoding="utf-8")
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
            if location == "none":
                location = "file"
    return location


def gemini_key_location() -> SecretLocation:
    """Report where the active Gemini key lives, without returning its value."""
    path = config_path()
    if path.exists():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict) and parsed.get("gemini_api_key"):
            return "file"
        raw = parsed if isinstance(parsed, dict) else {}
    else:
        raw = {}
    if _active_provider(raw) == "gemini" and load_secret(GEMINI_KEY_NAME):
        return "keyring"
    return "none"


def apply_stored_config_to_env() -> StoredConfig:
    """Overlay persisted settings onto os.environ without clobbering existing env vars."""
    stored = load_stored_config()
    for key, value in stored.to_env_patch().items():
        os.environ.setdefault(key, value)
    return stored


def is_configured(stored: StoredConfig | None = None) -> bool:
    """A setup is complete once a provider is chosen and its minimum inputs are present."""
    stored = stored if stored is not None else load_stored_config()
    provider = (stored.provider or os.getenv("MODEL_PROVIDER", "") or "").strip().lower()
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


def _active_provider(raw: dict[str, object]) -> str:
    return (
        _as_str(os.getenv("MODEL_PROVIDER"))
        or _as_str(raw.get("provider"))
        or ""
    ).lower()


def get_privacy_dial():  # type: ignore[no-untyped-def]
    """Return the user's persisted privacy dial, defaulting to PRIVATE_BY_DEFAULT.

    Lazy-imports the enum from ``routing`` to avoid a module-load cycle.
    """
    from .routing import UserPrivacyDial

    stored = load_stored_config()
    raw = (stored.privacy_dial or "").strip().lower()
    try:
        return UserPrivacyDial(raw) if raw else UserPrivacyDial.PRIVATE_BY_DEFAULT
    except ValueError:
        return UserPrivacyDial.PRIVATE_BY_DEFAULT


def set_privacy_dial(dial) -> Path:  # type: ignore[no-untyped-def]
    """Persist the user's privacy dial and return the config file path."""
    from .routing import UserPrivacyDial

    if not isinstance(dial, UserPrivacyDial):
        raise TypeError(f"Expected UserPrivacyDial, got {type(dial).__name__}")
    stored = load_stored_config()
    stored.privacy_dial = dial.value
    return save_stored_config(stored)


_SALIENCE_KEYS = ("rel", "rec", "freq")


def get_salience_weights() -> dict[str, float]:
    """Read runtime salience weights from config.json (empty when unset).

    Keys: ``rel``, ``rec``, ``freq``. These are the user-tunable recall weights
    (S5.4); ``SalienceWeights.load`` layers them over env and defaults.
    """
    path = config_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key in _SALIENCE_KEYS:
        value = raw.get(f"salience_{key}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = float(value)
    return out


def set_salience_weights(
    *, rel: float | None = None, rec: float | None = None, freq: float | None = None
) -> Path:
    """Persist salience weights into config.json, merging with existing config."""
    path = config_path()
    raw: dict[str, object] = {}
    if path.exists():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                raw = parsed
        except (OSError, json.JSONDecodeError):
            raw = {}
    for key, value in (("rel", rel), ("rec", rec), ("freq", freq)):
        if value is not None:
            raw[f"salience_{key}"] = float(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
    return path
