"""API key generation and validation for June.

The key is generated on first access and stored in ``config_store``.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import logging

from .config_store import load_stored_config, save_stored_config

logger = logging.getLogger(__name__)

_CONFIG_KEY = "_api_key"
_KEY_BYTES = 32  # 256-bit key → 64 hex chars


def _ensure_key() -> str:
    """Return the stored API key, generating and persisting one if missing."""
    stored = load_stored_config()
    key = stored.extras.get(_CONFIG_KEY)
    if not key:
        key = secrets.token_hex(_KEY_BYTES)
        stored.extras[_CONFIG_KEY] = key
        save_stored_config(stored)
        logger.info("API key generated and stored")
    return key


def get_api_key() -> str:
    """Return the current API key (idempotent)."""
    return _ensure_key()


def validate_api_key(provided: str | None) -> bool:
    """Constant-time comparison of the provided key against the stored key."""
    if not provided:
        return False
    stored = _ensure_key()
    return hmac.compare_digest(stored, provided)


def get_api_key_hash() -> str:
    """Return a SHA-256 prefix of the key for display (e.g. ``abc…1234``)."""
    key = _ensure_key()
    return hashlib.sha256(key.encode()).hexdigest()[:8]
