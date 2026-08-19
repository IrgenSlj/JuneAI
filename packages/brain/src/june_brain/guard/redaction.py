"""Secret redaction — scrub credentials before they are persisted (ADR 0021, S6.4).

OpenClaw stored credentials in plaintext; June must demonstrably not leak them
into the glass-box trace files (which hold the rendered prompt and full tool
I/O). Two layers:

1. Known-value redaction — the actual configured secrets (Gemini / Brave /
   Telegram keys, from env and the OS keyring) are replaced wherever they appear.
   Precise, no false positives.
2. Pattern redaction — narrow regexes for well-known key formats, so a secret a
   user pasted that June never stored is still scrubbed.

Redaction is best-effort and must never raise (a trace is a debug convenience).
"""

from __future__ import annotations

import os
import re

from ..failure import degrade_quietly

REDACTED = "[REDACTED]"

# Env vars that hold secrets. Kept in one place so new keys are easy to add.
_SECRET_ENV_NAMES = (
    "GEMINI_API_KEY",
    "BRAVE_SEARCH_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "LLM_API_KEY",
)

# Keyring secret names (see config_store / secret_store).
_SECRET_STORE_NAMES = ("gemini_api_key",)

_MIN_SECRET_LEN = 8

# Narrow patterns for common credential formats — specific enough to avoid
# scrubbing ordinary prose.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),              # OpenAI-style
    re.compile(r"AIza[A-Za-z0-9_-]{20,}"),             # Google API keys
    re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{30,}\b"),    # Telegram bot token
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{16,}"),       # bearer tokens
)


def _known_secret_values() -> list[str]:
    """Collect the actual configured secret values (env + keyring)."""
    values: list[str] = []
    for name in _SECRET_ENV_NAMES:
        value = (os.environ.get(name) or "").strip()
        if len(value) >= _MIN_SECRET_LEN:
            values.append(value)
    try:
        from june_brain.secret_store import load_secret

        for name in _SECRET_STORE_NAMES:
            value = (load_secret(name) or "").strip()
            if len(value) >= _MIN_SECRET_LEN:
                values.append(value)
    except Exception:  # noqa: BLE001 — redaction must never raise
        degrade_quietly("secret-store redaction sweep")
    # Longest first, so a key that contains another substring is replaced whole.
    return sorted(set(values), key=len, reverse=True)


def redact_secrets(text: str) -> str:
    """Return ``text`` with configured secrets and key-shaped tokens redacted."""
    if not text:
        return text
    try:
        for value in _known_secret_values():
            if value in text:
                text = text.replace(value, REDACTED)
        for pattern in _PATTERNS:
            text = pattern.sub(REDACTED, text)
    except Exception:  # noqa: BLE001 — never break a trace write
        return text
    return text
