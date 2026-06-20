"""Cross-platform credential storage for June.

Secrets (currently just the Gemini API key) should live in the OS credential
store whenever possible:

- macOS: Keychain
- Linux with desktop environment: Secret Service / GNOME Keyring / KWallet
- Windows: Credential Manager

When no keyring backend is available (headless Linux, CI, Docker) we fall
back to the mode-0600 ``config.json`` managed by ``config_store``. The fall
back is intentional — a missing keyring should not block a developer from
using the product — but the settings UI surfaces which storage is active so
the user knows.

``keyring`` is a small, pure-Python dependency with platform-specific
backends loaded at runtime. Importing it is cheap; calling it without a
backend returns ``None`` rather than raising, which lets callers treat
"no keyring available" as a normal branch.
"""

from __future__ import annotations

from typing import Literal

SERVICE_NAME = "JuneAI"

SecretLocation = Literal["keyring", "file", "none"]


def save_secret(name: str, value: str) -> SecretLocation:
    """Persist a secret. Returns where it actually landed.

    ``"keyring"`` means the OS credential store accepted it. ``"file"`` means
    the caller should fall back to the JSON config (this module doesn't
    touch that file itself — ``config_store.save_stored_config`` handles the
    write after inspecting the return value).
    """
    backend = _load_keyring()
    if backend is None:
        return "file"
    try:
        backend.set_password(SERVICE_NAME, name, value)
    except Exception:  # noqa: BLE001 — any backend error means we fall back to file
        return "file"
    return "keyring"


def load_secret(name: str) -> str | None:
    """Read a secret from the OS credential store. Returns None when absent."""
    backend = _load_keyring()
    if backend is None:
        return None
    try:
        return backend.get_password(SERVICE_NAME, name)
    except Exception:  # noqa: BLE001 — malformed keychain entries shouldn't crash startup
        return None


def delete_secret(name: str) -> bool:
    """Remove a secret from the OS credential store.

    Returns True when a delete was performed, False when no keyring is
    available. A missing entry also returns False — callers that want
    belt-and-suspenders deletion should also wipe the JSON fallback.
    """
    backend = _load_keyring()
    if backend is None:
        return False
    try:
        backend.delete_password(SERVICE_NAME, name)
        return True
    except Exception:  # noqa: BLE001 — entry missing or backend quirk; treat as no-op
        return False


def keyring_available() -> bool:
    """True when a usable OS credential backend is loaded."""
    return _load_keyring() is not None


def _load_keyring():
    """Import ``keyring`` lazily so the brain starts even if it's uninstalled."""
    try:
        import keyring
        from keyring.backends import fail
    except ImportError:
        return None

    backend = keyring.get_keyring()
    if isinstance(backend, fail.Keyring):
        return None
    return backend
