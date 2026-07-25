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

**Every call here runs under a hard deadline.** A credential store can block
*indefinitely* rather than fail: on macOS, an item whose ACL does not list the
calling binary makes ``SecItemCopyMatching`` wait on an authorization decision
that never arrives in a headless process. That is not hypothetical — the
packaged (ad-hoc-signed) sidecar has a different code identity from the dev
interpreter that created the items, so every chat turn hung forever inside
``SecKeychainItemCopyContent`` while the Trust Ledger tried to load its signing
key. A secret read must never be able to stall a turn, so a call that overruns
``JUNE_KEYRING_TIMEOUT_S`` (default 2s) is abandoned and the whole module
latches into a degraded mode that returns the file-fallback answer immediately.

The latch is deliberate and process-wide: a thread blocked in the platform
keychain cannot be cancelled, so retrying would leak one stuck daemon thread per
call. One abandoned thread per process is an acceptable price for never hanging;
one per call is not.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import Any, Literal

logger = logging.getLogger(__name__)

SERVICE_NAME = "JuneAI"

SecretLocation = Literal["keyring", "file", "none"]

# Deadline for any single credential-store operation. Generous enough that a
# healthy keychain (single-digit milliseconds) is never affected, short enough
# that a blocked one cannot be felt inside a chat turn.
DEFAULT_TIMEOUT_S = 2.0

_unresponsive = threading.Event()
_warned = threading.Event()


def _timeout_s() -> float:
    """Read the deadline at call time so tests and operators can tune it."""
    raw = os.getenv("JUNE_KEYRING_TIMEOUT_S", "")
    try:
        value = float(raw) if raw else DEFAULT_TIMEOUT_S
    except ValueError:
        return DEFAULT_TIMEOUT_S
    return value if value > 0 else DEFAULT_TIMEOUT_S


def keyring_unresponsive() -> bool:
    """True once a credential-store call has overrun its deadline.

    Sticky for the life of the process. Callers that surface storage location to
    the user (settings) can use this to explain why a secret landed in the file
    fallback instead of the OS credential store.
    """
    return _unresponsive.is_set()


def _reset_unresponsive_for_tests() -> None:
    """Clear the latch. Tests only — production never un-latches."""
    _unresponsive.clear()
    _warned.clear()


def _run_guarded(op: str, fn: Callable[[], Any], fallback: Any) -> Any:
    """Run ``fn`` under the deadline, returning ``fallback`` if it overruns.

    The worker is a daemon thread: if the platform call never returns, the
    thread stays parked for the life of the process and cannot hold up exit.
    """
    if _unresponsive.is_set():
        return fallback

    result: list[Any] = [fallback]

    def _target() -> None:
        try:
            result[0] = fn()
        except Exception:  # noqa: BLE001 — any backend error means fall back
            result[0] = fallback

    worker = threading.Thread(
        target=_target, name=f"june-keyring-{op}", daemon=True
    )
    worker.start()
    worker.join(_timeout_s())

    if worker.is_alive():
        _unresponsive.set()
        if not _warned.is_set():
            _warned.set()
            logger.warning(
                "credential store did not respond within %.1fs during %s; "
                "falling back to file storage for the rest of this session. "
                "On macOS this usually means the keychain item was created by a "
                "different build and is waiting on an authorization prompt.",
                _timeout_s(),
                op,
            )
        return fallback

    return result[0]


def save_secret(name: str, value: str) -> SecretLocation:
    """Persist a secret. Returns where it actually landed.

    ``"keyring"`` means the OS credential store accepted it. ``"file"`` means
    the caller should fall back to the JSON config (this module doesn't
    touch that file itself — ``config_store.save_stored_config`` handles the
    write after inspecting the return value).
    """
    def _write() -> SecretLocation:
        backend = _load_keyring()
        if backend is None:
            return "file"
        try:
            backend.set_password(SERVICE_NAME, name, value)
        except Exception:  # noqa: BLE001 — any backend error means we fall back to file
            return "file"
        return "keyring"

    return _run_guarded("save", _write, "file")


def load_secret(name: str) -> str | None:
    """Read a secret from the OS credential store. Returns None when absent.

    Returns None rather than blocking when the credential store is unresponsive
    — see the module docstring. Callers already treat None as "not stored here"
    and fall through to the file config, so a stalled keychain degrades to the
    same path as a missing one.
    """

    def _read() -> str | None:
        backend = _load_keyring()
        if backend is None:
            return None
        try:
            return backend.get_password(SERVICE_NAME, name)
        except Exception:  # noqa: BLE001 — malformed keychain entries shouldn't crash startup
            return None

    return _run_guarded("load", _read, None)


def delete_secret(name: str) -> bool:
    """Remove a secret from the OS credential store.

    Returns True when a delete was performed, False when no keyring is
    available. A missing entry also returns False — callers that want
    belt-and-suspenders deletion should also wipe the JSON fallback.
    """
    def _delete() -> bool:
        backend = _load_keyring()
        if backend is None:
            return False
        try:
            backend.delete_password(SERVICE_NAME, name)
            return True
        except Exception:  # noqa: BLE001 — entry missing or backend quirk; treat as no-op
            return False

    return _run_guarded("delete", _delete, False)


def keyring_available() -> bool:
    """True when a usable OS credential backend is loaded and responsive."""
    return _run_guarded("available", lambda: _load_keyring() is not None, False)


def _load_keyring():
    """Import ``keyring`` lazily so the brain starts even if it's uninstalled."""
    if os.getenv("JUNE_DISABLE_KEYRING", "").lower() in ("1", "true", "yes"):
        return None
    try:
        import keyring
        from keyring.backends import fail
    except ImportError:
        return None

    backend = keyring.get_keyring()
    if isinstance(backend, fail.Keyring):
        return None
    return backend
