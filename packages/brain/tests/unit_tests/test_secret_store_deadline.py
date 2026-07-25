"""The credential store must never be able to stall a turn.

Regression cover for the packaged-app hang: an ad-hoc-signed sidecar reading a
keychain item created by a different build blocks forever inside
``SecItemCopyMatching``, which froze every chat turn at the Trust Ledger's
signing-key load. These tests pin the deadline + latch behaviour that makes a
blocked credential store degrade instead of hang.
"""

from __future__ import annotations

import threading
import time

import june_brain.secret_store as secret_store
import pytest


@pytest.fixture(autouse=True)
def _clear_latch():
    secret_store._reset_unresponsive_for_tests()
    yield
    secret_store._reset_unresponsive_for_tests()


class _BlockingBackend:
    """Stands in for a keychain waiting on an authorization prompt."""

    def __init__(self, release: threading.Event) -> None:
        self.release = release
        self.calls = 0

    def get_password(self, service: str, name: str) -> str | None:
        self.calls += 1
        self.release.wait(30)
        return "never-arrives"

    def set_password(self, service: str, name: str, value: str) -> None:
        self.calls += 1
        self.release.wait(30)

    def delete_password(self, service: str, name: str) -> None:
        self.calls += 1
        self.release.wait(30)


def test_load_secret_returns_none_when_backend_blocks(monkeypatch) -> None:
    release = threading.Event()
    backend = _BlockingBackend(release)
    monkeypatch.setattr(secret_store, "_load_keyring", lambda: backend)
    monkeypatch.setenv("JUNE_KEYRING_TIMEOUT_S", "0.1")

    started = time.monotonic()
    try:
        assert secret_store.load_secret("gemini") is None
        # The deadline, not the backend, decides when we return.
        assert time.monotonic() - started < 5
        assert secret_store.keyring_unresponsive() is True
    finally:
        release.set()


def test_latch_stops_further_calls_reaching_the_backend(monkeypatch) -> None:
    release = threading.Event()
    backend = _BlockingBackend(release)
    monkeypatch.setattr(secret_store, "_load_keyring", lambda: backend)
    monkeypatch.setenv("JUNE_KEYRING_TIMEOUT_S", "0.1")

    try:
        secret_store.load_secret("gemini")
        assert backend.calls == 1

        # Every later call short-circuits: one parked daemon thread per process
        # is acceptable, one per call is not.
        for _ in range(5):
            assert secret_store.load_secret("gemini") is None
        assert secret_store.save_secret("gemini", "x") == "file"
        assert secret_store.delete_secret("gemini") is False
        assert secret_store.keyring_available() is False
        assert backend.calls == 1
    finally:
        release.set()


def test_save_secret_falls_back_to_file_when_backend_blocks(monkeypatch) -> None:
    release = threading.Event()
    backend = _BlockingBackend(release)
    monkeypatch.setattr(secret_store, "_load_keyring", lambda: backend)
    monkeypatch.setenv("JUNE_KEYRING_TIMEOUT_S", "0.1")

    try:
        assert secret_store.save_secret("gemini", "sk-test") == "file"
        assert secret_store.keyring_unresponsive() is True
    finally:
        release.set()


class _FastBackend:
    def __init__(self) -> None:
        self.stored: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, name: str) -> str | None:
        return self.stored.get((service, name))

    def set_password(self, service: str, name: str, value: str) -> None:
        self.stored[(service, name)] = value

    def delete_password(self, service: str, name: str) -> None:
        del self.stored[(service, name)]


def test_healthy_backend_round_trips_unchanged(monkeypatch) -> None:
    backend = _FastBackend()
    monkeypatch.setattr(secret_store, "_load_keyring", lambda: backend)

    assert secret_store.save_secret("gemini", "sk-live") == "keyring"
    assert secret_store.load_secret("gemini") == "sk-live"
    assert secret_store.keyring_available() is True
    assert secret_store.delete_secret("gemini") is True
    assert secret_store.load_secret("gemini") is None
    assert secret_store.keyring_unresponsive() is False


def test_missing_backend_still_reports_file_storage(monkeypatch) -> None:
    monkeypatch.setattr(secret_store, "_load_keyring", lambda: None)

    assert secret_store.save_secret("gemini", "sk-live") == "file"
    assert secret_store.load_secret("gemini") is None
    assert secret_store.keyring_available() is False
    # No backend is a normal branch, not a degraded one.
    assert secret_store.keyring_unresponsive() is False


def test_invalid_timeout_env_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("JUNE_KEYRING_TIMEOUT_S", "not-a-number")
    assert secret_store._timeout_s() == secret_store.DEFAULT_TIMEOUT_S

    monkeypatch.setenv("JUNE_KEYRING_TIMEOUT_S", "0")
    assert secret_store._timeout_s() == secret_store.DEFAULT_TIMEOUT_S
