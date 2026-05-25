"""Tests for the opt-in API key middleware.

Auth is off by default (local single-user on loopback). Setting
``JUNE_API_AUTH_ENABLED`` turns enforcement on; ``X-June-Api-Key`` is then
required on every non-exempt route.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from june_api.app import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Fresh app with isolated config + memory so the API key is sandboxed."""
    import june_brain.activity as activity_pkg
    import june_brain.config_store as config_store
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(config_store, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    activity_pkg.reset_for_tests()
    return TestClient(create_app())


def test_auth_off_by_default(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JUNE_API_AUTH_ENABLED", raising=False)
    res = client.get("/system")
    assert res.status_code == 200


def test_enabled_rejects_missing_key(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUNE_API_AUTH_ENABLED", "1")
    res = client.get("/system")
    assert res.status_code == 401
    assert "Missing API key" in res.json()["detail"]


def test_enabled_rejects_invalid_key(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUNE_API_AUTH_ENABLED", "1")
    res = client.get("/system", headers={"X-June-Api-Key": "not-the-key"})
    assert res.status_code == 401
    assert "Invalid API key" in res.json()["detail"]


def test_enabled_accepts_valid_key(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from june_brain.auth import get_api_key

    monkeypatch.setenv("JUNE_API_AUTH_ENABLED", "1")
    res = client.get("/system", headers={"X-June-Api-Key": get_api_key()})
    assert res.status_code == 200


def test_setup_status_exempt_when_enabled(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUNE_API_AUTH_ENABLED", "1")
    res = client.get("/setup/status")
    assert res.status_code == 200
