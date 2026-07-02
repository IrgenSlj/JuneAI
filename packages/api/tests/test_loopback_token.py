"""Tests for the opt-in loopback token middleware.

``JUNE_API_TOKEN`` unset → complete pass-through (existing behaviour unchanged).
``JUNE_API_TOKEN`` set   → ``X-June-Token`` header required on every non-exempt route.
``/healthz`` is always exempt regardless of token state.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from june_api.app import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Fresh app with isolated memory so each test starts clean."""
    import june_brain.activity as activity_pkg
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    activity_pkg.reset_for_tests()
    import june_brain.trust as trust_pkg

    trust_pkg.reset_for_tests()
    # Ensure the loopback token is off for every test unless explicitly set.
    monkeypatch.delenv("JUNE_API_TOKEN", raising=False)
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Token unset → pass-through (must not change any current behaviour)
# ---------------------------------------------------------------------------


def test_token_unset_passes_through_without_header(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("JUNE_API_TOKEN", raising=False)
    res = client.get("/system")
    assert res.status_code == 200


# ---------------------------------------------------------------------------
# Token set → enforcement
# ---------------------------------------------------------------------------


def test_token_set_rejects_missing_header(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JUNE_API_TOKEN", "supersecret")
    res = client.get("/system")
    assert res.status_code == 401
    assert "loopback token" in res.json()["detail"]


def test_token_set_rejects_wrong_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JUNE_API_TOKEN", "supersecret")
    res = client.get("/system", headers={"X-June-Token": "wrong"})
    assert res.status_code == 401
    assert "loopback token" in res.json()["detail"]


def test_token_set_accepts_correct_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JUNE_API_TOKEN", "supersecret")
    res = client.get("/system", headers={"X-June-Token": "supersecret"})
    assert res.status_code == 200


# ---------------------------------------------------------------------------
# /healthz is always exempt
# ---------------------------------------------------------------------------


def test_healthz_exempt_when_token_set_and_no_header(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JUNE_API_TOKEN", "supersecret")
    res = client.get("/healthz")
    assert res.status_code == 200
