"""Tests for TrustedHostMiddleware — the DNS-rebinding defense."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from june_api.app import create_app


def test_default_client_healthz_ok() -> None:
    """Default TestClient sends Host: testserver — must be allowed."""
    client = TestClient(create_app())
    response = client.get("/healthz")
    assert response.status_code == 200


def test_evil_host_rejected() -> None:
    """A Host header from an attacker domain is rejected with 400."""
    client = TestClient(create_app())
    response = client.get("/healthz", headers={"host": "evil.example"})
    assert response.status_code == 400


def test_loopback_hosts_allowed() -> None:
    """127.0.0.1 and localhost are in the default allowlist."""
    client = TestClient(create_app())
    assert client.get("/healthz", headers={"host": "127.0.0.1"}).status_code == 200
    assert client.get("/healthz", headers={"host": "localhost"}).status_code == 200


def test_env_override_changes_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    """JUNE_API_ALLOWED_HOSTS replaces the default allowlist entirely."""
    monkeypatch.setenv("JUNE_API_ALLOWED_HOSTS", "example.com")
    client = TestClient(create_app())
    assert client.get("/healthz", headers={"host": "example.com"}).status_code == 200
    assert client.get("/healthz", headers={"host": "localhost"}).status_code == 400


def test_wildcard_disables_host_checking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting JUNE_API_ALLOWED_HOSTS=* disables host checking entirely."""
    monkeypatch.setenv("JUNE_API_ALLOWED_HOSTS", "*")
    client = TestClient(create_app())
    assert client.get("/healthz", headers={"host": "evil.example"}).status_code == 200
