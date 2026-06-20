"""Tests for _cors_origins() — the CORS allowlist resolver."""

from __future__ import annotations

import pytest
from june_api.app import _cors_origins


def test_default_does_not_allow_wildcard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JUNE_API_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("JUNE_API_DEV_OPEN_CORS", raising=False)
    result = _cors_origins()
    assert "*" not in result
    assert "http://localhost:5173" in result


def test_dev_open_cors_returns_wildcard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JUNE_API_CORS_ORIGINS", raising=False)
    monkeypatch.setenv("JUNE_API_DEV_OPEN_CORS", "1")
    result = _cors_origins()
    assert result == ["*"]


def test_explicit_origins_env_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUNE_API_CORS_ORIGINS", "https://a.test, https://b.test")
    monkeypatch.delenv("JUNE_API_DEV_OPEN_CORS", raising=False)
    result = _cors_origins()
    assert result == ["https://a.test", "https://b.test"]
