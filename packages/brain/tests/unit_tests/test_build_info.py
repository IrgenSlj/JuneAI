"""Tests for the build_version() helper."""

from __future__ import annotations

import pytest
from june_brain.build_info import build_version


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    build_version.cache_clear()
    monkeypatch.setenv("JUNE_BUILD_SHA", "abc1234")
    build_version.cache_clear()
    assert build_version() == "abc1234"
    build_version.cache_clear()


def test_returns_nonempty_string_in_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    build_version.cache_clear()
    monkeypatch.delenv("JUNE_BUILD_SHA", raising=False)
    build_version.cache_clear()
    result = build_version()
    assert isinstance(result, str)
    assert len(result) > 0
    build_version.cache_clear()
