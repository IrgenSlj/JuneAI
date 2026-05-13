"""Regression tests for settings runtime invalidation."""

from __future__ import annotations

import os

from june_api.routes import settings as settings_route


def test_forget_key_scrubs_env_and_invalidates_agent(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    monkeypatch.setattr(settings_route, "forget_gemini_key", lambda: "file")
    monkeypatch.setattr(
        settings_route.brain_graph,
        "invalidate_agent",
        lambda: calls.append("invalidate"),
    )

    response = settings_route.forget_key()

    assert response.cleared_from == "file"
    assert "GEMINI_API_KEY" not in os.environ
    assert calls == ["invalidate"]
