"""Test the /greeting route wiring (greeting logic is covered in the brain)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from june_api.app import create_app


def test_greeting_route_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    import june_api.routes.greeting as gr

    monkeypatch.setattr(
        gr, "build_greeting",
        lambda uid, name="": {"greeting": f"Hi {name}", "has_context": False},
    )
    client = TestClient(create_app())

    res = client.get("/greeting/alice?name=Alex")
    assert res.status_code == 200
    assert res.json() == {"greeting": "Hi Alex", "has_context": False}
