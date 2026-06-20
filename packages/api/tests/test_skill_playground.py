"""Tests for the /skills/{key}/tools/{tool}/invoke playground endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from june_api.app import create_app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A fresh app with the supervisor stubbed so no real subprocess is spawned."""
    from june_api.routes import skills as skills_route

    monkeypatch.setattr(skills_route, "reload_skills", lambda: None)

    # Pretend the supervisor knows about a single skill with one tool.
    monkeypatch.setattr(
        skills_route,
        "list_status",
        lambda: [
            {
                "key": "fake",
                "description": "test skill",
                "enabled": True,
                "status": "running",
                "error": "",
                "tools": [
                    {
                        "name": "echo",
                        "description": "echoes input back",
                        "enabled": True,
                        "input_schema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    }
                ],
            }
        ],
    )
    return TestClient(create_app())


def test_skills_response_exposes_input_schema(client: TestClient) -> None:
    res = client.get("/skills")
    assert res.status_code == 200
    skill = res.json()["skills"][0]
    tool = skill["tools"][0]
    assert tool["input_schema"]["properties"]["text"]["type"] == "string"


def test_invoke_returns_stringified_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from june_api.routes import skills as skills_route

    captured: dict[str, object] = {}

    def fake_call(skill_key: str, tool_name: str, arguments: dict) -> str:
        captured["skill_key"] = skill_key
        captured["tool_name"] = tool_name
        captured["arguments"] = arguments
        return "echoed hello"

    monkeypatch.setattr(skills_route, "call_skill_tool", fake_call)

    res = client.post(
        "/skills/fake/tools/echo/invoke",
        json={"arguments": {"text": "hello"}},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["result"] == "echoed hello"
    assert body["error"] == ""
    assert body["latency_ms"] >= 0
    assert captured == {
        "skill_key": "fake",
        "tool_name": "echo",
        "arguments": {"text": "hello"},
    }


def test_invoke_surface_supervisor_failure_message_as_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from june_api.routes import skills as skills_route

    monkeypatch.setattr(
        skills_route,
        "call_skill_tool",
        lambda *a, **k: "Skill 'fake' tool 'echo' failed: boom",
    )

    res = client.post(
        "/skills/fake/tools/echo/invoke",
        json={"arguments": {"text": "hi"}},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is False
    assert "boom" in body["error"]
    assert body["result"] == ""


def test_invoke_raises_404_for_unknown_skill(client: TestClient) -> None:
    res = client.post(
        "/skills/never-real/tools/echo/invoke",
        json={"arguments": {}},
    )
    assert res.status_code == 404


def test_invoke_catches_unexpected_exception(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from june_api.routes import skills as skills_route

    def boom(*args, **kwargs):
        raise RuntimeError("supervisor segfault")

    monkeypatch.setattr(skills_route, "call_skill_tool", boom)

    res = client.post("/skills/fake/tools/echo/invoke", json={"arguments": {}})
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is False
    assert "supervisor segfault" in body["error"]
