"""Tests for /skills list + toggle routes.

These tests inject a fake supervisor so the FastAPI layer is exercised
without spawning real MCP subprocesses. The fake mirrors the subset of
the SkillSupervisor surface that routes/skills.py consumes:
``list_status`` and ``set_enabled``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from june_api.app import create_app
from june_brain.skills.supervisor import SkillStatus


@dataclass
class _FakeEntry:
    key: str
    enabled: bool = True


@dataclass
class _FakeSkill:
    entry: _FakeEntry
    status: SkillStatus = SkillStatus.RUNNING
    error: str = ""


@dataclass
class _FakeSupervisor:
    """Stand-in for SkillSupervisor that records toggles."""

    skills: dict[str, _FakeSkill] = field(default_factory=dict)
    reload_calls: int = 0

    def list_status(self) -> list[dict[str, Any]]:
        return [
            {
                "key": s.entry.key,
                "description": f"{s.entry.key} skill",
                "enabled": s.entry.enabled,
                "status": s.status.value,
                "error": s.error,
                "tools": [{"name": f"{s.entry.key}_tool", "description": "demo"}],
            }
            for s in self.skills.values()
        ]

    def set_enabled(self, key: str, enabled: bool):
        skill = self.skills.get(key)
        if skill is None:
            return None
        skill.entry.enabled = enabled
        skill.status = SkillStatus.RUNNING if enabled else SkillStatus.DISABLED
        return skill


@pytest.fixture
def supervisor(monkeypatch):
    """Install a fake supervisor and a no-op reload_agent hook."""
    fake = _FakeSupervisor(
        skills={
            "calendar": _FakeSkill(entry=_FakeEntry(key="calendar", enabled=True)),
            "research": _FakeSkill(entry=_FakeEntry(key="research", enabled=True)),
        }
    )

    from june_api.routes import skills as skills_route

    monkeypatch.setattr(skills_route, "list_status", fake.list_status)
    monkeypatch.setattr(skills_route, "set_skill_enabled", fake.set_enabled)

    # Swallow brain_graph.reload_agent() — it would rebuild the real agent.
    from june_brain import graph as brain_graph

    monkeypatch.setattr(brain_graph, "reload_agent", lambda: setattr(fake, "reload_calls", fake.reload_calls + 1))
    return fake


@pytest.fixture
def client(supervisor):
    return TestClient(create_app())


def test_list_skills_returns_full_snapshot(client):
    response = client.get("/skills")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    keys = [s["key"] for s in payload["skills"]]
    assert keys == ["calendar", "research"]
    assert payload["skills"][0]["tools"] == [
        {"name": "calendar_tool", "description": "demo"}
    ]


def test_toggle_unknown_skill_returns_404(client):
    response = client.post("/skills/nonexistent/toggle", json={"enabled": False})
    assert response.status_code == 404


def test_toggle_flips_state_and_reloads_agent(client, supervisor):
    response = client.post("/skills/research/toggle", json={"enabled": False})
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "key": "research",
        "enabled": False,
        "status": "disabled",
        "error": "",
    }
    # The route must rebuild the agent so the next chat turn sees the change.
    assert supervisor.reload_calls == 1

    # Subsequent listing reflects the flip.
    after = client.get("/skills").json()
    statuses = {s["key"]: s["enabled"] for s in after["skills"]}
    assert statuses == {"calendar": True, "research": False}
