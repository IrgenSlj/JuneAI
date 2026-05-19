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
    disabled_tools: list[str] = field(default_factory=list)


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
                "tools": [
                    {
                        "name": f"{s.entry.key}_tool",
                        "description": "demo",
                        "enabled": f"{s.entry.key}_tool" not in s.entry.disabled_tools,
                    }
                ],
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

    def set_tool_enabled(self, key: str, tool_name: str, enabled: bool):
        skill = self.skills.get(key)
        if skill is None:
            return None
        if enabled:
            skill.entry.disabled_tools = [
                t for t in skill.entry.disabled_tools if t != tool_name
            ]
        elif tool_name not in skill.entry.disabled_tools:
            skill.entry.disabled_tools = [*skill.entry.disabled_tools, tool_name]
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
    monkeypatch.setattr(skills_route, "set_skill_tool_enabled", fake.set_tool_enabled)

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
        {
            "name": "calendar_tool",
            "description": "demo",
            "enabled": True,
            "input_schema": {},
        }
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


def test_skill_writes_returns_only_that_skills_facts(client, tmp_path, monkeypatch):
    """The endpoint should filter the vector shadow table by source prefix."""
    import hashlib
    from chromadb.api.types import EmbeddingFunction

    from june_brain.memory import vector as vector_module
    from june_brain.memory.sqlite import Memory

    class _HashEmbedder(EmbeddingFunction):
        def __init__(self) -> None:
            super().__init__()

        def __call__(self, input):
            texts = [input] if isinstance(input, str) else list(input)
            vecs = []
            for t in texts:
                d = hashlib.sha256(t.encode("utf-8")).digest()
                vecs.append([(d[i % len(d)] / 255.0) * 2 - 1 for i in range(64)])
            return vecs[0] if isinstance(input, str) else vecs

        @staticmethod
        def name() -> str:
            return "test-hash-embedder"

        @staticmethod
        def build_from_config(_):
            return _HashEmbedder()

        def get_config(self):
            return {}

    monkeypatch.setattr("june_brain.memory.MEMORY_DIR", str(tmp_path))
    monkeypatch.setattr(
        vector_module, "_get_embedding_function", lambda: _HashEmbedder()
    )
    vector_module.reset_singletons()

    Memory("alex")  # ensure schema exists in tmp_path
    v = vector_module.VectorStore("alex", embedding_function=_HashEmbedder())
    v.upsert("daily fact 1", source="skill:daily:track_goal")
    v.upsert("daily fact 2", source="skill:daily:save_journal_entry")
    v.upsert("calendar fact", source="skill:calendar:save_calendar_item")
    v.upsert("extracted fact", source="extraction")

    res = client.get("/skills/calendar/writes/alex")
    assert res.status_code == 200
    payload = res.json()
    assert payload["skill"] == "calendar"
    assert payload["count"] == 1
    assert payload["writes"][0]["text"] == "calendar fact"
    assert payload["writes"][0]["tool"] == "save_calendar_item"
    assert payload["writes"][0]["source"] == "skill:calendar:save_calendar_item"
    assert payload["writes"][0]["ref"].startswith("semantic:")

    res = client.get("/skills/research/writes/alex")
    assert res.status_code == 200
    assert res.json()["count"] == 0  # no writes from research

    vector_module.reset_singletons()


def test_skill_writes_unknown_skill_returns_404(client):
    res = client.get("/skills/nonexistent/writes/alex")
    assert res.status_code == 404


def test_tool_toggle_flips_state_and_reloads_agent(client, supervisor):
    res = client.post(
        "/skills/calendar/tools/calendar_tool/toggle",
        json={"enabled": False},
    )
    assert res.status_code == 200
    assert res.json() == {"key": "calendar", "tool": "calendar_tool", "enabled": False}
    assert supervisor.reload_calls == 1

    # Listing should now show the tool as disabled.
    after = client.get("/skills").json()
    cal = next(s for s in after["skills"] if s["key"] == "calendar")
    tool = next(t for t in cal["tools"] if t["name"] == "calendar_tool")
    assert tool["enabled"] is False

    # Re-enable.
    res2 = client.post(
        "/skills/calendar/tools/calendar_tool/toggle",
        json={"enabled": True},
    )
    assert res2.json()["enabled"] is True


def test_tool_toggle_unknown_skill_returns_404(client):
    res = client.post(
        "/skills/nonexistent/tools/anything/toggle",
        json={"enabled": False},
    )
    assert res.status_code == 404
