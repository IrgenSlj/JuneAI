"""Tests for /skills/registry routes (Sprint 1.6)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from june_api.app import create_app


@pytest.fixture(autouse=True)
def isolated_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test against a fresh manifest in tmp_path."""
    monkeypatch.setenv("JUNE_CONFIG_ROOT", str(tmp_path))


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # The skills route calls brain_graph.reload_agent() on every mutation; stub
    # it out so the test does not have to bring up the full LangGraph agent.
    from june_api.routes import skills as skills_route

    monkeypatch.setattr(skills_route.brain_graph, "reload_agent", lambda: None)
    return TestClient(create_app())


def test_list_registry_includes_curated_entries(client: TestClient) -> None:
    res = client.get("/skills/registry")
    assert res.status_code == 200
    body = res.json()
    assert body["count"] >= 3
    keys = {e["key"] for e in body["entries"]}
    assert "filesystem-mcp" in keys
    # Every entry has a usable install block.
    for entry in body["entries"]:
        assert entry["install"]["command"]
        assert entry["model_policy"] in {"local_only", "cloud_allowed", "cloud_required"}
        assert isinstance(entry["installed"], bool)


def test_install_then_marked_installed(client: TestClient) -> None:
    before = client.get("/skills/registry").json()
    pick = next(e for e in before["entries"] if not e["installed"])

    install = client.post(f"/skills/registry/{pick['key']}/install")
    assert install.status_code == 200
    assert install.json()["installed"] is True

    after = client.get("/skills/registry").json()
    found = next(e for e in after["entries"] if e["key"] == pick["key"])
    assert found["installed"] is True


def test_install_rejects_unknown_key(client: TestClient) -> None:
    res = client.post("/skills/registry/never-real/install")
    assert res.status_code == 404


def test_install_rejects_duplicate(client: TestClient) -> None:
    pick = client.get("/skills/registry").json()["entries"][0]
    assert client.post(f"/skills/registry/{pick['key']}/install").status_code == 200
    res = client.post(f"/skills/registry/{pick['key']}/install")
    assert res.status_code == 409


def test_uninstall_removes_entry(client: TestClient) -> None:
    pick = client.get("/skills/registry").json()["entries"][0]
    client.post(f"/skills/registry/{pick['key']}/install")
    res = client.delete(f"/skills/registry/{pick['key']}")
    assert res.status_code == 200
    assert res.json() == {"key": pick["key"], "uninstalled": True}
    # Second delete is a 404.
    assert client.delete(f"/skills/registry/{pick['key']}").status_code == 404


def test_install_response_carries_required_env(client: TestClient) -> None:
    body = client.get("/skills/registry").json()
    pick = next(
        (e for e in body["entries"] if e["install"]["env_required"]),
        None,
    )
    if pick is None:
        pytest.skip("no curated entries declare env_required")
    res = client.post(f"/skills/registry/{pick['key']}/install")
    assert res.status_code == 200
    assert res.json()["requires_env"] == pick["install"]["env_required"]
