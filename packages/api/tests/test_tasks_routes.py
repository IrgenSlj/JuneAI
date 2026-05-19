"""Tests for /tasks CRUD routes (ADR 0010, Sprint 1.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from june_api.app import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A fresh app with a tmp data dir so tasks land in an isolated SQLite file.

    Also stubs the TaskRuntime trigger so PATCH status=running does not try to
    spin up the real LangGraph agent inside the test process. Runtime behaviour
    has its own test module under packages/brain.
    """
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite
    import june_api.routes.tasks as tasks_route

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    monkeypatch.setattr(tasks_route, "execute_task_in_background", lambda *a, **k: None)
    return TestClient(create_app())


def test_create_task_returns_201_and_view(client: TestClient) -> None:
    res = client.post("/tasks/alice", json={"goal": "Plan a weekend trip"})
    assert res.status_code == 201
    body = res.json()
    assert body["goal"] == "Plan a weekend trip"
    assert body["status"] == "planning"
    assert body["user_id"] == "alice"
    assert body["id"]
    assert body["plan"] == []


def test_create_task_rejects_empty_goal(client: TestClient) -> None:
    res = client.post("/tasks/alice", json={"goal": ""})
    assert res.status_code == 422


def test_list_tasks_empty_then_populated(client: TestClient) -> None:
    empty = client.get("/tasks/alice").json()
    assert empty == {"tasks": [], "count": 0}

    client.post("/tasks/alice", json={"goal": "one"})
    client.post("/tasks/alice", json={"goal": "two"})
    listing = client.get("/tasks/alice").json()
    assert listing["count"] == 2
    assert {t["goal"] for t in listing["tasks"]} == {"one", "two"}


def test_get_task_returns_404_for_unknown(client: TestClient) -> None:
    res = client.get("/tasks/alice/does-not-exist")
    assert res.status_code == 404


def test_patch_status_running_then_completed(client: TestClient) -> None:
    created = client.post("/tasks/alice", json={"goal": "x"}).json()
    tid = created["id"]

    running = client.patch(f"/tasks/alice/{tid}", json={"status": "running"}).json()
    assert running["status"] == "running"
    assert running["started_at"] is not None

    done = client.patch(f"/tasks/alice/{tid}", json={"status": "completed"}).json()
    assert done["status"] == "completed"
    assert done["finished_at"] is not None


def test_patch_status_failed_with_error(client: TestClient) -> None:
    created = client.post("/tasks/alice", json={"goal": "x"}).json()
    tid = created["id"]
    res = client.patch(f"/tasks/alice/{tid}", json={"status": "failed", "error": "nope"})
    body = res.json()
    assert body["status"] == "failed"
    assert body["error"] == "nope"


def test_patch_rejects_unknown_status(client: TestClient) -> None:
    created = client.post("/tasks/alice", json={"goal": "x"}).json()
    res = client.patch(f"/tasks/alice/{created['id']}", json={"status": "wibble"})
    assert res.status_code == 400


def test_list_filters_by_status(client: TestClient) -> None:
    a = client.post("/tasks/alice", json={"goal": "still planning"}).json()
    b = client.post("/tasks/alice", json={"goal": "now running"}).json()
    client.patch(f"/tasks/alice/{b['id']}", json={"status": "running"})

    running = client.get("/tasks/alice?status=running").json()
    assert running["count"] == 1
    assert running["tasks"][0]["id"] == b["id"]

    planning = client.get("/tasks/alice?status=planning").json()
    assert {t["id"] for t in planning["tasks"]} == {a["id"]}


def test_delete_removes_task(client: TestClient) -> None:
    created = client.post("/tasks/alice", json={"goal": "x"}).json()
    tid = created["id"]

    res = client.delete(f"/tasks/alice/{tid}")
    assert res.status_code == 200
    body = res.json()
    assert body == {"deleted": True, "task_id": tid}

    # Subsequent delete is a 404.
    res2 = client.delete(f"/tasks/alice/{tid}")
    assert res2.status_code == 404


def test_post_run_starts_planning_task(client: TestClient) -> None:
    created = client.post("/tasks/alice", json={"goal": "x"}).json()
    res = client.post(f"/tasks/alice/{created['id']}/run")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "running"
    assert body["started_at"] is not None


def test_post_run_idempotent_for_completed_task(client: TestClient) -> None:
    created = client.post("/tasks/alice", json={"goal": "x"}).json()
    client.patch(f"/tasks/alice/{created['id']}", json={"status": "completed"})
    res = client.post(f"/tasks/alice/{created['id']}/run")
    assert res.status_code == 200
    body = res.json()
    # Stays completed; not restarted.
    assert body["status"] == "completed"


def test_post_run_returns_404_for_unknown(client: TestClient) -> None:
    res = client.post("/tasks/alice/missing/run")
    assert res.status_code == 404


def test_users_are_scoped(client: TestClient) -> None:
    a = client.post("/tasks/alice", json={"goal": "alice task"}).json()
    client.post("/tasks/bob", json={"goal": "bob task"})

    bob_tasks = client.get("/tasks/bob").json()
    assert bob_tasks["count"] == 1
    assert bob_tasks["tasks"][0]["goal"] == "bob task"

    # Bob cannot see Alice's task directly.
    assert client.get(f"/tasks/bob/{a['id']}").status_code == 404
