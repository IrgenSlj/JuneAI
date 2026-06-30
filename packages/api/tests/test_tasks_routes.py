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
    run the real loop inside the test process. Runtime behaviour has its own
    test module under packages/brain.
    """
    import june_api.routes.tasks as tasks_route
    import june_brain.activity as activity_pkg
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    monkeypatch.setattr(tasks_route, "execute_task_in_background", lambda *a, **k: None)
    activity_pkg.reset_for_tests()
    import june_brain.trust as trust_pkg

    trust_pkg.reset_for_tests()
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


def test_task_view_includes_promise_metadata(client: TestClient) -> None:
    from june_brain.tasks import TasksStore

    created = client.post("/tasks/alice", json={"goal": "search the web"}).json()
    store = TasksStore(user_id="alice")
    store.set_blocked(
        created["id"],
        reason="web_search is blocked by local-only mode.",
        next_action="Switch privacy mode, then retry.",
        final_deliverable="I need approval before searching.",
    )

    body = client.get(f"/tasks/alice/{created['id']}").json()
    assert body["status"] == "awaiting_user"
    assert body["blocked_reason"] == "web_search is blocked by local-only mode."
    assert body["next_action"] == "Switch privacy mode, then retry."
    assert body["final_deliverable"] == "I need approval before searching."


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


def test_create_with_due_at_and_patch_clears_it(client: TestClient) -> None:
    created = client.post(
        "/tasks/alice",
        json={"goal": "file taxes", "due_at": "2026-07-01T17:00:00Z"},
    ).json()
    assert created["due_at"] == "2026-07-01T17:00:00Z"

    # Reschedule via patch.
    patched = client.patch(
        f"/tasks/alice/{created['id']}", json={"due_at": "2026-07-05T09:00:00Z"}
    ).json()
    assert patched["due_at"] == "2026-07-05T09:00:00Z"

    # Empty string clears the deadline; null would leave it unchanged.
    cleared = client.patch(f"/tasks/alice/{created['id']}", json={"due_at": ""}).json()
    assert cleared["due_at"] is None


def test_patch_due_at_null_leaves_deadline_unchanged(client: TestClient) -> None:
    created = client.post(
        "/tasks/alice", json={"goal": "x", "due_at": "2026-08-01T00:00:00Z"}
    ).json()
    # A status-only patch must not wipe the deadline.
    patched = client.patch(f"/tasks/alice/{created['id']}", json={"status": "paused"}).json()
    assert patched["due_at"] == "2026-08-01T00:00:00Z"


def test_post_approve_adds_tool_and_reruns(client: TestClient) -> None:
    created = client.post("/tasks/alice", json={"goal": "text my friend"}).json()
    res = client.post(
        f"/tasks/alice/{created['id']}/approve",
        json={"tool": "send_telegram_message"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "running"
    assert body["approved_tools"] == ["send_telegram_message"]
    assert body["started_at"] is not None


def test_post_approve_records_approval_activity(client: TestClient) -> None:
    created = client.post("/tasks/alice", json={"goal": "text my friend"}).json()
    client.post(
        f"/tasks/alice/{created['id']}/approve",
        json={"tool": "send_telegram_message"},
    )
    approvals = client.get("/system/activity?kind=approval").json()["entries"]
    assert len(approvals) == 1
    assert approvals[0]["label"] == "approved · send_telegram_message"
    assert approvals[0]["detail"]["decision"] == "approved"
    assert approvals[0]["detail"]["task_id"] == created["id"]
    # The grant is also a tamper-evident Trust Ledger entry (ADR 0022), with
    # actor="user" because it records the human's decision.
    from june_brain.trust import LedgerReader

    ledger_approvals = [e for e in LedgerReader().page(limit=50) if e.kind == "approval"]
    assert len(ledger_approvals) == 1
    assert ledger_approvals[0].actor == "user"
    assert ledger_approvals[0].payload["tool"] == "send_telegram_message"
    assert LedgerReader().verify_chain().ok is True


def test_post_approve_rejects_empty_tool(client: TestClient) -> None:
    created = client.post("/tasks/alice", json={"goal": "x"}).json()
    res = client.post(f"/tasks/alice/{created['id']}/approve", json={"tool": ""})
    assert res.status_code == 422


def test_post_approve_returns_404_for_unknown(client: TestClient) -> None:
    res = client.post("/tasks/alice/missing/approve", json={"tool": "web_search"})
    assert res.status_code == 404


def test_patch_returns_404_for_unknown(client: TestClient) -> None:
    res = client.patch("/tasks/alice/missing", json={"status": "running"})
    assert res.status_code == 404


def test_patch_rejects_double_start(client: TestClient) -> None:
    """PATCH status=running must 409 when the task is already running.

    Otherwise a double-click would queue two background runtimes against
    the same row.
    """
    created = client.post("/tasks/alice", json={"goal": "x"}).json()
    tid = created["id"]
    first = client.patch(f"/tasks/alice/{tid}", json={"status": "running"})
    assert first.status_code == 200
    second = client.patch(f"/tasks/alice/{tid}", json={"status": "running"})
    assert second.status_code == 409


def test_post_run_idempotent_for_running_task(client: TestClient) -> None:
    """POST /run on an already-running task returns the current view as-is."""
    created = client.post("/tasks/alice", json={"goal": "x"}).json()
    client.patch(f"/tasks/alice/{created['id']}", json={"status": "running"})
    res = client.post(f"/tasks/alice/{created['id']}/run")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "running"


def test_users_are_scoped(client: TestClient) -> None:
    a = client.post("/tasks/alice", json={"goal": "alice task"}).json()
    client.post("/tasks/bob", json={"goal": "bob task"})

    bob_tasks = client.get("/tasks/bob").json()
    assert bob_tasks["count"] == 1
    assert bob_tasks["tasks"][0]["goal"] == "bob task"

    # Bob cannot see Alice's task directly.
    assert client.get(f"/tasks/bob/{a['id']}").status_code == 404


# ---------------------------------------------------------------------------
# SSE event-stream route tests
# ---------------------------------------------------------------------------


def test_task_events_404_for_unknown(client: TestClient) -> None:
    """GET /tasks/{user_id}/{task_id}/events on a non-existent task returns 404."""
    res = client.get("/tasks/alice/no-such-task/events")
    assert res.status_code == 404


def test_task_events_content_type(client: TestClient) -> None:
    """GET /tasks/{user_id}/{task_id}/events returns text/event-stream content-type.

    The task is completed first so the poll generator emits status + done and
    returns promptly; otherwise a non-terminal task would keep the stream open
    for the full 30-minute ceiling and the TestClient read would block.
    """
    created = client.post("/tasks/alice", json={"goal": "watch me"}).json()
    tid = created["id"]
    client.patch(f"/tasks/alice/{tid}", json={"status": "completed"})
    with client.stream("GET", f"/tasks/alice/{tid}/events") as res:
        assert res.status_code == 200
        assert "text/event-stream" in res.headers.get("content-type", "")
        # Drain so the server-side generator runs to its `done` frame and
        # returns, rather than being left dangling at context exit.
        body = "".join(res.iter_text())
    assert "done" in body


def test_task_events_yields_initial_status_and_done(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The poll generator emits at least a status frame and a done frame for a terminal task.

    Strategy: create a task, immediately mark it completed so the first poll
    sees a terminal status, then read the SSE stream until we hit 'done'.
    We unit-test the async generator directly to avoid asyncio complications
    with TestClient's sync wrapper.
    """
    import asyncio
    import json as _json

    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite
    from june_api.routes.tasks import _poll_task_events
    from june_brain.tasks import TasksStore, TaskStatus

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())

    store = TasksStore(user_id="eve")
    task = store.create(goal="unit poll test")
    store.set_status(task.id, TaskStatus.COMPLETED)

    async def collect() -> list[dict]:
        frames = []
        async for raw in _poll_task_events(store, task.id):
            # raw is "data: {...}\n\n" — strip prefix
            payload = raw.removeprefix("data: ").strip()
            frames.append(_json.loads(payload))
            if frames[-1]["type"] == "done":
                break
        return frames

    frames = asyncio.run(collect())
    types = [f["type"] for f in frames]
    assert "status" in types
    assert "done" in types
    # Ensure the status frame carries the right value.
    status_frames = [f for f in frames if f["type"] == "status"]
    assert any(f["status"] == "completed" for f in status_frames)


def test_task_view_includes_attempts_field(client: TestClient) -> None:
    """TaskView must expose the attempts counter (starts at 0 for a new promise)."""
    created = client.post("/tasks/alice", json={"goal": "retry test"}).json()
    assert "attempts" in created
    assert created["attempts"] == 0

    # Fetch via GET too.
    fetched = client.get(f"/tasks/alice/{created['id']}").json()
    assert "attempts" in fetched
    assert fetched["attempts"] == 0


def test_task_events_emits_step_frame(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The poll generator emits a step frame when a step exists at poll time."""
    import asyncio
    import json as _json

    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite
    from june_api.routes.tasks import _poll_task_events
    from june_brain.tasks import TasksStore, TaskStatus, TaskStep, TaskStepStatus

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())

    store = TasksStore(user_id="frank")
    task = store.create(goal="step emission test")
    # Append a step before moving to terminal so the first poll sees it.
    step = TaskStep(description="do something", status=TaskStepStatus.COMPLETED)
    store.append_step(task.id, step)
    store.set_status(task.id, TaskStatus.COMPLETED)

    async def collect() -> list[dict]:
        frames = []
        async for raw in _poll_task_events(store, task.id):
            payload = raw.removeprefix("data: ").strip()
            frames.append(_json.loads(payload))
            if frames[-1]["type"] == "done":
                break
        return frames

    frames = asyncio.run(collect())
    step_frames = [f for f in frames if f["type"] == "step"]
    assert len(step_frames) >= 1
    assert step_frames[0]["step"]["description"] == "do something"
