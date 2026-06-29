"""Tests for GET /home/{user_id}/holdings (handoff S5, slice 8)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from june_api.app import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    import june_brain.activity as activity_pkg
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite
    import june_brain.silence as silence_pkg
    import june_brain.trust as trust_pkg

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    activity_pkg.reset_for_tests()
    trust_pkg.reset_for_tests()
    silence_pkg.reset_for_tests()
    return TestClient(create_app())


def test_empty_holdings(client: TestClient) -> None:
    body = client.get("/home/alice/holdings").json()
    assert body["open_promises"] == 0
    assert body["waiting_on_user"] == 0
    assert body["blocked_by_local_only"] == 0
    assert body["next_deadline"] is None
    assert body["held_digest"] == []
    assert body["held_count"] == 0
    assert body["egress_today"] == 0
    assert body["chain_verified"] is None


def test_holdings_counts_promises_and_blocked(client: TestClient) -> None:
    from june_brain.tasks.models import TaskStatus
    from june_brain.tasks.store import TasksStore

    store = TasksStore(user_id="alice")
    running = store.create(goal="ship the thing")
    store.set_status(running.id, TaskStatus.RUNNING)
    waiting = store.create(goal="needs a reply")
    store.set_status(waiting.id, TaskStatus.AWAITING_USER)
    blocked = store.create(goal="wants the network")
    store.set_blocked(
        blocked.id,
        reason="a networked tool is blocked by local-only mode",
        next_action="switch the dial, then retry",
        kind="local_only",
    )

    body = client.get("/home/alice/holdings").json()
    assert body["open_promises"] == 3
    assert body["waiting_on_user"] == 2
    assert body["blocked_by_local_only"] == 1


def test_holdings_reports_soonest_deadline(client: TestClient) -> None:
    from june_brain.tasks.store import TasksStore

    store = TasksStore(user_id="alice")
    a = store.create(goal="later")
    store.set_due(a.id, "2026-07-10T09:00:00+00:00")
    b = store.create(goal="sooner")
    store.set_due(b.id, "2026-07-01T09:00:00+00:00")

    body = client.get("/home/alice/holdings").json()
    assert body["next_deadline"]["goal"] == "sooner"
    assert body["next_deadline"]["due_at"] == "2026-07-01T09:00:00+00:00"


def test_holdings_surfaces_held_digest(client: TestClient) -> None:
    from june_brain.silence import SurfacingCandidate, SurfacingContext, decide, get_store

    cand = SurfacingCandidate(candidate_id="c1", kind="promise_nudge", salience=0.3)
    ctx = SurfacingContext(now="2026-06-29T12:00:00+00:00", presence_state="absent")
    get_store().record(cand, decide(cand, ctx))  # -> batch, held

    body = client.get("/home/alice/holdings").json()
    assert body["held_count"] == 1
    assert len(body["held_digest"]) == 1
    assert body["held_digest"][0]["action"] == "batch"
    assert "away" in body["held_digest"][0]["reason"]


def test_holdings_surfaces_needs_now(client: TestClient) -> None:
    """Holdings response includes needs_now entries for unresolved 'now' decisions."""
    from june_brain.tasks.store import TasksStore

    store = TasksStore(user_id="alice")
    task = store.create(goal="overdue report")
    # Set a past due_at so the silence producer classifies it as 'now'.
    store.set_due(task.id, "2026-06-01T09:00:00+00:00")

    from june_brain.silence.producers import run_silence_producers

    run_silence_producers("alice", now_iso="2026-06-29T12:00:00+00:00")

    body = client.get("/home/alice/holdings").json()
    assert "needs_now" in body
    assert len(body["needs_now"]) >= 1
    assert body["needs_now"][0]["action"] == "now"


def test_holdings_reports_egress_and_chain(client: TestClient) -> None:
    # One cloud egress + a verification; the home reflects both from the ledger.
    from june_brain.providers.provenance import CloudCallEvent, record_cloud_call

    record_cloud_call(CloudCallEvent(model_id="gemini-x", phase="start", payload_summary="s"))
    client.post("/system/ledger/verify")

    body = client.get("/home/alice/holdings").json()
    assert body["egress_today"] == 1
    assert body["chain_verified"] is True
