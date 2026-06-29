"""Tests for the Silence Model API (ADR 0023, slice 7)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from june_api.app import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Fresh app + isolated SQLite so surfacing decisions land in a throwaway db."""
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


def _record_batch(candidate_id: str = "c1") -> str:
    from june_brain.silence import SurfacingCandidate, SurfacingContext, decide, get_store

    cand = SurfacingCandidate(candidate_id=candidate_id, kind="promise_nudge", salience=0.4)
    ctx = SurfacingContext(now="2026-06-29T12:00:00+00:00", presence_state="absent")
    return get_store().record(cand, decide(cand, ctx)).id


def test_surfacing_page_empty(client: TestClient) -> None:
    body = client.get("/system/surfacing").json()
    assert body["entries"] == []
    assert body["count"] == 0
    assert body["next_cursor"] is None


def test_surfacing_page_lists_decisions_with_reasons(client: TestClient) -> None:
    _record_batch("c1")
    body = client.get("/system/surfacing").json()
    assert body["count"] == 1
    entry = body["entries"][0]
    assert entry["action"] == "batch"
    assert "away" in entry["reason"]  # truthful reason surfaced
    assert entry["features"]["presence_state"] == "absent"


def test_feedback_bad_on_surfacing_marks_dismissed(client: TestClient) -> None:
    decision_id = _record_batch("c1")
    res = client.post(f"/system/surfacing/{decision_id}/feedback", json={"verdict": "bad"})
    assert res.status_code == 200
    body = res.json()
    assert body["verdict"] == "bad"
    # "bad" on a surfacing → the user didn't want it → dismissed.
    assert body["outcome"] == "dismissed"


def test_feedback_good_on_surfacing_marks_engaged(client: TestClient) -> None:
    decision_id = _record_batch("c1")
    body = client.post(f"/system/surfacing/{decision_id}/feedback", json={"verdict": "good"}).json()
    assert body["outcome"] == "engaged"


def test_feedback_influences_dismissals_for_similar(client: TestClient) -> None:
    from june_brain.silence import get_store

    ids = [_record_batch(f"c{i}") for i in range(2)]
    assert get_store().dismissals_for_similar("promise_nudge") == 0
    for decision_id in ids:
        client.post(f"/system/surfacing/{decision_id}/feedback", json={"verdict": "bad"})
    assert get_store().dismissals_for_similar("promise_nudge") == 2


def test_feedback_unknown_id_returns_404(client: TestClient) -> None:
    res = client.post("/system/surfacing/nope/feedback", json={"verdict": "good"})
    assert res.status_code == 404


def test_feedback_rejects_invalid_verdict(client: TestClient) -> None:
    decision_id = _record_batch("c1")
    res = client.post(f"/system/surfacing/{decision_id}/feedback", json={"verdict": "meh"})
    assert res.status_code == 422  # schema-validated literal


def test_drain_surfacing_returns_batch_items_and_empties(client: TestClient) -> None:
    """POST /system/surfacing/drain returns held items; a second call returns empty."""
    _record_batch("drain1")

    res = client.post("/system/surfacing/drain")
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 1
    assert len(body["drained"]) == 1
    assert body["drained"][0]["action"] == "batch"

    # Second drain must be empty — everything was marked surfaced.
    res2 = client.post("/system/surfacing/drain")
    body2 = res2.json()
    assert body2["count"] == 0
    assert body2["drained"] == []


def test_drain_surfacing_empty_when_nothing_held(client: TestClient) -> None:
    """POST /system/surfacing/drain returns empty when there are no held items."""
    res = client.post("/system/surfacing/drain")
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 0
    assert body["drained"] == []
