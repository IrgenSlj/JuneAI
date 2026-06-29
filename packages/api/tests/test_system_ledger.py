"""Tests for the Trust Ledger API (ADR 0022, slice 4)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from june_api.app import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Fresh app + isolated SQLite so the ledger lives in a throwaway db."""
    import june_brain.activity as activity_pkg
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite
    import june_brain.trust as trust_pkg

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    activity_pkg.reset_for_tests()
    trust_pkg.reset_for_tests()
    return TestClient(create_app())


def _seed(n: int = 3) -> None:
    from june_brain.trust import LedgerWriter

    w = LedgerWriter()
    for i in range(n):
        w.append(kind="egress", actor="june", payload={"model_id": "gemini-x", "i": i})


def test_ledger_page_empty(client: TestClient) -> None:
    body = client.get("/system/ledger").json()
    assert body["entries"] == []
    assert body["count"] == 0
    assert body["next_cursor"] is None


def test_ledger_page_newest_first(client: TestClient) -> None:
    _seed(3)
    body = client.get("/system/ledger").json()
    assert body["count"] == 3
    seqs = [e["seq"] for e in body["entries"]]
    assert seqs == [3, 2, 1]
    assert body["entries"][0]["kind"] == "egress"
    assert body["entries"][0]["prev_hash"]
    assert body["entries"][0]["entry_hash"]


def test_ledger_pagination_cursor(client: TestClient) -> None:
    _seed(5)
    first = client.get("/system/ledger?limit=2").json()
    assert [e["seq"] for e in first["entries"]] == [5, 4]
    assert first["next_cursor"] == 4
    second = client.get(f"/system/ledger?limit=2&cursor={first['next_cursor']}").json()
    assert [e["seq"] for e in second["entries"]] == [3, 2]


def test_ledger_verify_ok(client: TestClient) -> None:
    _seed(3)
    body = client.post("/system/ledger/verify").json()
    assert body["ok"] is True
    assert body["first_broken_seq"] is None
    assert body["signed"] is False


def test_ledger_verify_detects_tamper(client: TestClient) -> None:
    _seed(4)
    # Mutate seq 2's payload directly in the db, then verify via the API.
    from june_brain.memory.sqlite import _get_connection, db_path

    conn = _get_connection(db_path())
    conn.execute("UPDATE trust_ledger SET payload=? WHERE seq=2", ('{"tampered":true}',))
    conn.commit()
    body = client.post("/system/ledger/verify").json()
    assert body["ok"] is False
    assert body["first_broken_seq"] == 2


def test_system_status_includes_ledger_summary(client: TestClient) -> None:
    _seed(2)
    # Verify once so chain_verified_at is populated.
    client.post("/system/ledger/verify")
    summary = client.get("/system").json()["ledger_summary"]
    assert summary is not None
    assert summary["count"] == 2
    assert summary["egress_today"] == 2
    assert summary["chain_verified"] is True
    assert summary["chain_verified_at"]
    assert summary["last_entry_ts"]
