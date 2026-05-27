"""Tests for the /capture routes (P3)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from june_api.app import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    return TestClient(create_app())


def test_post_capture_classifies_and_returns_candidates(client: TestClient) -> None:
    res = client.post("/capture/alice", json={"text": "Tomorrow call Sam and finish the deck"})
    assert res.status_code == 201
    body = res.json()
    assert "task" in body["kinds"]
    assert body["candidates"]
    assert body["id"]


def test_get_recent_captures_returns_saved(client: TestClient) -> None:
    client.post("/capture/alice", json={"text": "I promised Lisa the file Friday"})
    res = client.get("/capture/alice/recent")
    assert res.status_code == 200
    items = res.json()["items"]
    assert items and "promise" in items[0]["kinds"]
