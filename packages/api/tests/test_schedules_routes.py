"""Regression tests for the /schedules CRUD route.

The route's ``_store()`` helper used to crash on ``MEMORY_DIR / "june.db"``
(str, not Path), and ``create_schedule`` constructed ``Schedule(...)`` without
the required ``id`` — so the endpoint 500'd on every call. It had no test, so
the suite stayed green. These lock the create/list/get path.
"""

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


def test_create_schedule_returns_201_and_backfills_id(client: TestClient) -> None:
    res = client.post(
        "/schedules/alice",
        json={"name": "Morning briefing", "cron_expression": "0 8 * * *"},
    )
    assert res.status_code == 201
    body = res.json()
    assert body["name"] == "Morning briefing"
    assert body["id"]  # store.create backfilled a generated id


def test_created_schedule_is_listable_and_gettable(client: TestClient) -> None:
    created = client.post("/schedules/alice", json={"name": "Evening review"}).json()

    listing = client.get("/schedules/alice")
    assert listing.status_code == 200
    assert any(s["id"] == created["id"] for s in listing.json())

    fetched = client.get(f"/schedules/alice/{created['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["name"] == "Evening review"
