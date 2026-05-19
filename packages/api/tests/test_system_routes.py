"""Tests for /system and /system/activity routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from june_api.app import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Fresh app + isolated SQLite for the activity log."""
    import june_brain.activity as activity_pkg
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    activity_pkg.reset_for_tests()
    return TestClient(create_app())


def test_system_status_returns_runtime_view(client: TestClient) -> None:
    res = client.get("/system")
    assert res.status_code == 200
    body = res.json()
    # The shape must always include these fields; values vary by env.
    for key in (
        "provider",
        "label",
        "model",
        "mode",
        "privacy_label",
        "base_url",
        "ollama_reachable",
        "ollama_has_model",
        "api_key_present",
    ):
        assert key in body


def test_activity_list_starts_empty(client: TestClient) -> None:
    res = client.get("/system/activity")
    assert res.status_code == 200
    assert res.json() == {"entries": [], "count": 0}


def test_activity_records_real_requests(client: TestClient) -> None:
    """A non-skipped request lands as an activity entry via the middleware.

    The middleware fires the write off the response path, so we make a
    second request that itself surfaces the prior one — the read happens
    after the write thread has settled.
    """
    # Hitting /system is not on the skip list, so it should be logged.
    client.get("/system")
    # Give the background asyncio.to_thread write a moment to flush.
    import time

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        body = client.get("/system/activity").json()
        if body["count"] > 0:
            break
        time.sleep(0.05)
    else:  # pragma: no cover - only fires on a real regression
        pytest.fail("activity log did not record a /system hit within 2s")

    labels = [e["label"] for e in body["entries"]]
    assert any(label.startswith("GET /system") for label in labels)


def test_activity_filter_by_kind(client: TestClient) -> None:
    from june_brain.activity import ActivityLog

    log = ActivityLog()
    log.record(kind="tool", label="health.log_body_metric")
    log.record(kind="request", label="GET /memory/alice")

    tools = client.get("/system/activity?kind=tool").json()
    assert tools["count"] == 1
    assert tools["entries"][0]["label"] == "health.log_body_metric"

    requests = client.get("/system/activity?kind=request").json()
    # Could be one or more depending on whether the activity-log
    # middleware caught its own /system/activity?kind=tool read, but
    # /system/activity is on the skip list, so it should be exactly one.
    labels = [e["label"] for e in requests["entries"]]
    assert "GET /memory/alice" in labels


def test_activity_clear_empties_the_table(client: TestClient) -> None:
    from june_brain.activity import ActivityLog

    ActivityLog().record(kind="tool", label="x")
    pre = client.get("/system/activity").json()
    assert pre["count"] >= 1

    res = client.delete("/system/activity")
    assert res.status_code == 200
    assert res.json() == {"entries": [], "count": 0}

    after = client.get("/system/activity").json()
    assert after["count"] == 0


def test_activity_limit_is_clamped(client: TestClient) -> None:
    """limit out of bounds is clamped silently, not 400'd."""
    res = client.get("/system/activity?limit=0")
    assert res.status_code == 200
    res = client.get("/system/activity?limit=99999")
    assert res.status_code == 200
