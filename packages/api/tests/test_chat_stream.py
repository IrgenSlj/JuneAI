"""Integration test for POST /chat.

There is one chat engine (ADR 0018): the hand-written loop. The test overrides
``get_harness_loop`` with a fake loop whose ``stream_turn`` yields
``StreamEvent`` objects — matching the real hand-written loop interface.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from june_api.app import create_app
from june_api.routes.chat import get_harness_loop
from june_brain.loop.interface import StreamEvent, TurnProvenance
from june_brain.memory import vector as vector_module

# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------

class _FakeLoop:
    """Replays a scripted sequence of StreamEvents via stream_turn."""

    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events

    async def stream_turn(
        self, session: Any, user_msg: Any
    ) -> AsyncIterator[StreamEvent]:
        for ev in self._events:
            yield ev


class _BrokenLoop:
    """A loop whose stream_turn raises after emitting one token."""

    async def stream_turn(
        self, session: Any, user_msg: Any
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type="token", content="partial ")
        raise RuntimeError("harness exploded")


def _parse_sse(body: str) -> list[dict]:
    events = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk.startswith("data:"):
            continue
        events.append(json.loads(chunk[len("data:"):].strip()))
    return events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def harness_client(tmp_path):
    """Build a TestClient whose /chat uses a caller-supplied fake harness loop."""
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)), patch(
        "june_api.routes.chat.MemoryManager.extract",
        return_value={"facts": 0, "entities": 0, "relations": 0},
    ):
        vector_module.reset_singletons()
        app = create_app()

        def _install(loop: Any) -> TestClient:
            app.dependency_overrides[get_harness_loop] = lambda: loop
            return TestClient(app)

        yield _install
        vector_module.reset_singletons()


# ---------------------------------------------------------------------------
# Chat path — the one hand-written engine
# ---------------------------------------------------------------------------

def _make_provenance(
    *,
    provider: str = "local",
    model: str = "mock-local",
    tier: str = "local-fast",
    latency_ms: int = 42,
    cloud_call: bool = False,
    cloud_payload_summary: str | None = None,
    memories_recalled: int = 0,
    skills_called: list[str] | None = None,
    rationale: str = "test",
) -> TurnProvenance:
    return TurnProvenance(
        tiers_used=[tier],
        cloud_call=cloud_call,
        model_ids=[model],
        memories_recalled=memories_recalled,
        skills_called=skills_called or [],
        rationale=rationale,
        latency_ms=latency_ms,
        cloud_payload_summary=cloud_payload_summary,
    )


def test_chat_harness_opt_in_emits_token_provenance_and_done(harness_client):
    prov = _make_provenance(
        provider="local",
        model="mock-local",
        tier="local-fast",
        latency_ms=42,
        cloud_call=False,
        memories_recalled=2,
        skills_called=["lookup"],
        rationale="Handled by test harness.",
    )
    loop = _FakeLoop([
        StreamEvent(type="token", content="Harness reply."),
        StreamEvent(type="provenance", provenance=prov),
        StreamEvent(type="done"),
    ])
    client = harness_client(loop)
    response = client.post("/chat", json={"user_id": "u", "message": "hi"})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert [event["type"] for event in events] == ["token", "provenance", "done"]
    assert events[0]["content"] == "Harness reply."

    provenance = events[1]["provenance"]
    assert provenance["provider"] == "local"
    assert provenance["model"] == "mock-local"
    assert provenance["tier"] == "local-fast"
    assert provenance["cloud_call"] is False
    assert provenance["memories_recalled"] == 2
    assert provenance["skills_called"] == ["lookup"]
    assert provenance["rationale"] == "Handled by test harness."


def test_chat_harness_opt_in_emits_error_on_failure(harness_client):
    loop = _BrokenLoop()
    client = harness_client(loop)
    response = client.post("/chat", json={"user_id": "u", "message": "hi"})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert [event["type"] for event in events] == ["token", "error"]
    assert "harness exploded" in events[-1]["content"]


def test_chat_streams_tokens_tool_calls_and_done(harness_client):
    prov = _make_provenance()
    loop = _FakeLoop([
        StreamEvent(
            type="recall",
            recall_hits=[
                {
                    "ref": "semantic:abc",
                    "text": "User loves ramen",
                    "source": "vector",
                    "kind": "fact",
                    "score": 0.2,
                }
            ],
        ),
        StreamEvent(type="token", content="Hello "),
        StreamEvent(type="token", content="Alex."),
        StreamEvent(type="tool_call", tool_name="log_mood", tool_args={"mood": "good"}),
        StreamEvent(type="tool_result", tool_name="log_mood", tool_result="logged"),
        StreamEvent(type="provenance", provenance=prov),
        StreamEvent(type="done"),
    ])
    client = harness_client(loop)
    response = client.post(
        "/chat",
        json={"user_id": "test-user", "message": "hi"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    types = [event["type"] for event in events]

    assert types == ["recall", "token", "token", "tool_call", "tool_result", "provenance", "done"]

    tokens = [event["content"] for event in events if event["type"] == "token"]
    assert "".join(tokens) == "Hello Alex."

    recall = next(event for event in events if event["type"] == "recall")
    assert len(recall["recall_hits"]) == 1
    assert recall["recall_hits"][0]["ref"] == "semantic:abc"
    assert recall["recall_hits"][0]["text"] == "User loves ramen"
    assert recall["recall_hits"][0]["source"] == "vector"

    tool_call = next(event for event in events if event["type"] == "tool_call")
    assert tool_call["tool_name"] == "log_mood"
    assert tool_call["tool_args"] == {"mood": "good"}

    tool_result = next(event for event in events if event["type"] == "tool_result")
    assert tool_result["tool_name"] == "log_mood"
    assert tool_result["tool_result"] == "logged"


def test_chat_forwards_local_only_tool_blocked(harness_client):
    """A Local-only block forwards as network=True without an approval flag."""
    loop = _FakeLoop([
        StreamEvent(
            type="tool_blocked",
            tool_name="web_search",
            tool_args={"query": "weather"},
            detail="needs the internet",
            network=True,
        ),
        StreamEvent(type="done"),
    ])
    client = harness_client(loop)
    response = client.post("/chat", json={"user_id": "u", "message": "search"})
    events = _parse_sse(response.text)
    blocked = next(e for e in events if e["type"] == "tool_blocked")
    assert blocked["tool_name"] == "web_search"
    assert blocked["network"] is True
    assert blocked["needs_approval"] is False
    assert blocked["action_class"] == ""


def test_chat_forwards_approval_needed_tool_blocked(harness_client):
    """A guard approval block forwards needs_approval and the action class."""
    loop = _FakeLoop([
        StreamEvent(
            type="tool_blocked",
            tool_name="send_telegram_message",
            tool_args={"text": "hi"},
            detail="Sends data off your device",
            network=False,
            needs_approval=True,
            action_class="write_network",
        ),
        StreamEvent(type="done"),
    ])
    client = harness_client(loop)
    response = client.post("/chat", json={"user_id": "u", "message": "text my friend"})
    events = _parse_sse(response.text)
    blocked = next(e for e in events if e["type"] == "tool_blocked")
    assert blocked["tool_name"] == "send_telegram_message"
    assert blocked["needs_approval"] is True
    assert blocked["action_class"] == "write_network"
    assert blocked["network"] is False
    assert blocked["detail"] == "Sends data off your device"


def test_chat_skips_empty_recall_hits(harness_client):
    """A recall event with no hits should not emit a frame to avoid empty UI noise."""
    loop = _FakeLoop([
        StreamEvent(type="recall", recall_hits=[]),
        StreamEvent(type="token", content="hi"),
        StreamEvent(type="done"),
    ])
    client = harness_client(loop)
    response = client.post("/chat", json={"user_id": "u", "message": "hi"})
    events = _parse_sse(response.text)
    types = [e["type"] for e in events]
    assert "recall" not in types
    assert types == ["token", "done"]


def test_chat_ignores_invalid_recall_hit_dicts(harness_client):
    """Non-dict entries in recall_hits are skipped defensively."""
    loop = _FakeLoop([
        StreamEvent(type="recall", recall_hits=["not-a-dict", None]),
        StreamEvent(type="token", content="hi"),
        StreamEvent(type="done"),
    ])
    client = harness_client(loop)
    response = client.post("/chat", json={"user_id": "u", "message": "hi"})
    events = _parse_sse(response.text)
    types = [e["type"] for e in events]
    assert "recall" not in types
    assert types == ["token", "done"]


def test_chat_deduplication_not_needed_in_harness(harness_client):
    """The harness loop owns dedup; the API layer simply forwards each StreamEvent."""
    loop = _FakeLoop([
        StreamEvent(type="tool_call", tool_name="log_mood", tool_args={"mood": "ok"}),
        StreamEvent(type="done"),
    ])
    client = harness_client(loop)
    response = client.post(
        "/chat",
        json={"user_id": "test-user", "message": "hi"},
    )
    events = _parse_sse(response.text)
    tool_call_events = [event for event in events if event["type"] == "tool_call"]
    assert len(tool_call_events) == 1


def test_chat_reuses_persisted_history_between_turns(harness_client):
    prov = _make_provenance()
    first_loop = _FakeLoop([
        StreamEvent(type="token", content="I will remember green."),
        StreamEvent(type="provenance", provenance=prov),
        StreamEvent(type="done"),
    ])
    client = harness_client(first_loop)
    first = client.post(
        "/chat",
        json={"user_id": "context-user", "message": "My favorite color is green."},
    )
    assert first.status_code == 200

    second_loop = _FakeLoop([
        StreamEvent(type="token", content="You told me it is green."),
        StreamEvent(type="provenance", provenance=prov),
        StreamEvent(type="done"),
    ])
    client = harness_client(second_loop)
    second = client.post(
        "/chat",
        json={"user_id": "context-user", "message": "What color do I like?"},
    )
    assert second.status_code == 200
    events = _parse_sse(second.text)
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert "".join(tokens) == "You told me it is green."


def test_chat_emits_error_frame_on_failure(harness_client):
    class _BrokenAgentLoop:
        async def stream_turn(self, session: Any, user_msg: Any) -> AsyncIterator[StreamEvent]:
            yield StreamEvent(type="token", content="partial ")
            raise RuntimeError("upstream exploded")

    client = harness_client(_BrokenAgentLoop())

    response = client.post(
        "/chat",
        json={"user_id": "test-user", "message": "hi"},
    )
    events = _parse_sse(response.text)
    assert [event["type"] for event in events] == ["token", "error"]
    assert "upstream exploded" in events[-1]["content"]


def test_chat_emits_provenance_event_with_all_keys(harness_client):
    """A provenance StreamEvent must propagate all keys to the SSE frame."""
    prov = _make_provenance(
        model="gemma4:e4b",
        tier="gemma",
        latency_ms=42,
        cloud_call=False,
        cloud_payload_summary=None,
        memories_recalled=2,
        skills_called=[],
        rationale="Answered on-device with gemma4:e4b; recalled 2 memories.",
    )
    loop = _FakeLoop([
        StreamEvent(type="token", content="hi"),
        StreamEvent(type="provenance", provenance=prov),
        StreamEvent(type="done"),
    ])
    client = harness_client(loop)
    response = client.post("/chat", json={"user_id": "u", "message": "hi"})
    events = _parse_sse(response.text)
    prov_events = [e for e in events if e["type"] == "provenance"]
    assert len(prov_events) == 1
    p = prov_events[0]["provenance"]
    assert p["cloud_call"] is False
    assert p["cloud_payload_summary"] is None
    assert p["memories_recalled"] == 2
    assert p["skills_called"] == []
    assert p["rationale"] == "Answered on-device with gemma4:e4b; recalled 2 memories."


def test_chat_provenance_cloud_call_true(harness_client):
    """When cloud_call is True the provenance dict must carry the payload summary."""
    prov = _make_provenance(
        model="gemini-2.0-flash",
        tier="gemini",
        latency_ms=300,
        cloud_call=True,
        cloud_payload_summary="5 messages this turn",
        memories_recalled=0,
        skills_called=["log_mood"],
        rationale="Used gemini-2.0-flash (cloud) for this turn — sent 5 messages this turn; recalled 0 memories.",
    )
    loop = _FakeLoop([
        StreamEvent(type="token", content="done"),
        StreamEvent(type="provenance", provenance=prov),
        StreamEvent(type="done"),
    ])
    client = harness_client(loop)
    response = client.post("/chat", json={"user_id": "u", "message": "hi"})
    events = _parse_sse(response.text)
    prov_events = [e for e in events if e["type"] == "provenance"]
    assert len(prov_events) == 1
    p = prov_events[0]["provenance"]
    assert p["cloud_call"] is True
    assert p["cloud_payload_summary"] == "5 messages this turn"
    assert p["skills_called"] == ["log_mood"]
    assert "cloud" in p["rationale"]
