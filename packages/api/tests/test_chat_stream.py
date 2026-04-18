"""Integration test for POST /chat.

Uses a fake agent that yields the same chunk shape LangGraph produces
(``(mode, chunk)`` pairs), so we exercise the real SSE framing, schema
validation, and tool-call dedup without booting an LLM or Ollama.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from june_api.app import create_app
from june_api.routes.chat import get_agent


class _FakeAgent:
    """Replays a scripted sequence of (mode, chunk) pairs via astream."""

    def __init__(self, script: list[tuple[str, Any]]) -> None:
        self._script = script

    async def astream(
        self, _state: dict, stream_mode: list[str] | None = None
    ) -> AsyncIterator[tuple[str, Any]]:
        for item in self._script:
            yield item


def _parse_sse(body: str) -> list[dict]:
    events = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk.startswith("data:"):
            continue
        events.append(json.loads(chunk[len("data:"):].strip()))
    return events


@pytest.fixture
def client_with_agent():
    """Build a TestClient whose /chat uses a caller-supplied fake agent."""
    app = create_app()

    def _install(agent: _FakeAgent) -> TestClient:
        app.dependency_overrides[get_agent] = lambda: agent
        return TestClient(app)

    return _install


def test_chat_streams_tokens_tool_calls_and_done(client_with_agent):
    script = [
        ("messages", (AIMessage(content="Hello "), {})),
        ("messages", (AIMessage(content="Alex."), {})),
        (
            "updates",
            {
                "chat": {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": "call_1",
                                    "name": "log_mood",
                                    "args": {"mood": "good"},
                                }
                            ],
                        )
                    ]
                }
            },
        ),
        (
            "updates",
            {
                "tools": {
                    "messages": [
                        ToolMessage(
                            content="logged",
                            name="log_mood",
                            tool_call_id="call_1",
                        )
                    ]
                }
            },
        ),
    ]

    client = client_with_agent(_FakeAgent(script))
    response = client.post(
        "/chat",
        json={"user_id": "test-user", "message": "hi"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    types = [event["type"] for event in events]

    assert types == ["token", "token", "tool_call", "tool_result", "done"]

    tokens = [event["content"] for event in events if event["type"] == "token"]
    assert "".join(tokens) == "Hello Alex."

    tool_call = next(event for event in events if event["type"] == "tool_call")
    assert tool_call["tool_name"] == "log_mood"
    assert tool_call["tool_args"] == {"mood": "good"}

    tool_result = next(event for event in events if event["type"] == "tool_result")
    assert tool_result["tool_name"] == "log_mood"
    assert tool_result["tool_result"] == "logged"


def test_chat_dedupes_tool_calls_across_message_and_update(client_with_agent):
    """LangGraph replays the same AIMessage in both ``messages`` and ``updates``
    streams; the route must emit each tool_call exactly once."""

    tool_call = {"id": "call_abc", "name": "log_mood", "args": {"mood": "ok"}}
    ai_with_call = AIMessage(content="", tool_calls=[tool_call])

    script = [
        ("updates", {"chat": {"messages": [ai_with_call]}}),
        ("updates", {"chat": {"messages": [ai_with_call]}}),
    ]

    client = client_with_agent(_FakeAgent(script))
    response = client.post(
        "/chat",
        json={"user_id": "test-user", "message": "hi"},
    )
    events = _parse_sse(response.text)
    tool_call_events = [event for event in events if event["type"] == "tool_call"]
    assert len(tool_call_events) == 1


def test_chat_emits_error_frame_on_failure(client_with_agent):
    class _BrokenAgent:
        async def astream(self, _state, stream_mode=None):
            yield ("messages", (AIMessage(content="partial "), {}))
            raise RuntimeError("upstream exploded")

    app = create_app()
    app.dependency_overrides[get_agent] = lambda: _BrokenAgent()
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"user_id": "test-user", "message": "hi"},
    )
    events = _parse_sse(response.text)
    assert [event["type"] for event in events] == ["token", "error"]
    assert "upstream exploded" in events[-1]["content"]
