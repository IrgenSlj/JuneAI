"""Integration test for POST /chat.

Uses a fake agent that yields the same chunk shape LangGraph produces
(``(mode, chunk)`` pairs), so we exercise the real SSE framing, schema
validation, and tool-call dedup without booting an LLM or Ollama.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from june_api.app import create_app
from june_api.routes.chat import get_agent
from june_brain.memory import vector as vector_module
from langchain_core.messages import AIMessage, ToolMessage


class _FakeAgent:
    """Replays a scripted sequence of (mode, chunk) pairs via astream."""

    def __init__(self, script: list[tuple[str, Any]]) -> None:
        self._script = script
        self.states: list[dict[str, Any]] = []

    async def astream(
        self, state: dict, stream_mode: list[str] | None = None
    ) -> AsyncIterator[tuple[str, Any]]:
        self.states.append(state)
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
def client_with_agent(tmp_path):
    """Build a TestClient whose /chat uses a caller-supplied fake agent."""
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)), patch(
        "june_api.routes.chat.MemoryManager.extract",
        return_value={"facts": 0, "entities": 0, "relations": 0},
    ):
        vector_module.reset_singletons()
        app = create_app()

        def _install(agent: Any) -> TestClient:
            app.dependency_overrides[get_agent] = lambda: agent
            return TestClient(app)

        yield _install
        vector_module.reset_singletons()


def test_chat_streams_tokens_tool_calls_and_done(client_with_agent):
    script = [
        (
            "custom",
            {
                "event": "recall",
                "hits": [
                    {
                        "ref": "semantic:abc",
                        "text": "User loves ramen",
                        "source": "vector",
                        "kind": "fact",
                        "score": 0.2,
                    }
                ],
            },
        ),
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

    assert types == ["recall", "token", "token", "tool_call", "tool_result", "done"]

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


def test_chat_skips_empty_recall_hits(client_with_agent):
    """A recall event with no hits should not emit a frame to avoid empty UI noise."""
    script = [
        ("custom", {"event": "recall", "hits": []}),
        ("messages", (AIMessage(content="hi"), {})),
    ]
    client = client_with_agent(_FakeAgent(script))
    response = client.post("/chat", json={"user_id": "u", "message": "hi"})
    events = _parse_sse(response.text)
    types = [e["type"] for e in events]
    assert "recall" not in types
    assert types == ["token", "done"]


def test_chat_ignores_unknown_custom_events(client_with_agent):
    """The chat stream subscribes to 'custom' for recall; other custom events shouldn't leak."""
    script = [
        ("custom", {"event": "chat_started", "skill": "assistant"}),
        ("messages", (AIMessage(content="hi"), {})),
    ]
    client = client_with_agent(_FakeAgent(script))
    response = client.post("/chat", json={"user_id": "u", "message": "hi"})
    events = _parse_sse(response.text)
    types = [e["type"] for e in events]
    assert types == ["token", "done"]


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


def test_chat_reuses_persisted_history_between_turns(client_with_agent):
    first_agent = _FakeAgent(
        [("messages", (AIMessage(content="I will remember green."), {}))]
    )
    client = client_with_agent(first_agent)
    first = client.post(
        "/chat",
        json={"user_id": "context-user", "message": "My favorite color is green."},
    )
    assert first.status_code == 200

    second_agent = _FakeAgent(
        [("messages", (AIMessage(content="You told me it is green."), {}))]
    )
    client = client_with_agent(second_agent)
    second = client.post(
        "/chat",
        json={"user_id": "context-user", "message": "What color do I like?"},
    )
    assert second.status_code == 200

    contents = [message.content for message in second_agent.states[0]["messages"]]
    assert contents[-3:] == [
        "My favorite color is green.",
        "I will remember green.",
        "What color do I like?",
    ]


def test_chat_emits_error_frame_on_failure(client_with_agent):
    class _BrokenAgent:
        async def astream(self, _state, stream_mode=None):
            yield ("messages", (AIMessage(content="partial "), {}))
            raise RuntimeError("upstream exploded")

    client = client_with_agent(_BrokenAgent())

    response = client.post(
        "/chat",
        json={"user_id": "test-user", "message": "hi"},
    )
    events = _parse_sse(response.text)
    assert [event["type"] for event in events] == ["token", "error"]
    assert "upstream exploded" in events[-1]["content"]
