"""Tests for C.6 provenance dict enrichment in the chat node.

Drives the graph chat node with a fake LLM that returns a known response,
captures writer events, and asserts the provenance event carries all
required C.6 keys with correct values for both local and cloud runtimes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from june_brain import graph as brain_graph
from june_brain.config import RuntimeConfig
from langchain_core.messages import AIMessage, HumanMessage


class _LocalLLM:
    """Fake local LLM — returns a simple answer, no tool calls."""

    def bind_tools(self, _tools: Any) -> "_LocalLLM":
        return self

    def invoke(self, _messages: Any) -> AIMessage:
        return AIMessage(content="hello")


class _CloudLLM:
    """Fake cloud LLM — returns a simple answer, no tool calls."""

    def bind_tools(self, _tools: Any) -> "_CloudLLM":
        return self

    def invoke(self, _messages: Any) -> AIMessage:
        return AIMessage(content="hello from cloud")


def _make_local_runtime() -> RuntimeConfig:
    return RuntimeConfig(
        preset_key="gemma",
        provider="openai_compatible",
        model="gemma4:e4b",
        base_url="http://127.0.0.1:11434/v1",
        api_key="",
        label="Gemma 4 (local)",
        tool_strategy="recovery",
        temperature=0.7,
        max_tokens=2048,
    )


def _make_cloud_runtime() -> RuntimeConfig:
    return RuntimeConfig(
        preset_key="gemini",
        provider="openai_compatible",
        model="gemini-2.0-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="test-key",
        label="Gemini 2.0 Flash",
        tool_strategy="native",
        temperature=0.7,
        max_tokens=2048,
    )


def _collect_writer_events(
    llm: Any,
    runtime: RuntimeConfig,
    tmp_path: Any,
) -> list[dict]:
    """Build the chat node, run it with a single human message, collect writer events."""
    events: list[dict] = []

    def _writer(event: dict) -> None:
        events.append(event)

    with (
        patch("june_brain.graph.load_skill_tools", return_value=[]),
        patch("june_brain.memory.MEMORY_DIR", str(tmp_path)),
        patch("june_brain.graph._build_memory_context", return_value=""),
        patch("june_brain.graph._recall_with_hits", return_value=("", [])),
    ):
        agent = brain_graph.create_june_agent(llm=llm, runtime=runtime)
        state = {
            "user_id": "test-user",
            "messages": [HumanMessage(content="hi")],
            "skill": "assistant",
        }
        import asyncio

        async def _run() -> None:
            async for chunk in agent.astream(
                state, stream_mode=["custom", "messages", "updates"]
            ):
                if isinstance(chunk, tuple) and chunk[0] == "custom":
                    events.append(chunk[1])

        asyncio.run(_run())

    return events


@pytest.fixture(autouse=True)
def _reset_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    brain_graph.invalidate_agent()
    yield
    brain_graph.invalidate_agent()


def test_local_provenance_has_cloud_call_false(tmp_path: Any) -> None:
    """Local runtime must emit cloud_call=False and a non-empty rationale."""
    events = _collect_writer_events(_LocalLLM(), _make_local_runtime(), tmp_path)
    prov = next((e for e in events if e.get("event") == "provenance"), None)
    assert prov is not None, "provenance event not emitted"
    assert prov["cloud_call"] is False
    assert prov["cloud_payload_summary"] is None
    assert isinstance(prov["rationale"], str) and len(prov["rationale"]) > 0
    assert "locally" in prov["rationale"]
    assert isinstance(prov["memories_recalled"], int)
    assert isinstance(prov["skills_called"], list)


def test_cloud_provenance_has_cloud_call_true(tmp_path: Any) -> None:
    """Cloud runtime must emit cloud_call=True with a payload summary and rationale."""
    events = _collect_writer_events(_CloudLLM(), _make_cloud_runtime(), tmp_path)
    prov = next((e for e in events if e.get("event") == "provenance"), None)
    assert prov is not None, "provenance event not emitted"
    assert prov["cloud_call"] is True
    assert prov["cloud_payload_summary"] is not None
    assert "messages" in prov["cloud_payload_summary"]
    assert isinstance(prov["rationale"], str) and len(prov["rationale"]) > 0
    assert "cloud" in prov["rationale"]


def test_provenance_carries_existing_keys(tmp_path: Any) -> None:
    """Existing keys provider/model/tier/latency_ms must still be present."""
    events = _collect_writer_events(_LocalLLM(), _make_local_runtime(), tmp_path)
    prov = next((e for e in events if e.get("event") == "provenance"), None)
    assert prov is not None
    assert prov["provider"] == "gemma"
    assert prov["model"] == "gemma4:e4b"
    assert prov["tier"] == "local"
    assert isinstance(prov["latency_ms"], int)
