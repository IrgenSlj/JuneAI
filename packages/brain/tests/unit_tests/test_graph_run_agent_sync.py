"""Regression tests for synchronous scheduled-agent invocation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from june_brain import graph


class _FakeAgent:
    def __init__(self) -> None:
        self.state: dict[str, Any] | None = None

    def invoke(self, state: dict[str, Any]) -> dict[str, list[AIMessage]]:
        self.state = state
        return {"messages": [AIMessage(content="scheduled reply")]}


def test_run_agent_sync_uses_existing_prompt_module(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fake = _FakeAgent()
    monkeypatch.setattr("june_brain.memory.MEMORY_DIR", str(tmp_path))
    monkeypatch.setattr(graph, "get_or_create_agent", lambda: fake)

    result = graph.run_agent_sync("summarize my day", user_id="scheduled-user")

    assert result == "scheduled reply"
    assert fake.state is not None
    assert fake.state["user_id"] == "scheduled-user"
    assert fake.state["skill"] == "default"
    assert fake.state["messages"][-1].content == "summarize my day"
