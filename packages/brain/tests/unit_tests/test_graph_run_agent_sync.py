"""Regression tests for synchronous scheduled-agent invocation."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from june_brain import graph


class _FakeModel:
    def __init__(self) -> None:
        self.messages = []

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content="scheduled reply")


def test_run_agent_sync_uses_existing_prompt_module_without_bound_agent_tools(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fake = _FakeModel()
    monkeypatch.setattr("june_brain.memory.MEMORY_DIR", str(tmp_path))
    monkeypatch.setattr(graph, "get_or_create_agent", lambda: (_ for _ in ()).throw(AssertionError("agent should not be used")))
    monkeypatch.setattr("june_brain.config.resolve_runtime_config", lambda: object())
    monkeypatch.setattr("june_brain.models.build_chat_model", lambda _runtime: fake)

    result = graph.run_agent_sync("summarize my day", user_id="scheduled-user")

    assert result == "scheduled reply"
    assert fake.messages[-1].content == "summarize my day"
