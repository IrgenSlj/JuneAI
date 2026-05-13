from june_brain import graph as brain_graph
from langchain_core.messages import AIMessage
from langgraph.pregel import Pregel


class _FakeLLM:
    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return AIMessage(content="OK")


def test_agent_factory_returns_compiled_graph(monkeypatch) -> None:
    monkeypatch.setattr(brain_graph, "load_skill_tools", lambda: [])
    agent = brain_graph.create_june_agent(llm=_FakeLLM())
    assert isinstance(agent, Pregel)


def test_get_or_create_agent_rebuilds_when_runtime_changes(monkeypatch) -> None:
    builds: list[str] = []

    def _build_agent(*, runtime):
        builds.append(runtime.model)
        return object()

    brain_graph.invalidate_agent()
    monkeypatch.setenv("MODEL_PROVIDER", "gemma")
    monkeypatch.setenv("GEMMA_MODEL", "gemma-one")
    monkeypatch.setattr(brain_graph, "create_june_agent", _build_agent)

    first = brain_graph.get_or_create_agent()
    second = brain_graph.get_or_create_agent()
    monkeypatch.setenv("GEMMA_MODEL", "gemma-two")
    third = brain_graph.get_or_create_agent()

    assert first is second
    assert third is not first
    assert builds == ["gemma-one", "gemma-two"]
    brain_graph.invalidate_agent()


def test_detect_tool_strategy_returns_native() -> None:
    # Per ADR 0002 both supported model families use native
    # OpenAI-compatible tool calling, so this helper is a constant.
    from june_brain.config import detect_tool_strategy
    assert detect_tool_strategy("gemma4:e4b") == "native"
    assert detect_tool_strategy("gemini-2.0-flash") == "native"
    assert detect_tool_strategy("anything-else") == "native"
