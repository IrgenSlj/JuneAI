from langgraph.pregel import Pregel

from june_brain.graph import june_agent


def test_agent_is_compiled_graph() -> None:
    assert isinstance(june_agent, Pregel)


def test_detect_tool_strategy_returns_native() -> None:
    # Per ADR 0002 both supported model families use native
    # OpenAI-compatible tool calling, so this helper is a constant.
    from june_brain.config import detect_tool_strategy
    assert detect_tool_strategy("gemma4:e4b") == "native"
    assert detect_tool_strategy("gemini-2.0-flash") == "native"
    assert detect_tool_strategy("anything-else") == "native"
