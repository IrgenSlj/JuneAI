from langgraph.pregel import Pregel

from june_brain.graph import june_agent


def test_agent_is_compiled_graph() -> None:
    assert isinstance(june_agent, Pregel)


def test_detect_tool_strategy_gemma4() -> None:
    from june_brain.config import detect_tool_strategy
    assert detect_tool_strategy("gemma4:e4b") == "native"
    assert detect_tool_strategy("gemma4") == "native"
    assert detect_tool_strategy("gemma-4-it") == "native"


def test_detect_tool_strategy_mistral() -> None:
    from june_brain.config import detect_tool_strategy
    assert detect_tool_strategy("mistral:7b-instruct-v0.3") == "native"
    assert detect_tool_strategy("mistral-nemo") == "native"
    assert detect_tool_strategy("unknown-model-xyz") == "recovery"
