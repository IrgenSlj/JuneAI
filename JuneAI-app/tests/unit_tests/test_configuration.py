from langgraph.pregel import Pregel

from agent.graph import june_agent


def test_agent_is_compiled_graph() -> None:
    assert isinstance(june_agent, Pregel)
