import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import create_june_agent

pytestmark = pytest.mark.anyio


class FakeLLM:
    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return AIMessage(content="Hello. I'm ready to help.")


async def test_agent_simple_response() -> None:
    agent = create_june_agent(llm=FakeLLM())
    inputs = {
        "messages": [HumanMessage(content="Hello, how are you?")],
        "user_id": "test_user",
        "skill": "assistant",
        "ui_state": {
            "layout": "split",
            "focus_title": "Workspace",
            "focus_body": "",
            "checklist_title": "Next steps",
            "checklist_items": [],
            "notice": "",
        },
        "tool_stats": {"requested": 0, "succeeded": 0, "failed": 0, "last_calls": []},
    }
    res = await agent.ainvoke(inputs)
    assert res is not None
    assert len(res["messages"]) > 1
