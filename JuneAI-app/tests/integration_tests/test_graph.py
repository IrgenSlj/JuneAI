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
            "selected_chapter": "",
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


class FakeToolCallJSONLLM:
    def __init__(self):
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content=(
                    '[{"tool":"set_ui_chapter","args":{"chapter":"plans"}},'
                    '{"tool":"set_ui_layout","args":{"layout":"focus"}}]'
                )
            )
        return AIMessage(content="Ready.")


async def test_agent_recovers_multiple_json_tool_calls() -> None:
    agent = create_june_agent(llm=FakeToolCallJSONLLM())
    inputs = {
        "messages": [HumanMessage(content="Show me my plans and focus the page.")],
        "user_id": "test_user",
        "skill": "assistant",
        "ui_state": {
            "layout": "split",
            "selected_chapter": "",
            "focus_title": "Workspace",
            "focus_body": "",
            "checklist_title": "Next steps",
            "checklist_items": [],
            "notice": "",
        },
        "tool_stats": {"requested": 0, "succeeded": 0, "failed": 0, "last_calls": []},
    }
    res = await agent.ainvoke(inputs)
    assert res["ui_state"]["selected_chapter"] == "plans"
    assert res["ui_state"]["layout"] == "focus"
