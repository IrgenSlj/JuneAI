from unittest.mock import patch

import pytest
from june_brain.graph import create_june_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

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


class FakeLayoutLLM:
    def __init__(self):
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "set_ui_layout",
                        "args": {"layout": "focus"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="Layout updated.")


async def test_agent_handles_single_command_toolnode_result(tmp_path) -> None:
    agent = create_june_agent(llm=FakeLayoutLLM())
    inputs = {
        "messages": [HumanMessage(content="Switch to focus mode.")],
        "user_id": "command_tool_user",
        "skill": "assistant",
        "ui_state": {
            "layout": "split",
            "show_right_panel": True,
            "selected_chapter": "",
            "focus_title": "Workspace",
            "focus_body": "",
            "checklist_title": "Next steps",
            "checklist_items": [],
            "notice": "",
        },
        "tool_stats": {"requested": 0, "succeeded": 0, "failed": 0, "last_calls": []},
    }

    command_result = Command(
        update={
            "ui_state": {"layout": "focus", "show_right_panel": False},
            "messages": [
                ToolMessage(
                    content="Workspace layout set to 'focus'.",
                    tool_call_id="call_1",
                    status="success",
                )
            ],
        }
    )

    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)), patch(
        "june_brain.graph.ToolNode._func",
        return_value=command_result,
    ):
        result = await agent.ainvoke(inputs)

    assert result["ui_state"]["layout"] == "focus"
    assert result["ui_state"]["show_right_panel"] is False
    assert result["tool_stats"]["requested"] == 1
    assert result["tool_stats"]["succeeded"] == 1
    assert result["tool_stats"]["failed"] == 0
