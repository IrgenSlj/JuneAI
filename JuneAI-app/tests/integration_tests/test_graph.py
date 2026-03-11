import pytest
from langchain_core.messages import HumanMessage

from agent.graph import june_agent

pytestmark = pytest.mark.anyio


@pytest.mark.langsmith
async def test_agent_simple_response() -> None:
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
    }
    res = await june_agent.ainvoke(inputs)
    assert res is not None
    assert len(res["messages"]) > 1
