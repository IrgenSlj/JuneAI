"""Integration: agent fires memory-writing tool calls; SQLite is the source of truth.

Each test runs the full LangGraph turn with a fake LLM that returns a native
tool_calls payload.  After the turn we read back from a fresh Memory instance
(different object, same db file) to confirm the write really landed.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import create_june_agent
from agent.memory import Memory

pytestmark = pytest.mark.anyio

_BASE_STATE = {
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


class _ToolLLM:
    """Fake LLM: first call returns a native tool_calls list, second call replies."""

    def __init__(self, tool_name: str, tool_args: dict, call_id: str = "c1"):
        self._tool_name = tool_name
        self._tool_args = tool_args
        self._call_id = call_id
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
                        "name": self._tool_name,
                        "args": self._tool_args,
                        "id": self._call_id,
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="Done.")


async def test_save_calendar_item_persists_to_sqlite(tmp_path):
    """A calendar save tool call written in one agent turn is readable afterwards."""
    user_id = "cal_user"
    agent = create_june_agent(llm=_ToolLLM("save_calendar_item", {
        "title": "Doctor appointment",
        "date": "2026-05-15",
        "time": "10:00",
        "details": "Annual check-up",
    }))

    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        await agent.ainvoke({
            **_BASE_STATE,
            "messages": [HumanMessage(content="Doctor appointment May 15 at 10am.")],
            "user_id": user_id,
        })
        # Fresh instance — reads from same db file
        mem = Memory(user_id)
        items = mem.get_calendar_items(status="", limit=10)

    assert any(item["title"] == "Doctor appointment" for item in items)
    match = next(i for i in items if i["title"] == "Doctor appointment")
    assert match["date"] == "2026-05-15"
    assert match["time"] == "10:00"


async def test_track_goal_persists_to_sqlite(tmp_path):
    """A track_goal tool call is readable from a separate Memory instance."""
    user_id = "goal_user"
    agent = create_june_agent(llm=_ToolLLM("track_goal", {
        "title": "Run a half marathon",
        "next_step": "Sign up for a local 5k first",
        "target_date": "2026-09-01",
        "category": "fitness",
    }))

    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        await agent.ainvoke({
            **_BASE_STATE,
            "messages": [HumanMessage(content="I want to run a half marathon by September.")],
            "user_id": user_id,
        })
        mem = Memory(user_id)
        goals = mem.get_goals(status="", limit=10)

    assert any(g["title"] == "Run a half marathon" for g in goals)
    match = next(g for g in goals if g["title"] == "Run a half marathon")
    assert match["next_step"] == "Sign up for a local 5k first"
    assert match["category"] == "fitness"


async def test_log_workout_session_persists_to_sqlite(tmp_path):
    """A log_workout_session tool call is readable from a separate Memory instance."""
    user_id = "workout_user"
    agent = create_june_agent(llm=_ToolLLM("log_workout_session", {
        "plan_name": "Push Day",
        "exercises": "Bench press 4x8, OHP 3x10, Tricep dips 3x12",
        "duration_min": 55,
        "energy_rating": 4,
        "notes": "Felt strong today",
    }))

    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        await agent.ainvoke({
            **_BASE_STATE,
            "messages": [HumanMessage(content="Just finished push day — bench, OHP, dips.")],
            "user_id": user_id,
        })
        mem = Memory(user_id)
        sessions = mem.get_workout_sessions(limit=10)

    assert sessions, "No workout session saved"
    assert sessions[0]["plan_name"] == "Push Day"
    assert sessions[0]["duration_min"] == 55
    assert sessions[0]["energy_rating"] == 4


async def test_tool_stats_are_updated_after_successful_call(tmp_path):
    """tool_stats.succeeded increments when a memory tool completes without error."""
    user_id = "stats_user"
    agent = create_june_agent(llm=_ToolLLM("save_open_loop", {
        "topic": "Follow up with landlord",
        "next_step": "Send email by Friday",
    }))

    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        result = await agent.ainvoke({
            **_BASE_STATE,
            "messages": [HumanMessage(content="I need to follow up with the landlord.")],
            "user_id": user_id,
        })

    assert result["tool_stats"]["requested"] >= 1
    assert result["tool_stats"]["succeeded"] >= 1
    assert result["tool_stats"]["failed"] == 0


async def test_ui_state_and_memory_update_in_same_turn(tmp_path):
    """set_ui_chapter + save_calendar_item can both fire in one turn."""

    class _TwoToolLLM:
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
                            "name": "save_calendar_item",
                            "args": {"title": "Team lunch", "date": "2026-05-20"},
                            "id": "c1",
                            "type": "tool_call",
                        },
                        {
                            "name": "set_ui_chapter",
                            "args": {"chapter": "calendar"},
                            "id": "c2",
                            "type": "tool_call",
                        },
                    ],
                )
            return AIMessage(content="Saved and panel updated.")

    user_id = "combo_user"
    agent = create_june_agent(llm=_TwoToolLLM())

    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        result = await agent.ainvoke({
            **_BASE_STATE,
            "messages": [HumanMessage(content="Team lunch on May 20.")],
            "user_id": user_id,
        })
        mem = Memory(user_id)
        items = mem.get_calendar_items(status="", limit=10)

    assert result["ui_state"]["selected_chapter"] == "calendar"
    assert any(i["title"] == "Team lunch" for i in items)
