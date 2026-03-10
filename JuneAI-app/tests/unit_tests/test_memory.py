"""Unit tests for the Memory class and tool functions."""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

from agent.memory import Memory
from agent.tools import (
    clear_ui_workspace,
    draft_reply,
    get_journal,
    get_mood_history,
    get_relationship_context,
    list_goals,
    list_open_loops,
    log_mood,
    save_open_loop,
    save_journal_entry,
    save_relationship_profile,
    set_ui_checklist,
    set_ui_focus,
    set_ui_layout,
    summarize_progress,
    track_goal,
)


@pytest.fixture
def memory_dir(tmp_path):
    """Patch the memory directory for each test."""
    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        yield


@pytest.fixture
def mem(memory_dir):
    """Memory instance backed by a temporary directory."""
    return Memory("test_user")


# --- Memory class tests ---

def test_save_and_load_message(mem):
    mem.save_message("user", "hello")
    mem.save_message("assistant", "hi there")
    history = mem.load_chat()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "hi there"


def test_chat_history_capped_at_50(mem):
    for i in range(60):
        mem.save_message("user", f"msg {i}")
    history = mem.load_chat()
    assert len(history) == 50
    assert history[0]["content"] == "msg 10"


def test_log_and_get_mood(mem):
    mem.log_mood("happy", "great day")
    mem.log_mood("anxious", "big meeting")
    moods = mem.get_mood_history()
    assert len(moods) == 2
    assert moods[0]["mood"] == "happy"
    assert moods[1]["note"] == "big meeting"


def test_get_mood_history_limit(mem):
    for i in range(15):
        mem.log_mood(f"mood_{i}")
    recent = mem.get_mood_history(5)
    assert len(recent) == 5
    assert recent[-1]["mood"] == "mood_14"


def test_save_and_get_journal(mem):
    mem.save_journal("Today I felt brave.")
    mem.save_journal("I set a boundary.")
    entries = mem.get_journal()
    assert len(entries) == 2
    assert entries[0]["entry"] == "Today I felt brave."


def test_journal_limit(mem):
    for i in range(10):
        mem.save_journal(f"entry {i}")
    recent = mem.get_journal(3)
    assert len(recent) == 3
    assert recent[-1]["entry"] == "entry 9"


def test_save_and_get_relationship_profile(mem):
    mem.save_relationship_profile(
        person="Alex",
        relationship="dating",
        summary="Early stage, strong chemistry, uneven texting.",
        user_needs="Consistency",
        cautions="Avoid over-investing too early",
    )
    profiles = mem.get_relationship_profiles("Alex")
    assert len(profiles) == 1
    assert profiles[0]["relationship"] == "dating"
    assert profiles[0]["user_needs"] == "Consistency"


def test_save_and_filter_goals(mem):
    mem.save_goal("Send a clear follow-up", category="dating")
    mem.save_goal("Journal after the date", status="done")
    active_goals = mem.get_goals(status="active")
    assert len(active_goals) == 1
    assert active_goals[0]["title"] == "Send a clear follow-up"


def test_save_and_filter_open_loops(mem):
    mem.save_open_loop("Decide whether to reach out", next_step="Wait until Friday")
    mem.save_open_loop("Book therapy session", status="closed")
    open_loops = mem.get_open_loops(status="open")
    assert len(open_loops) == 1
    assert open_loops[0]["topic"] == "Decide whether to reach out"


@pytest.fixture
def tool_node(memory_dir):
    return ToolNode([
        log_mood,
        get_mood_history,
        save_journal_entry,
        get_journal,
        save_relationship_profile,
        get_relationship_context,
        track_goal,
        list_goals,
        save_open_loop,
        list_open_loops,
        summarize_progress,
        draft_reply,
        set_ui_focus,
        set_ui_checklist,
        set_ui_layout,
        clear_ui_workspace,
    ])


def _run_tool(tool_node, tool_name, args):
    return tool_node.invoke({
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": tool_name,
                    "args": args,
                    "id": "call_1",
                    "type": "tool_call",
                }],
            )
        ],
        "user_id": "test_user",
        "skill": "strategy",
        "ui_state": {
            "layout": "split",
            "focus_title": "Workspace",
            "focus_body": "",
            "checklist_title": "Next steps",
            "checklist_items": [],
            "notice": "",
        },
    })


def test_log_mood_tool(tool_node, mem):
    result = _run_tool(
        tool_node,
        "log_mood",
        {"mood": "calm", "note": "quiet morning"},
    )["messages"][-1].content
    assert "calm" in result
    assert len(mem.get_mood_history()) == 1


def test_get_mood_history_tool_empty(tool_node):
    result = _run_tool(tool_node, "get_mood_history", {})["messages"][-1].content
    assert "No mood history" in result


def test_get_mood_history_tool_with_data(tool_node, mem):
    mem.log_mood("joyful", "sunshine")
    result = _run_tool(tool_node, "get_mood_history", {})["messages"][-1].content
    assert "joyful" in result


def test_save_journal_tool(tool_node, mem):
    result = _run_tool(
        tool_node,
        "save_journal_entry",
        {"entry": "I faced my fear today."},
    )["messages"][-1].content
    assert "saved" in result.lower()
    assert len(mem.get_journal()) == 1


def test_get_journal_tool_with_data(tool_node, mem):
    mem.save_journal("A meaningful reflection.")
    result = _run_tool(tool_node, "get_journal", {})["messages"][-1].content
    assert "A meaningful reflection." in result


def test_relationship_context_tool(tool_node, mem):
    mem.save_relationship_profile(
        person="Taylor",
        relationship="ex",
        summary="Recent contact restarted old confusion.",
        user_needs="Clarity",
    )
    result = _run_tool(
        tool_node,
        "get_relationship_context",
        {"person": "Taylor"},
    )["messages"][-1].content
    assert "Taylor" in result
    assert "Clarity" in result


def test_goal_and_open_loop_tools(tool_node):
    goal_result = _run_tool(
        tool_node,
        "track_goal",
        {
            "title": "Have the boundary conversation",
            "category": "relationship",
            "next_step": "Draft the opener",
        },
    )["messages"][-1].content
    loop_result = _run_tool(
        tool_node,
        "save_open_loop",
        {
            "topic": "Clarify exclusivity",
            "next_step": "Ask directly on Saturday",
        },
    )["messages"][-1].content
    goals = _run_tool(tool_node, "list_goals", {})["messages"][-1].content
    loops = _run_tool(tool_node, "list_open_loops", {})["messages"][-1].content
    assert "Saved goal" in goal_result
    assert "Saved open loop" in loop_result
    assert "Have the boundary conversation" in goals
    assert "Clarify exclusivity" in loops


def test_summarize_progress_tool(tool_node, mem):
    mem.log_mood("steady", "less reactive than last week")
    mem.save_journal("I handled a tense conversation calmly.")
    mem.save_goal("Keep my standards clear", category="dating")
    result = _run_tool(tool_node, "summarize_progress", {})["messages"][-1].content
    assert "Progress snapshot" in result
    assert "Latest mood: steady" in result


def test_ui_tools_update_ui_state(tool_node):
    result = _run_tool(
        tool_node,
        "set_ui_focus",
        {
            "title": "Boundary conversation",
            "body": "State the issue clearly and ask for a direct answer.",
            "footer": "Prepare before Friday.",
        },
    )[0].update
    assert result["ui_state"]["focus_title"] == "Boundary conversation"
    assert "Prepare before Friday." == result["ui_state"]["notice"]


def test_ui_checklist_and_layout_tools(tool_node):
    checklist_result = _run_tool(
        tool_node,
        "set_ui_checklist",
        {
            "title": "Next actions",
            "items": "- Draft opener\n- Ask for clarity\n- Journal after",
        },
    )[0].update
    layout_result = _run_tool(
        tool_node,
        "set_ui_layout",
        {
            "layout": "focus",
            "notice": "Tighten the plan before sending anything.",
        },
    )[0].update
    assert checklist_result["ui_state"]["checklist_items"][1] == "Ask for clarity"
    assert layout_result["ui_state"]["layout"] == "focus"


def test_clear_ui_workspace_tool(tool_node):
    result = _run_tool(tool_node, "clear_ui_workspace", {})[0].update
    assert result["ui_state"]["focus_title"] == "Workspace"
