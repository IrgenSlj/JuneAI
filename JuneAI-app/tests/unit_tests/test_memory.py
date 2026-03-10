"""Unit tests for the Memory class and tool functions."""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

from agent.memory import Memory
from agent.tools import (
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
    ])


def _run_tool(tool_node, tool_name, args):
    result = tool_node.invoke({
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
    })
    return result["messages"][-1].content


def test_log_mood_tool(tool_node, mem):
    result = _run_tool(
        tool_node,
        "log_mood",
        {"mood": "calm", "note": "quiet morning"},
    )
    assert "calm" in result
    assert len(mem.get_mood_history()) == 1


def test_get_mood_history_tool_empty(tool_node):
    result = _run_tool(tool_node, "get_mood_history", {})
    assert "No mood history" in result


def test_get_mood_history_tool_with_data(tool_node, mem):
    mem.log_mood("joyful", "sunshine")
    result = _run_tool(tool_node, "get_mood_history", {})
    assert "joyful" in result


def test_save_journal_tool(tool_node, mem):
    result = _run_tool(
        tool_node,
        "save_journal_entry",
        {"entry": "I faced my fear today."},
    )
    assert "saved" in result.lower()
    assert len(mem.get_journal()) == 1


def test_get_journal_tool_with_data(tool_node, mem):
    mem.save_journal("A meaningful reflection.")
    result = _run_tool(tool_node, "get_journal", {})
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
    )
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
    )
    loop_result = _run_tool(
        tool_node,
        "save_open_loop",
        {
            "topic": "Clarify exclusivity",
            "next_step": "Ask directly on Saturday",
        },
    )
    goals = _run_tool(tool_node, "list_goals", {})
    loops = _run_tool(tool_node, "list_open_loops", {})
    assert "Saved goal" in goal_result
    assert "Saved open loop" in loop_result
    assert "Have the boundary conversation" in goals
    assert "Clarify exclusivity" in loops


def test_summarize_progress_tool(tool_node, mem):
    mem.log_mood("steady", "less reactive than last week")
    mem.save_journal("I handled a tense conversation calmly.")
    mem.save_goal("Keep my standards clear", category="dating")
    result = _run_tool(tool_node, "summarize_progress", {})
    assert "Progress snapshot" in result
    assert "Latest mood: steady" in result
