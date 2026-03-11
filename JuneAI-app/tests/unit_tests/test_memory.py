"""Unit tests for the Memory class and tool functions."""

from datetime import date, timedelta
from unittest.mock import patch

import pytest

from agent.memory import Memory
from agent.tools import (
    clear_ui_workspace,
    get_journal,
    get_mood_history,
    get_user_preferences,
    list_calendar_items,
    list_favorites,
    list_food_programs,
    list_goals,
    list_gym_plans,
    list_open_loops,
    log_mood,
    save_calendar_item,
    save_favorite_recommendation,
    save_food_program,
    save_journal_entry,
    save_open_loop,
    save_relationship_profile,
    save_user_preference,
    set_ui_checklist,
    set_ui_focus,
    set_ui_layout,
    summarize_progress,
    track_goal,
    save_gym_plan,
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


def test_save_calendar_items_are_sorted(mem):
    mem.save_calendar_item("Old item", "2023-01-01", "09:00")
    mem.save_calendar_item("Dinner", "2026-04-02", "20:00")
    mem.save_calendar_item("Workout", "2026-03-15", "08:00")
    items = mem.get_calendar_items()
    assert items[0]["title"] == "Workout"
    assert items[1]["title"] == "Dinner"
    assert items[-1]["title"] == "Old item"


def test_save_preferences_and_favorites(mem):
    mem.save_preference("books", "literary fiction", "prefers character-driven novels")
    mem.save_favorite("movie", "Past Lives", "quiet emotional tension")
    preferences = mem.get_preferences()
    favorites = mem.get_favorites()
    assert preferences[0]["value"] == "literary fiction"
    assert favorites[0]["title"] == "Past Lives"


def test_save_gym_and_food_programs(mem):
    mem.save_gym_plan("Spring Split", "Mon push, Wed pull, Fri legs", goal="build muscle")
    mem.save_food_program("Cut Phase", "fat loss", "High protein lunch and dinner")
    gym_plans = mem.get_gym_plans()
    food_programs = mem.get_food_programs()
    assert gym_plans[0]["goal"] == "build muscle"
    assert food_programs[0]["daily_structure"] == "High protein lunch and dinner"


def test_progress_snapshot_includes_new_surfaces(mem):
    mem.save_preference("movie", "slow cinema")
    mem.save_calendar_item("Dentist", "2026-05-01")
    mem.save_favorite("book", "The Neapolitan Novels")
    mem.save_gym_plan("Base", "Tue and Thu full body")
    mem.save_food_program("Maintenance", "energy", "3 meals and 1 snack")
    snapshot = mem.get_progress_snapshot()
    assert snapshot["preference_count"] == 1
    assert snapshot["calendar_count"] == 1
    assert snapshot["favorite_count"] == 1
    assert snapshot["gym_plan_count"] == 1
    assert snapshot["food_program_count"] == 1


def test_daily_checkin_state(mem):
    assert mem.should_send_daily_checkin() is True
    mem.mark_daily_checkin_sent()
    assert mem.should_send_daily_checkin() is False


def test_upcoming_notifications(mem):
    in_three_days = (date.today() + timedelta(days=3)).isoformat()
    in_five_days = (date.today() + timedelta(days=5)).isoformat()
    mem.save_calendar_item("Mom birthday", in_three_days, details="birthday dinner")
    mem.save_open_loop("Book train", due_date=in_five_days, next_step="Choose the morning train")
    notifications = mem.get_upcoming_notifications(limit=5)
    titles = {item["title"] for item in notifications}
    assert "Mom birthday" in titles
    assert "Book train" in titles


@pytest.fixture
def tool_state():
    return {
        "messages": [],
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


def test_log_mood_tool(tool_state, mem):
    result = log_mood.func(mood="calm", note="quiet morning", state=tool_state)
    assert "calm" in result
    assert len(mem.get_mood_history()) == 1


def test_preference_tool(tool_state, mem):
    result = save_user_preference.func(
        category="books",
        value="lyrical fiction",
        context="likes reflective novels",
        state=tool_state,
    )
    saved = mem.get_preferences("books")
    assert "Saved preference" in result
    assert saved[0]["value"] == "lyrical fiction"


def test_calendar_and_favorites_tools(tool_state):
    calendar_result = save_calendar_item.func(
        title="Pick up dry cleaning",
        date="2026-03-18",
        time="18:00",
        state=tool_state,
    )
    favorite_result = save_favorite_recommendation.func(
        category="movie",
        title="Perfect Days",
        reason="quiet observational drama",
        state=tool_state,
    )
    calendar = list_calendar_items.func(state=tool_state)
    favorites = list_favorites.func(state=tool_state)
    assert "Saved calendar item" in calendar_result
    assert "Saved movie recommendation" in favorite_result
    assert "Pick up dry cleaning" in calendar
    assert "Perfect Days" in favorites


def test_wellness_tools(tool_state):
    gym_result = save_gym_plan.func(
        name="Lean Gain",
        schedule="Mon upper, Wed lower, Sat full body",
        goal="muscle",
        state=tool_state,
    )
    food_result = save_food_program.func(
        name="Workday Fuel",
        goal="steady energy",
        daily_structure="Protein breakfast, light lunch, solid dinner",
        state=tool_state,
    )
    gym = list_gym_plans.func(state=tool_state)
    food = list_food_programs.func(state=tool_state)
    assert "Saved gym plan" in gym_result
    assert "Saved food program" in food_result
    assert "Lean Gain" in gym
    assert "Workday Fuel" in food


def test_summarize_progress_tool(tool_state, mem):
    mem.log_mood("steady", "less reactive than last week")
    mem.save_calendar_item("Design review", "2026-03-20")
    mem.save_favorite("book", "Stoner")
    result = summarize_progress.func(state=tool_state)
    assert "Progress snapshot" in result
    assert "Latest mood: steady" in result
    assert "Calendar items: 1" in result


def test_ui_tools_update_ui_state(tool_state):
    result = set_ui_focus.func(
        title="Weekly review",
        body="Capture open loops, schedule priorities, and clear friction.",
        footer="Return here after the next planning pass.",
        state=tool_state,
        tool_call_id="call_1",
    ).update
    assert result["ui_state"]["focus_title"] == "Weekly review"
    assert result["ui_state"]["notice"] == "Return here after the next planning pass."


def test_ui_checklist_and_layout_tools(tool_state):
    checklist_result = set_ui_checklist.func(
        title="Next actions",
        items="- Book trainer\n- Save meal rhythm\n- Add Friday dinner",
        state=tool_state,
        tool_call_id="call_1",
    ).update
    layout_result = set_ui_layout.func(
        layout="focus",
        notice="Surface the highest leverage moves.",
        state=tool_state,
        tool_call_id="call_1",
    ).update
    assert checklist_result["ui_state"]["checklist_items"][1] == "Save meal rhythm"
    assert layout_result["ui_state"]["layout"] == "focus"


def test_clear_ui_workspace_tool(tool_state):
    result = clear_ui_workspace.func(
        state=tool_state,
        tool_call_id="call_1",
    ).update
    assert result["ui_state"]["focus_title"] == "Workspace"
