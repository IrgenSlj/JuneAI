"""Unit tests for the Memory class.

The tool half went with the v1 domain writers (D.5a / ADR 0032). The store
methods stay: existing rows still feed recall through its keyword channel, so
they are live until D.5b decides the store layer's fate. The four tools that
replaced them are covered in `test_memory_tools.py`.
"""

from datetime import date, timedelta
from unittest.mock import patch

import pytest
from june_brain.memory import Memory


@pytest.fixture
def memory_dir(tmp_path):
    """Patch the memory directory for each test."""
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
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


def test_memory_creates_nested_directories(tmp_path):
    nested_dir = tmp_path / "nested" / "memory" / "state"
    with patch("june_brain.memory.MEMORY_DIR", str(nested_dir)):
        memory = Memory("test_user")
        memory.save_message("user", "hello")

    assert (nested_dir / "memory").exists()
    assert (nested_dir / "memory" / "june.db").exists()


def test_chat_history_capped_at_50(mem):
    for i in range(60):
        mem.save_message("user", f"msg {i}")
    history = mem.load_chat()
    assert len(history) == 50
    assert history[0]["content"] == "msg 10"


def test_chat_history_is_durable_across_instances(memory_dir):
    """Messages written by one Memory instance are visible to another."""
    m1 = Memory("test_user")
    m1.save_message("user", "hello")
    m2 = Memory("test_user")
    history = m2.load_chat()
    assert len(history) == 1
    assert history[0]["content"] == "hello"


def test_save_calendar_items_are_sorted(mem):
    today = date.today()
    mem.save_calendar_item("Old item", (today - timedelta(days=30)).isoformat(), "09:00")
    mem.save_calendar_item("Dinner", (today + timedelta(days=5)).isoformat(), "20:00")
    mem.save_calendar_item("Workout", (today + timedelta(days=1)).isoformat(), "08:00")
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


def test_daily_checkin_state(mem):
    assert mem.should_send_daily_checkin() is True
    mem.mark_daily_checkin_sent()
    assert mem.should_send_daily_checkin() is False


def test_upcoming_notifications(mem):
    in_three_days = (date.today() + timedelta(days=3)).isoformat()
    in_five_days = (date.today() + timedelta(days=5)).isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    mem.save_calendar_item("Mom birthday", in_three_days, details="birthday dinner")
    mem.save_calendar_item("Archived call", in_three_days, status="completed")
    mem.save_calendar_item("Yesterday trip", yesterday, details="weekend getaway")
    mem.save_open_loop("Book train", due_date=in_five_days, next_step="Choose the morning train")
    notifications = mem.get_upcoming_notifications(limit=5)
    titles = {item["title"] for item in notifications}
    assert "Mom birthday" in titles
    assert "Book train" in titles
    assert "Yesterday trip" in titles
    assert "Archived call" not in titles


def test_status_transition_methods(mem):
    mem.save_calendar_item("Dentist", "2026-04-02")
    mem.save_goal("Write proposal")
    mem.save_open_loop("Book train")

    calendar_item = mem.update_calendar_item_status("Dentist", "completed")
    goal = mem.update_goal_status("Write proposal", "paused")
    loop = mem.update_open_loop_status("Book train", "closed")

    assert calendar_item["status"] == "completed"
    assert goal["status"] == "paused"
    assert loop["status"] == "closed"
