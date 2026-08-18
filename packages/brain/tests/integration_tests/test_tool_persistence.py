"""Integration: native memory-writing tools persist to SQLite.

Each test invokes a native tool directly with an injected ``state`` carrying
the user_id. After the call we read back from a fresh Memory instance
(different object, same db file) to confirm the write really landed.
"""

from __future__ import annotations

from unittest.mock import patch

from june_brain.memory import Memory
from june_brain.tools import JUNE_TOOLS

_TOOLS = {t.name: t for t in JUNE_TOOLS}


def _invoke(name: str, args: dict, user_id: str):
    return _TOOLS[name].invoke({**args, "state": {"user_id": user_id}})


def test_save_calendar_item_persists_to_sqlite(tmp_path):
    """A calendar save tool call is readable afterwards."""
    user_id = "cal_user"

    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        _invoke(
            "save_calendar_item",
            {
                "title": "Doctor appointment",
                "date": "2026-05-15",
                "time": "10:00",
                "details": "Annual check-up",
            },
            user_id,
        )
        mem = Memory(user_id)
        items = mem.get_calendar_items(status="", limit=10)

    assert any(item["title"] == "Doctor appointment" for item in items)
    match = next(i for i in items if i["title"] == "Doctor appointment")
    assert match["date"] == "2026-05-15"
    assert match["time"] == "10:00"


def test_track_goal_persists_to_sqlite(tmp_path):
    """A track_goal tool call is readable from a separate Memory instance."""
    user_id = "goal_user"

    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        _invoke(
            "track_goal",
            {
                "title": "Run a half marathon",
                "next_step": "Sign up for a local 5k first",
                "target_date": "2026-09-01",
                "category": "fitness",
            },
            user_id,
        )
        mem = Memory(user_id)
        goals = mem.get_goals(status="", limit=10)

    assert any(g["title"] == "Run a half marathon" for g in goals)
    match = next(g for g in goals if g["title"] == "Run a half marathon")
    assert match["next_step"] == "Sign up for a local 5k first"
    assert match["category"] == "fitness"



def test_save_open_loop_persists_to_sqlite(tmp_path):
    """A save_open_loop tool call is readable from a separate Memory instance."""
    user_id = "stats_user"

    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        _invoke(
            "save_open_loop",
            {
                "topic": "Follow up with landlord",
                "next_step": "Send email by Friday",
            },
            user_id,
        )
        mem = Memory(user_id)
        loops = mem.get_open_loops(status="", limit=10)

    assert any(loop_item["topic"] == "Follow up with landlord" for loop_item in loops)
    match = next(loop_item for loop_item in loops if loop_item["topic"] == "Follow up with landlord")
    assert match["next_step"] == "Send email by Friday"


def test_two_tools_write_in_same_user(tmp_path):
    """Two writing tools both persist under the same user."""
    user_id = "combo_user"

    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        _invoke(
            "save_calendar_item",
            {"title": "Team lunch", "date": "2026-05-20"},
            user_id,
        )
        _invoke(
            "save_open_loop",
            {"topic": "Confirm the room", "next_step": "Ask on Monday"},
            user_id,
        )
        mem = Memory(user_id)
        items = mem.get_calendar_items(status="", limit=10)
        loops = mem.get_open_loops(status="open", limit=10)

    assert any(i["title"] == "Team lunch" for i in items)
    assert any(loop_item["topic"] == "Confirm the room" for loop_item in loops)
