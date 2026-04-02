from agent_ui.layout import (
    layout_column_widths,
    normalize_layout,
    normalize_rail_view,
    sync_rail_view,
    sync_right_panel_visibility,
    sync_ui_layout,
)
from agent_ui.state import (
    reset_shell_state,
    shell_state_defaults,
    sync_selected_chapter,
    sync_shell_flags,
)


def test_shell_state_defaults_normalize_and_copy_nested_values() -> None:
    ui_state = {
        "layout": "Focus",
        "selected_chapter": "Plans",
        "show_right_panel": 0,
        "checklist_items": ["Capture next step"],
        "notice": "Keep the main thing visible.",
    }
    tool_stats = {
        "requested": 2,
        "succeeded": 1,
        "failed": 0,
        "last_calls": [{"name": "save_goal", "ok": True}],
    }

    state = shell_state_defaults(
        "admin",
        messages=("hello",),
        activity_log=("saved goal",),
        ui_state=ui_state,
        tool_stats=tool_stats,
        rail_view="Workspace",
    )

    state["ui_state"]["checklist_items"].append("Extra item")
    state["tool_stats"]["last_calls"].append({"name": "new_call"})

    assert state["messages"] == ["hello"]
    assert state["activity_log"] == ["saved goal"]
    assert state["ui_state"]["layout"] == "focus"
    assert state["ui_state"]["selected_chapter"] == "plans"
    assert state["ui_state"]["show_right_panel"] is False
    assert state["selected_chapter"] == "plans"
    assert state["show_right_panel"] is False
    assert state["rail_view"] == "workspace"
    assert state["tool_stats"]["requested"] == 2
    assert state["tool_stats"]["last_calls"][0]["name"] == "save_goal"
    assert tool_stats["last_calls"] == [{"name": "save_goal", "ok": True}]
    assert ui_state["checklist_items"] == ["Capture next step"]


def test_reset_shell_state_updates_mapping_in_place() -> None:
    session_state = {"messages": ["old"], "selected_chapter": "old"}

    result = reset_shell_state(
        session_state,
        "fresh_user",
        messages=("one", "two"),
        ui_state={"layout": "chat", "selected_chapter": "Calendar", "show_right_panel": True},
        rail_view="debug",
    )

    assert session_state is not result
    assert session_state["messages"] == ["one", "two"]
    assert session_state["selected_chapter"] == "calendar"
    assert session_state["show_right_panel"] is True
    assert session_state["rail_view"] == "debug"
    assert session_state["last_user_id"] == "fresh_user"


def test_sync_helpers_mirror_shell_flags_and_selected_chapter() -> None:
    selected_session_state = {"ui_state": {"selected_chapter": "Habits", "show_right_panel": False}}
    mirrored_session_state = {"ui_state": {"selected_chapter": "Plans", "show_right_panel": False}}

    selected = sync_selected_chapter(selected_session_state, "Body")
    mirrored = sync_shell_flags(mirrored_session_state)

    assert selected == "body"
    assert selected_session_state["selected_chapter"] == "body"
    assert selected_session_state["ui_state"]["selected_chapter"] == "body"
    assert mirrored == ("plans", False)
    assert mirrored_session_state["selected_chapter"] == "plans"
    assert mirrored_session_state["show_right_panel"] is False


def test_layout_and_rail_helpers_normalize_and_compute_widths() -> None:
    ui_state = {"layout": "chat"}
    session_state = {"rail_view": "today"}

    assert normalize_layout("focus") == "focus"
    assert normalize_layout("unknown") == "split"
    assert normalize_rail_view("memory") == "memory"
    assert normalize_rail_view("Onboarding") == "onboarding"
    assert normalize_rail_view("unknown") == "today"
    assert layout_column_widths("chat", True) == [2.45, 0.8]
    assert layout_column_widths("focus", True) == [1.3, 1.55]
    assert layout_column_widths("split", True) == [1.9, 1.0]
    assert layout_column_widths("split", False) == [1.0]
    assert sync_ui_layout(ui_state, "Focus") == "focus"
    assert sync_right_panel_visibility(ui_state, "") is False
    assert sync_rail_view(session_state, "Workspace") == "workspace"
