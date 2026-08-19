"""JuneAI native tools exposed to the hand-written loop."""

from __future__ import annotations

import logging
from datetime import UTC
from typing import Annotated, Any

from .config import apply_runtime_preset_switch
from .runtime_privacy import (
    build_runtime_preset_switch_preview,
    build_runtime_privacy_status,
    format_runtime_preset_switch_plan,
    format_runtime_privacy_status,
)
from .tools_base import Inject, tool
from .tools_memory import forget, list_promises, remember, update_promise

logger = logging.getLogger(__name__)

type AgentPayload = dict[str, Any]
type AgentState = dict[str, Any] | None
InjectedAgentState = Annotated[AgentState, Inject]


@tool
def get_runtime_privacy_status() -> str:
    """Get the current runtime and privacy mode for the active model configuration."""
    status = build_runtime_privacy_status()
    return format_runtime_privacy_status(status)


@tool
def preview_runtime_preset_switch(
    preset_key: str,
) -> str:
    """Preview a runtime preset switch without mutating the current process."""
    plan = build_runtime_preset_switch_preview(preset_key)
    return format_runtime_preset_switch_plan(plan)


@tool
def switch_runtime_preset(
    preset_key: str,
    confirm_api_transition: bool = False,
) -> str:
    """Apply a runtime preset switch safely, requiring confirmation for local-to-API transitions."""
    result = apply_runtime_preset_switch(
        preset_key,
        confirm_api_transition=confirm_api_transition,
    )
    return format_runtime_preset_switch_plan(result)


@tool
def draft_reply(
    recipient: str,
    context: str,
    tone: str = "warm",
    goal: str = "",
) -> str:
    """Prepare a message draft for the user."""
    return (
        f"Draft a reply to {recipient}.\n\n"
        f"Context: {context}\n"
        f"Tone: {tone}\n"
        f"Goal: {goal or 'Maintain clarity and forward motion'}\n\n"
        "Write one strong draft and two shorter alternatives."
    )



@tool
def run_diagnostics() -> str:
    """Run system diagnostics: check providers, memory, tools, skills, router, and scheduler. Reports what works and what doesn't."""
    from june_brain.self_test import format_markdown, run_all

    results = run_all()
    return format_markdown(results)


# Trimmed tool set for small local models (gemma4 4B, etc.). Fewer tools, and
# fewer tools whose descriptions overlap, is the whole reliability story here:
# a small model picking between near-synonyms is where wrong calls come from.
# The four memory tools (ADR 0032) lead because they are what the product is;
# `list_*` and multi-step reads stay out because they inflate the schema without
# adding a capability the model needs to reach on its own.
JUNE_TOOLS_GEMMA = [
    remember,
    forget,
    list_promises,
    update_promise,
    run_diagnostics,
]

# ---------------------------------------------------------------------------
# Scheduler tools
# ---------------------------------------------------------------------------


@tool
def create_schedule(
    name: str,
    description: str = "",
    cron_expression: str = "",
    interval_seconds: int = 0,
    action_type: str = "agent_invoke",
    action_prompt: str = "",
    state: InjectedAgentState = None,
) -> str:
    """Create a recurring or one-shot schedule. Use cron (e.g., '0 8 * * *') or interval_seconds for recurring schedules."""
    from june_brain.memory.sqlite import _get_connection, db_path
    from june_brain.scheduler.models import Schedule as _Schedule
    from june_brain.scheduler.store import ScheduleStore

    user_id = (state or {}).get("user_id", "default")
    conn = _get_connection(db_path())
    from june_brain.scheduler.models import _SCHEDULES_TABLE_SQL

    conn.executescript(_SCHEDULES_TABLE_SQL)
    conn.commit()
    store = ScheduleStore(conn)

    from datetime import datetime

    sched = _Schedule(
        user_id=user_id,
        name=name,
        description=description,
        cron_expression=cron_expression,
        interval_seconds=interval_seconds,
        scheduled_at=datetime.now(UTC).isoformat(),
        action_type=action_type,
        action_config={"prompt": action_prompt} if action_prompt else {},
    )
    store.create(sched)
    return f"Scheduled '{name}' to run every {interval_seconds}s." if interval_seconds else f"Scheduled '{name}' with cron '{cron_expression}'."


@tool
def list_schedules(
    state: InjectedAgentState = None,
) -> str:
    """List all active schedules."""
    from june_brain.memory.sqlite import _get_connection, db_path
    from june_brain.scheduler.models import _SCHEDULES_TABLE_SQL
    from june_brain.scheduler.store import ScheduleStore

    user_id = (state or {}).get("user_id", "default")
    conn = _get_connection(db_path())
    conn.executescript(_SCHEDULES_TABLE_SQL)
    conn.commit()
    store = ScheduleStore(conn)
    schedules = store.list(user_id)
    if not schedules:
        return "No schedules."
    lines = []
    for s in schedules:
        lines.append(f"- {s.name} ({'enabled' if s.enabled else 'disabled'}) next: {s.scheduled_at[:10]}")
    return "Schedules:\n" + "\n".join(lines)


@tool
def delete_schedule(
    schedule_id: str,
    state: InjectedAgentState = None,
) -> str:
    """Delete a schedule by its ID."""
    from june_brain.memory.sqlite import _get_connection, db_path
    from june_brain.scheduler.models import _SCHEDULES_TABLE_SQL
    from june_brain.scheduler.store import ScheduleStore

    conn = _get_connection(db_path())
    conn.executescript(_SCHEDULES_TABLE_SQL)
    conn.commit()
    store = ScheduleStore(conn)
    if store.delete(schedule_id):
        return f"Schedule {schedule_id} deleted."
    return f"Schedule {schedule_id} not found."

# Tool names retired with the v1 domain layer (D.5a). Kept as a denylist because
# deleting a tool does not durably remove the capability: `_select_tools_for_runtime`
# drops a skill's tool only when a *native* tool shadows the name, so removing the
# native copy unshadows whatever a skill declares. That is not hypothetical — it
# happened during D.5a, where skills/health and skills/daily silently took over
# six names the native registry had just dropped.
#
# A name here stays gone no matter who advertises it. Add to this set whenever a
# tool is retired rather than replaced; remove from it only when the name is
# deliberately brought back.
RETIRED_TOOL_NAMES = frozenset({
    # health and fitness
    "save_gym_plan", "list_gym_plans", "save_food_program", "list_food_programs",
    "log_workout_session", "log_body_metrics", "create_habit",
    "log_habit_completion", "get_habits_with_streaks", "log_nutrition",
    "log_water", "get_today_summary", "get_recovery_readiness_summary",
    "summarize_progress",
    # mood tracking — the behavioral floor says June is not a therapist
    "log_mood", "get_mood_history",
    # chapters
    "check_chapter_completeness", "ask_about_chapter", "generate_weekly_summary",
    # conversation coaching
    "analyze_compatibility", "generate_conversation_starters",
    "plan_difficult_conversation",
    # the no-op workspace panel
    "set_ui_focus", "set_ui_checklist", "set_ui_layout", "set_ui_chapter",
    "clear_ui_workspace",
    # v1 domain writers, replaced by the four deliberate memory tools (ADR 0032)
    "save_journal_entry", "get_journal",
    "save_relationship_profile", "get_relationship_context",
    "track_goal", "list_goals", "update_goal_status",
    "save_open_loop", "list_open_loops", "update_open_loop_status",
    "save_user_preference", "get_user_preferences",
    "save_favorite_recommendation", "list_favorites",
})


# Names deleted from the native registry that a bundled skill is *meant* to
# take over. This is the third state RETIRED_TOOL_NAMES cannot express, and the
# distinction is the whole point: removing a native tool unshadows the skill's
# copy of the name, which was a bug in D.5c (skills/health and skills/daily
# silently resurrected six deleted capabilities) and is the intent here.
#
# Calendar is a supported capability, not v1 residue — a default-enabled MCP
# skill with a declared scope contract, and the Time track in ROADMAP.md builds
# on it. So the native copies go as duplicates and the skill becomes the one
# implementation. A name belongs here only when a bundled skill actually
# advertises it; `test_a_handoff_actually_hands_off` is what checks that.
SKILL_OWNED_TOOL_NAMES = frozenset({
    "save_calendar_item", "list_calendar_items", "update_calendar_item_status",
})


JUNE_TOOLS = [
    remember,
    forget,
    list_promises,
    update_promise,
    get_runtime_privacy_status,
    preview_runtime_preset_switch,
    switch_runtime_preset,
    draft_reply,
    create_schedule,
    list_schedules,
    delete_schedule,
    run_diagnostics,
]
