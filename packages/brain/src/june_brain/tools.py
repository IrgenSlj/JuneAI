"""JuneAI native tools exposed to the hand-written loop."""

from __future__ import annotations

import logging
from datetime import UTC
from typing import Annotated, Any

from .config import apply_runtime_preset_switch
from .context_intelligence import (
    build_active_commitments_summary,
    format_active_commitments_summary,
)
from .memory import Memory
from .runtime_privacy import (
    build_runtime_preset_switch_preview,
    build_runtime_privacy_status,
    format_runtime_preset_switch_plan,
    format_runtime_privacy_status,
)
from .tools_base import Inject, tool

logger = logging.getLogger(__name__)

type AgentPayload = dict[str, Any]
type AgentState = dict[str, Any] | None
InjectedAgentState = Annotated[AgentState, Inject]

UI_CHAPTERS = {
    "calendar",
    "gym",
    "food",
    "trips",
    "plans",
    "habits",
    "body",
    "workouts",
    "nutrition",
    "water",
    "dating",
    "family",
    "birthdays",
}


def _memory_for_state(state: AgentState) -> Memory:
    """Resolve memory for the active user."""
    if state is None:
        raise ValueError("Tool execution requires injected agent state.")
    return Memory(state["user_id"])


@tool
def save_journal_entry(
    entry: str,
    state: InjectedAgentState = None,
) -> str:
    """Save an important reflection or conversation note."""
    memory = _memory_for_state(state)
    memory.save_journal(entry)
    return "I've saved that to your journal."


@tool
def get_journal(
    state: InjectedAgentState = None,
) -> str:
    """Retrieve recent journal entries."""
    memory = _memory_for_state(state)
    entries = memory.get_journal(5)
    if not entries:
        return "No journal entries saved yet."
    return "Recent journal entries:\n" + "\n".join(
        f"- {item['timestamp'][:10]}: {item['entry']}"
        for item in entries
    )


@tool
def save_relationship_profile(
    person: str,
    relationship: str,
    summary: str,
    user_needs: str = "",
    cautions: str = "",
    state: InjectedAgentState = None,
) -> str:
    """Save structured context about a person in the user's life."""
    memory = _memory_for_state(state)
    item = memory.save_relationship_profile(
        person=person,
        relationship=relationship,
        summary=summary,
        user_needs=user_needs,
        cautions=cautions,
    )
    return f"I've saved context for {item['person']} ({item['relationship']})."


@tool
def get_relationship_context(
    person: str = "",
    state: InjectedAgentState = None,
) -> str:
    """Retrieve relationship context for one person or all saved people."""
    memory = _memory_for_state(state)
    profiles = memory.get_relationship_profiles(person)
    if not profiles:
        if person:
            return f"No saved relationship context for {person}."
        return "No relationship context saved yet."

    lines = []
    for profile in profiles:
        line = (
            f"- {profile['person']} ({profile['relationship']}): {profile['summary']}"
        )
        if profile.get("user_needs"):
            line += f" | Needs: {profile['user_needs']}"
        if profile.get("cautions"):
            line += f" | Cautions: {profile['cautions']}"
        lines.append(line)
    return "Relationship context:\n" + "\n".join(lines)


@tool
def track_goal(
    title: str,
    category: str = "personal",
    target_date: str = "",
    next_step: str = "",
    status: str = "active",
    state: InjectedAgentState = None,
) -> str:
    """Create or update a goal."""
    memory = _memory_for_state(state)
    goal = memory.save_goal(
        title=title,
        category=category,
        target_date=target_date,
        next_step=next_step,
        status=status,
    )
    return f"I've added '{goal['title']}' to your goals."


@tool
def list_goals(
    status: str = "active",
    state: InjectedAgentState = None,
) -> str:
    """List saved goals, optionally filtered by status."""
    memory = _memory_for_state(state)
    goals = memory.get_goals(status=status)
    if not goals:
        return f"No goals found for status '{status}'."
    lines = []
    for goal in goals:
        line = f"- {goal['title']} [{goal['status']}] ({goal['category']})"
        if goal.get("next_step"):
            line += f" | Next: {goal['next_step']}"
        if goal.get("target_date"):
            line += f" | Target: {goal['target_date']}"
        lines.append(line)
    return "Goals:\n" + "\n".join(lines)


@tool
def update_goal_status(
    title: str,
    status: str,
    state: InjectedAgentState = None,
) -> str:
    """Update an existing goal's status."""
    memory = _memory_for_state(state)
    goal = memory.update_goal_status(title=title, status=status)
    if not goal:
        return f"No goal found with title '{title}'."
    return f"Updated '{goal['title']}' to {goal['status']}."


@tool
def save_open_loop(
    topic: str,
    next_step: str = "",
    due_date: str = "",
    status: str = "open",
    state: InjectedAgentState = None,
) -> str:
    """Track an unresolved issue, follow-up, or decision."""
    memory = _memory_for_state(state)
    item = memory.save_open_loop(
        topic=topic,
        next_step=next_step,
        due_date=due_date,
        status=status,
    )
    return f"I've noted '{item['topic']}' as an open loop."


@tool
def list_open_loops(
    status: str = "open",
    state: InjectedAgentState = None,
) -> str:
    """List unresolved issues or follow-ups."""
    memory = _memory_for_state(state)
    loops = memory.get_open_loops(status=status)
    if not loops:
        return f"No open loops found for status '{status}'."
    lines = []
    for loop in loops:
        line = f"- {loop['topic']} [{loop['status']}]"
        if loop.get("next_step"):
            line += f" | Next: {loop['next_step']}"
        if loop.get("due_date"):
            line += f" | Due: {loop['due_date']}"
        lines.append(line)
    return "Open loops:\n" + "\n".join(lines)


@tool
def update_open_loop_status(
    topic: str,
    status: str,
    state: InjectedAgentState = None,
) -> str:
    """Update an open loop's status."""
    memory = _memory_for_state(state)
    loop = memory.update_open_loop_status(topic=topic, status=status)
    if not loop:
        return f"No open loop found with topic '{topic}'."
    return f"Marked '{loop['topic']}' as {loop['status']}."


@tool
def save_user_preference(
    category: str,
    value: str,
    context: str = "",
    state: InjectedAgentState = None,
) -> str:
    """Save a stable user preference such as favorite genres, routines, or tastes.

    Use this when the user states a clear preference that will help future planning
    or recommendations.
    """
    memory = _memory_for_state(state)
    item = memory.save_preference(category=category, value=value, context=context)
    return f"Got it — I'll remember that about {item['category']}."


@tool
def get_user_preferences(
    category: str = "",
    state: InjectedAgentState = None,
) -> str:
    """Retrieve saved user preferences."""
    memory = _memory_for_state(state)
    preferences = memory.get_preferences(category=category)
    if not preferences:
        if category:
            return f"No preferences saved for category '{category}'."
        return "No user preferences saved yet."
    lines = []
    for item in preferences:
        line = f"- {item['category']}: {item['value']}"
        if item.get("context"):
            line += f" | {item['context']}"
        lines.append(line)
    return "Saved preferences:\n" + "\n".join(lines)


@tool
def save_calendar_item(
    title: str,
    date: str,
    time: str = "",
    details: str = "",
    status: str = "planned",
    source: str = "conversation",
    state: InjectedAgentState = None,
) -> str:
    """Save a calendar item when a concrete plan, event, or reminder appears.

    Prefer ISO dates like YYYY-MM-DD when possible. Use this proactively when the
    conversation produces a real commitment, follow-up, appointment, or trip.
    """
    memory = _memory_for_state(state)
    item = memory.save_calendar_item(
        title=title,
        date=date,
        time=time,
        details=details,
        status=status,
        source=source,
    )
    return f"I've added '{item['title']}' to your calendar on {item['date']}."


@tool
def list_calendar_items(
    status: str = "",
    state: InjectedAgentState = None,
) -> str:
    """List saved calendar items."""
    memory = _memory_for_state(state)
    items = memory.get_calendar_items(status=status)
    if not items:
        return "No calendar items saved yet."
    lines = []
    for item in items:
        line = f"- {item['date']}"
        if item.get("time"):
            line += f" {item['time']}"
        line += f": {item['title']}"
        if item.get("details"):
            line += f" | {item['details']}"
        if item.get("status"):
            line += f" [{item['status']}]"
        lines.append(line)
    return "Calendar:\n" + "\n".join(lines)


@tool
def update_calendar_item_status(
    title: str,
    status: str,
    date: str = "",
    time: str = "",
    state: InjectedAgentState = None,
) -> str:
    """Update a calendar item's status."""
    memory = _memory_for_state(state)
    item = memory.update_calendar_item_status(title=title, status=status, date=date, time=time)
    if not item:
        extra = []
        if date:
            extra.append(f"date '{date}'")
        if time:
            extra.append(f"time '{time}'")
        suffix = f" with {' and '.join(extra)}" if extra else ""
        return f"No calendar item found with title '{title}'{suffix}."
    return f"Updated '{item['title']}' to {item['status']}."


@tool
def save_favorite_recommendation(
    category: str,
    title: str,
    reason: str = "",
    creator: str = "",
    status: str = "saved",
    state: InjectedAgentState = None,
) -> str:
    """Save a movie, book, show, or other recommendation to the favorites shelf.

    Use this when the user asks for recommendations, reacts positively to one,
    or wants the assistant to keep track of their media taste.
    """
    memory = _memory_for_state(state)
    item = memory.save_favorite(
        category=category,
        title=title,
        reason=reason,
        creator=creator,
        status=status,
    )
    return f"I've saved '{item['title']}' to your {item['category']} list."


@tool
def list_favorites(
    category: str = "",
    state: InjectedAgentState = None,
) -> str:
    """List saved favorites and recommendations."""
    memory = _memory_for_state(state)
    items = memory.get_favorites(category=category)
    if not items:
        if category:
            return f"No saved favorites for category '{category}'."
        return "No favorites saved yet."
    lines = []
    for item in items:
        line = f"- {item['category']}: {item['title']}"
        if item.get("creator"):
            line += f" by {item['creator']}"
        if item.get("reason"):
            line += f" | {item['reason']}"
        lines.append(line)
    return "Favorites:\n" + "\n".join(lines)


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
def get_active_commitments_summary(
    state: InjectedAgentState = None,
) -> str:
    """Get June's unified active commitments summary."""
    memory = _memory_for_state(state)
    summary = build_active_commitments_summary(memory)
    return format_active_commitments_summary(summary)


@tool
def get_personal_context(
    topic: str,
    state: InjectedAgentState = None,
) -> str:
    """Return a natural-language summary of what June knows about a given topic.

    Use this when giving contextual advice — not just acknowledging, but actually knowing.
    Topics: training, sleep, nutrition, goals, relationships, calendar, habits, body, general.
    """
    memory = _memory_for_state(state)
    topic_lower = topic.strip().lower()
    parts: list[str] = []

    if any(t in topic_lower for t in ("training", "gym", "workout", "fitness", "exercise")):
        plans = memory.get_gym_plans()
        sessions = memory.get_workout_sessions(limit=3)
        if plans:
            p = plans[0]
            parts.append(f"Gym plan: {p.get('title','')} — {p.get('structure','')}, {p.get('frequency','')}.")
        if sessions:
            last = sessions[0]
            parts.append(
                f"Last session: {last.get('plan_name','')} on {last.get('date','')} "
                f"({last.get('duration_min',0)} min, energy {last.get('energy_rating',0)}/5)."
            )
        if not parts:
            parts.append("No training plan or sessions logged yet.")

    if any(t in topic_lower for t in ("sleep", "recovery", "rest", "energy")):
        metrics = memory.get_body_metrics(days=7)
        if metrics:
            recent = metrics[0]
            sleep = recent.get("sleep_hours")
            energy = recent.get("energy")
            stress = recent.get("stress")
            m_parts = []
            if sleep:
                m_parts.append(f"sleep {sleep}h")
            if energy:
                m_parts.append(f"energy {energy}/5")
            if stress:
                m_parts.append(f"stress {stress}/5")
            if m_parts:
                parts.append(f"Recent body metrics ({recent.get('date','')}): {', '.join(m_parts)}.")
        else:
            parts.append("No body metrics logged yet.")

    if any(t in topic_lower for t in ("nutrition", "food", "eating", "diet", "meal")):
        programs = memory.get_food_programs()
        meals_today = memory.get_nutrition_today()
        if programs:
            f_prog = programs[0]
            parts.append(f"Nutrition approach: {f_prog.get('title','')} — {f_prog.get('approach','')}.")
        if meals_today:
            cals = sum(int(m.get("calories_est") or 0) for m in meals_today)
            prot = sum(int(m.get("protein_est") or 0) for m in meals_today)
            parts.append(f"Today: {len(meals_today)} meal(s) logged, ~{cals} kcal, ~{prot}g protein.")
        elif not programs:
            parts.append("No nutrition program or meals logged yet.")

    if any(t in topic_lower for t in ("goal", "plan", "target", "objective")):
        goals = memory.get_goals(status="active", limit=5)
        loops = memory.get_open_loops(status="open", limit=3)
        if goals:
            g_lines = []
            for g in goals:
                line = g.get("title", "")
                if g.get("next_step"):
                    line += f" — next: {g['next_step']}"
                if g.get("target_date"):
                    line += f" (by {g['target_date']})"
                g_lines.append(line)
            parts.append("Active goals: " + "; ".join(g_lines) + ".")
        if loops:
            parts.append("Open loops: " + "; ".join(loop.get("topic", "") for loop in loops) + ".")
        if not goals and not loops:
            parts.append("No goals or open loops saved yet.")

    if any(t in topic_lower for t in ("relationship", "family", "partner", "people", "social")):
        profiles = memory.get_relationship_profiles()
        if profiles:
            r_lines = []
            for p in profiles[:4]:
                r_lines.append(f"{p.get('person','')} ({p.get('relationship','')}): {p.get('summary','')}")
            parts.append("Relationships: " + " | ".join(r_lines) + ".")
        else:
            parts.append("No relationship context saved yet.")

    if any(t in topic_lower for t in ("calendar", "upcoming", "schedule", "event", "appointment")):
        from datetime import date
        today = date.today()
        items = memory.get_calendar_items(limit=10)
        upcoming = [
            i for i in items
            if i.get("date", "") >= today.isoformat()
            and (i.get("status") or "").lower() not in {"done", "completed", "cancelled"}
        ][:5]
        if upcoming:
            parts.append(
                "Upcoming: " + "; ".join(
                    f"{i.get('title','')} on {i.get('date','')}" for i in upcoming
                ) + "."
            )
        else:
            parts.append("No upcoming calendar items.")

    if any(t in topic_lower for t in ("habit", "routine", "daily")):
        habits = memory.get_habits()
        if habits:
            done = [h for h in habits if h.get("done_today")]
            pending = [h for h in habits if not h.get("done_today")]
            habit_parts = []
            if done:
                habit_parts.append(f"Done today: {', '.join(h['name'] for h in done)}")
            if pending:
                habit_parts.append(f"Pending: {', '.join(h['name'] for h in pending)}")
            parts.append("Habits: " + "; ".join(habit_parts) + ".")
        else:
            parts.append("No habits tracked yet.")

    if any(t in topic_lower for t in ("weight", "body", "metrics", "steps", "heart rate")):
        metrics = memory.get_body_metrics(days=7)
        if metrics:
            recent = metrics[0]
            m_parts = []
            if recent.get("weight_kg"):
                m_parts.append(f"weight {recent['weight_kg']}kg")
            if recent.get("sleep_hours"):
                m_parts.append(f"sleep {recent['sleep_hours']}h")
            if recent.get("energy"):
                m_parts.append(f"energy {recent['energy']}/5")
            if recent.get("steps"):
                m_parts.append(f"steps {recent['steps']}")
            if m_parts:
                parts.append(f"Body ({recent.get('date','')}): {', '.join(m_parts)}.")
        else:
            parts.append("No body metrics logged yet.")

    if not parts or "general" in topic_lower:
        # Provide a broad overview
        c = memory.get_chapter_completeness()
        filled = [k for k, v in c.items() if v > 0]
        empty = [k for k, v in c.items() if v == 0]
        if filled:
            parts.append(f"Chapters with data: {', '.join(filled)}.")
        if empty:
            parts.append(f"Chapters with no data yet: {', '.join(empty)}.")

    return "\n".join(parts) if parts else f"Nothing saved about '{topic}' yet."


@tool
def run_diagnostics() -> str:
    """Run system diagnostics: check providers, memory, tools, skills, router, and scheduler. Reports what works and what doesn't."""
    from june_brain.self_test import format_markdown, run_all

    results = run_all()
    return format_markdown(results)


# Trimmed tool set for small local models (gemma4 4B, etc.).
# Keeps write-heavy capture tools and one summary read.
# Drops list_*, weekly_summary, and multi-step reasoning tools
# that inflate the tool-schema tokens and hurt small-model reliability.
JUNE_TOOLS_GEMMA = [
    save_journal_entry,
    save_relationship_profile,
    track_goal,
    update_goal_status,
    save_open_loop,
    update_open_loop_status,
    save_calendar_item,
    update_calendar_item_status,
    save_user_preference,
    save_favorite_recommendation,
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
})


JUNE_TOOLS = [
    save_journal_entry,
    get_journal,
    save_relationship_profile,
    get_relationship_context,
    track_goal,
    list_goals,
    update_goal_status,
    save_open_loop,
    list_open_loops,
    update_open_loop_status,
    save_user_preference,
    get_user_preferences,
    save_calendar_item,
    list_calendar_items,
    update_calendar_item_status,
    save_favorite_recommendation,
    list_favorites,
    get_active_commitments_summary,
    get_runtime_privacy_status,
    preview_runtime_preset_switch,
    switch_runtime_preset,
    draft_reply,
    get_personal_context,
    create_schedule,
    list_schedules,
    delete_schedule,
    run_diagnostics,
]
