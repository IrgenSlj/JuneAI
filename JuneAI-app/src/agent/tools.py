"""JuneAI tools exposed to the LangGraph agent."""

from __future__ import annotations

from typing import Annotated, Any, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import TypeAlias

from .context_intelligence import (
    build_active_commitments_summary,
    build_recovery_readiness_summary,
    format_active_commitments_summary,
    format_recovery_readiness_summary,
)
from .config import apply_runtime_preset_switch
from .memory import Memory
from .runtime_privacy import (
    build_runtime_privacy_status,
    build_runtime_preset_switch_preview,
    format_runtime_privacy_status,
    format_runtime_preset_switch_plan,
)

AgentPayload: TypeAlias = dict[str, Any]
AgentState: TypeAlias = Optional[AgentPayload]
InjectedAgentState = Annotated[AgentState, InjectedState]
ToolCommand: TypeAlias = Command[str]

DEFAULT_UI_STATE: dict[str, Any] = {
    "layout": "split",
    "show_right_panel": True,
    "selected_chapter": "",
    "focus_title": "Workspace",
    "focus_body": "June can pin structured notes, plans, and highlights here.",
    "checklist_title": "Next steps",
    "checklist_items": [],
    "notice": "",
}

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


def _merge_ui_state(current: Optional[dict[str, Any]], updates: dict[str, Any]) -> dict[str, Any]:
    """Merge UI state updates onto defaults and current state."""
    merged = dict(DEFAULT_UI_STATE)
    if current:
        merged.update(current)
    merged.update(updates)
    return merged


@tool
def log_mood(
    mood: str,
    note: str = "",
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Log the user's emotional state when they describe how they feel."""
    memory = _memory_for_state(state)
    entry = memory.log_mood(mood, note)
    return f"Noted your mood as '{mood}'."


@tool
def get_mood_history(
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Retrieve recent mood history for pattern recognition."""
    memory = _memory_for_state(state)
    moods = memory.get_mood_history(10)
    if not moods:
        return "No mood history recorded yet."
    lines = [
        f"- {item['timestamp'][:10]}: {item['mood']}"
        + (f" | {item['note']}" if item.get("note") else "")
        for item in moods
    ]
    return "Recent mood history:\n" + "\n".join(lines)


@tool
def save_journal_entry(
    entry: str,
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Save an important reflection or conversation note."""
    memory = _memory_for_state(state)
    item = memory.save_journal(entry)
    return "I've saved that to your journal."


@tool
def get_journal(
    state: Annotated[AgentState, InjectedState] = None,
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
    state: Annotated[AgentState, InjectedState] = None,
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
    state: Annotated[AgentState, InjectedState] = None,
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
    state: Annotated[AgentState, InjectedState] = None,
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
    state: Annotated[AgentState, InjectedState] = None,
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
    state: Annotated[AgentState, InjectedState] = None,
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
    state: Annotated[AgentState, InjectedState] = None,
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
    state: Annotated[AgentState, InjectedState] = None,
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
    state: Annotated[AgentState, InjectedState] = None,
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
    state: Annotated[AgentState, InjectedState] = None,
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
    state: Annotated[AgentState, InjectedState] = None,
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
    state: Annotated[AgentState, InjectedState] = None,
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
    state: Annotated[AgentState, InjectedState] = None,
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
    state: Annotated[AgentState, InjectedState] = None,
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
    state: Annotated[AgentState, InjectedState] = None,
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
    state: Annotated[AgentState, InjectedState] = None,
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
def save_gym_plan(
    name: str,
    schedule: str,
    goal: str = "",
    notes: str = "",
    status: str = "active",
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Save a workout schedule or gym program for the user."""
    memory = _memory_for_state(state)
    item = memory.save_gym_plan(
        name=name,
        schedule=schedule,
        goal=goal,
        notes=notes,
        status=status,
    )
    return f"I've saved your gym plan '{item['name']}'."


@tool
def list_gym_plans(
    status: str = "active",
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """List saved workout schedules and gym programs."""
    memory = _memory_for_state(state)
    plans = memory.get_gym_plans(status=status)
    if not plans:
        return f"No gym plans found for status '{status}'."
    lines = []
    for plan in plans:
        line = f"- {plan['name']} [{plan['status']}] | Schedule: {plan['schedule']}"
        if plan.get("goal"):
            line += f" | Goal: {plan['goal']}"
        if plan.get("notes"):
            line += f" | Notes: {plan['notes']}"
        lines.append(line)
    return "Gym plans:\n" + "\n".join(lines)


@tool
def save_food_program(
    name: str,
    goal: str,
    daily_structure: str,
    notes: str = "",
    status: str = "active",
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Save a meal plan or nutrition program."""
    memory = _memory_for_state(state)
    item = memory.save_food_program(
        name=name,
        goal=goal,
        daily_structure=daily_structure,
        notes=notes,
        status=status,
    )
    return f"I've saved your food program '{item['name']}'."


@tool
def list_food_programs(
    status: str = "active",
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """List saved meal plans and nutrition programs."""
    memory = _memory_for_state(state)
    programs = memory.get_food_programs(status=status)
    if not programs:
        return f"No food programs found for status '{status}'."
    lines = []
    for program in programs:
        line = (
            f"- {program['name']} [{program['status']}] | Goal: {program['goal']}"
            f" | Structure: {program['daily_structure']}"
        )
        if program.get("notes"):
            line += f" | Notes: {program['notes']}"
        lines.append(line)
    return "Food programs:\n" + "\n".join(lines)


@tool
def summarize_progress(
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Summarize the user's recent activity across assistant surfaces."""
    memory = _memory_for_state(state)
    snapshot = memory.get_progress_snapshot()
    parts = [
        f"Recent moods logged: {snapshot['mood_count']}",
        f"Journal entries: {snapshot['journal_count']}",
        f"Goals tracked: {snapshot['goal_count']}",
        f"Open loops: {snapshot['open_loop_count']}",
        f"Preferences saved: {snapshot['preference_count']}",
        f"Calendar items: {snapshot['calendar_count']}",
        f"Favorites saved: {snapshot['favorite_count']}",
        f"Gym plans: {snapshot['gym_plan_count']}",
        f"Food programs: {snapshot['food_program_count']}",
    ]
    if snapshot["latest_mood"]:
        parts.insert(1, f"Latest mood: {snapshot['latest_mood']}")
    return "Progress snapshot:\n- " + "\n- ".join(parts)


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
def analyze_compatibility(person1_description: str, person2_description: str) -> str:
    """Structure a compatibility analysis for two people."""
    return (
        "Compatibility analysis request:\n\n"
        f"Person 1: {person1_description}\n\n"
        f"Person 2: {person2_description}\n\n"
        "Provide shared strengths, friction points, communication style, and a score out of 10."
    )


@tool
def generate_conversation_starters(context: str) -> str:
    """Generate specific conversation starters for the given context."""
    return (
        f"Generate 5 specific conversation starters for this context: {context}\n\n"
        "Keep them natural, observant, and non-generic."
    )


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
def plan_difficult_conversation(
    person: str,
    situation: str,
    desired_outcome: str,
) -> str:
    """Plan a difficult conversation with structure and next steps."""
    return (
        f"Plan a difficult conversation with {person}.\n\n"
        f"Situation: {situation}\n"
        f"Desired outcome: {desired_outcome}\n\n"
        "Provide key points, likely friction, what to avoid, and one strong opening line."
    )


@tool
def set_ui_focus(
    title: str,
    body: str,
    footer: str = "",
    state: InjectedAgentState = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> ToolCommand:
    """Update the workspace focus panel with a title and body."""
    next_ui_state = _merge_ui_state(
        state.get("ui_state", {}) if state else {},
        {
            "focus_title": title.strip() or "Workspace",
            "focus_body": body.strip(),
            "notice": footer.strip(),
        },
    )
    return Command(update={
        "ui_state": next_ui_state,
        "messages": [
            ToolMessage(
                content=f"Workspace focus updated to '{next_ui_state['focus_title']}'.",
                tool_call_id=tool_call_id,
            )
        ],
    })


@tool
def set_ui_checklist(
    title: str,
    items: str,
    state: InjectedAgentState = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> ToolCommand:
    """Update the workspace checklist with newline-separated items."""
    checklist_items = [
        item.strip("- ").strip()
        for item in items.splitlines()
        if item.strip()
    ]
    next_ui_state = _merge_ui_state(
        state.get("ui_state", {}) if state else {},
        {
            "checklist_title": title.strip() or "Next steps",
            "checklist_items": checklist_items,
        },
    )
    return Command(update={
        "ui_state": next_ui_state,
        "messages": [
            ToolMessage(
                content=f"Workspace checklist updated with {len(checklist_items)} items.",
                tool_call_id=tool_call_id,
            )
        ],
    })


@tool
def set_ui_layout(
    layout: str,
    notice: str = "",
    state: InjectedAgentState = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> ToolCommand:
    """Set the workspace layout mode. Allowed values: split, focus, chat."""
    chosen = layout.strip().lower()
    if chosen not in {"split", "focus", "chat"}:
        chosen = "split"
    next_ui_state = _merge_ui_state(
        state.get("ui_state", {}) if state else {},
        {"layout": chosen, "notice": notice.strip()},
    )
    return Command(update={
        "ui_state": next_ui_state,
        "messages": [
            ToolMessage(
                content=f"Workspace layout set to '{chosen}'.",
                tool_call_id=tool_call_id,
            )
        ],
    })


@tool
def set_ui_chapter(
    chapter: str = "",
    notice: str = "",
    state: InjectedAgentState = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> ToolCommand:
    """Open a chapter in the single-page right panel, or clear it to return to workspace."""
    chosen = chapter.strip().lower()
    if chosen and chosen not in UI_CHAPTERS:
        chosen = ""
    next_ui_state = _merge_ui_state(
        state.get("ui_state", {}) if state else {},
        {"selected_chapter": chosen, "notice": notice.strip()},
    )
    label = chosen or "workspace"
    return Command(update={
        "ui_state": next_ui_state,
        "messages": [
            ToolMessage(
                content=f"UI focus switched to '{label}'.",
                tool_call_id=tool_call_id,
            )
        ],
    })


@tool
def clear_ui_workspace(
    state: InjectedAgentState = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> ToolCommand:
    """Reset the workspace panel to its default state."""
    next_ui_state = _merge_ui_state(
        state.get("ui_state", {}) if state else {},
        DEFAULT_UI_STATE,
    )
    return Command(update={
        "ui_state": next_ui_state,
        "messages": [
            ToolMessage(
                content="Workspace reset to its default state.",
                tool_call_id=tool_call_id,
            )
        ],
    })


@tool
def log_workout_session(
    plan_name: str,
    exercises: str = "",
    duration_min: int = 0,
    notes: str = "",
    energy_rating: int = 0,
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Log a completed workout session with exercises and duration in minutes."""
    memory = _memory_for_state(state)
    item = memory.log_workout_session(
        plan_name=plan_name,
        exercises=exercises,
        duration_min=duration_min,
        notes=notes,
        energy_rating=energy_rating,
    )
    return f"Logged your {item['plan_name']} session on {item['date']} ({item['duration_min']} min)."


@tool
def log_body_metrics(
    weight_kg: float = 0.0,
    sleep_hours: float = 0.0,
    sleep_quality: int = 0,
    energy: int = 0,
    stress: int = 0,
    soreness: int = 0,
    resting_hr: int = 0,
    steps: int = 0,
    notes: str = "",
    state: InjectedAgentState = None,
) -> str:
    """Log today's body metrics: recovery, strain, and activity signals."""
    memory = _memory_for_state(state)
    item = memory.log_body_metrics(
        weight_kg=weight_kg,
        sleep_hours=sleep_hours,
        sleep_quality=sleep_quality,
        energy=energy,
        stress=stress,
        soreness=soreness,
        resting_hr=resting_hr,
        steps=steps,
        notes=notes,
    )
    parts = []
    if item["weight_kg"]:
        parts.append(f"weight {item['weight_kg']}kg")
    if item["sleep_hours"]:
        parts.append(f"sleep {item['sleep_hours']}h")
    if item["sleep_quality"]:
        parts.append(f"sleep quality {item['sleep_quality']}/5")
    if item["energy"]:
        parts.append(f"energy {item['energy']}/5")
    if item["stress"]:
        parts.append(f"stress {item['stress']}/5")
    if item["soreness"]:
        parts.append(f"soreness {item['soreness']}/5")
    if item["resting_hr"]:
        parts.append(f"resting HR {item['resting_hr']}")
    if item["steps"]:
        parts.append(f"steps {item['steps']}")
    return f"Got it — logged {', '.join(parts) or 'your metrics'} for today."


@tool
def create_habit(
    name: str,
    category: str = "health",
    target_days: str = "daily",
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Create or update a tracked habit. Category: health, sport, focus, nutrition."""
    memory = _memory_for_state(state)
    item = memory.create_or_update_habit(name=name, category=category, target_days=target_days)
    return f"I'm now tracking '{item['name']}' for you."


@tool
def log_habit_completion(
    habit_name: str,
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Mark a habit as completed for today. Auto-creates the habit if it does not exist."""
    memory = _memory_for_state(state)
    item = memory.log_habit_completion(habit_name=habit_name)
    streak = item.get("streak", 0)
    return f"'{item['name']}' done. {streak}-day streak." if streak > 1 else f"'{item['name']}' checked off."


@tool
def get_habits_with_streaks(
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Get all tracked habits with current streaks and today's completion status."""
    memory = _memory_for_state(state)
    habits = memory.get_habits()
    if not habits:
        return "No habits tracked yet. Use create_habit to add one."
    lines = []
    for h in habits:
        status = "done" if h.get("done_today") else "pending"
        lines.append(
            f"- {h['name']} [{status}] streak: {h.get('streak', 0)} days"
            f" | {h.get('category', '')} | {h.get('target_days', '')}"
        )
    return "Habits:\n" + "\n".join(lines)


@tool
def log_nutrition(
    meal: str,
    description: str,
    calories_est: int = 0,
    protein_est: int = 0,
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Log a meal. Meal should be: breakfast, lunch, dinner, or snack. Estimates in kcal/g."""
    memory = _memory_for_state(state)
    item = memory.log_nutrition(
        meal=meal,
        description=description,
        calories_est=calories_est,
        protein_est=protein_est,
    )
    return f"I've logged your {item['meal']}: {item['description']}."


@tool
def log_water(
    glasses: int = 1,
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Log glasses of water consumed today. Increments today's running count."""
    memory = _memory_for_state(state)
    total = memory.log_water(glasses)
    return f"Water logged — {total} glass(es) today."


@tool
def get_today_summary(
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Get a full summary of today: habits, workout, body metrics, water, and nutrition."""
    memory = _memory_for_state(state)
    s = memory.get_today_summary()
    parts = [f"Today ({s['date']}):"]
    if s["habits_total"] > 0:
        parts.append(f"Habits: {s['habits_done']}/{s['habits_total']} done")
        if s["habits_done_names"]:
            parts.append(f"  Done: {', '.join(s['habits_done_names'])}")
        if s["habits_pending_names"]:
            parts.append(f"  Pending: {', '.join(s['habits_pending_names'])}")
    parts.append(f"Water: {s['water_glasses']} glasses")
    if s["body_metrics"]:
        m = s["body_metrics"]
        metric_parts = []
        if m.get("weight_kg"):
            metric_parts.append(f"weight {m['weight_kg']}kg")
        if m.get("sleep_hours"):
            metric_parts.append(f"sleep {m['sleep_hours']}h")
        if m.get("sleep_quality"):
            metric_parts.append(f"sleep quality {m['sleep_quality']}/5")
        if m.get("energy"):
            metric_parts.append(f"energy {m['energy']}/5")
        if m.get("stress"):
            metric_parts.append(f"stress {m['stress']}/5")
        if m.get("soreness"):
            metric_parts.append(f"soreness {m['soreness']}/5")
        if m.get("resting_hr"):
            metric_parts.append(f"resting HR {m['resting_hr']}")
        if m.get("steps"):
            metric_parts.append(f"steps {m['steps']}")
        if metric_parts:
            parts.append("Body: " + ", ".join(metric_parts))
    else:
        parts.append("Body metrics: not logged today")
    if s["workout"]:
        parts.append(f"Workout: {s['workout']['plan_name']} ({s['workout']['duration_min']} min)")
    else:
        parts.append("Workout: not logged today")
    if s["meals_logged"] > 0:
        parts.append(f"Nutrition: {s['meals_logged']} meals, ~{s['calories_est']} kcal, ~{s['protein_est']}g protein")
    return "\n".join(parts)


@tool
def get_recovery_readiness_summary(
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Get June's derived recovery and readiness summary."""
    memory = _memory_for_state(state)
    summary = build_recovery_readiness_summary(memory)
    return format_recovery_readiness_summary(summary)


@tool
def get_active_commitments_summary(
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Get June's unified active commitments summary."""
    memory = _memory_for_state(state)
    summary = build_active_commitments_summary(memory)
    return format_active_commitments_summary(summary)


@tool
def check_chapter_completeness(
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Check which of June's chapters have data and which are empty.

    Use this at the start of a conversation or when deciding what to ask about next.
    Chapters: Calendar, Gym Schedule, Food Schedule, Birthdays, Trips,
    Goals & Plans, Dating/Love, Family, Habits, Body Metrics.
    """
    memory = _memory_for_state(state)
    c = memory.get_chapter_completeness()
    empty = memory.get_chapters_needing_attention()

    rows = [
        ("Calendar",       c["calendar"]),
        ("Birthdays",      c["birthdays"]),
        ("Trips",          c["trips"]),
        ("Gym Schedule",   c["gym"]),
        ("Food Schedule",  c["food"]),
        ("Goals",          c["goals"]),
        ("Open Loops",     c["open_loops"]),
        ("Dating/Love",    c["dating"]),
        ("Family",         c["family"]),
        ("Habits",         c["habits"]),
        ("Body Metrics",   c["body_metrics"]),
        ("Workout Sessions", c["workout_sessions"]),
    ]
    lines = ["Chapter completeness:"]
    for label, count in rows:
        status = "EMPTY — needs data" if count == 0 else f"{count} item(s)"
        lines.append(f"  {label}: {status}")

    if empty:
        lines.append(f"\nEmpty chapters: {', '.join(empty)}")
        lines.append("Ask the user about these to keep their profile complete.")
    else:
        lines.append("\nAll chapters have at least some data.")
    return "\n".join(lines)


_CHAPTER_INTAKE_PROMPTS: dict[str, str] = {
    "calendar": (
        "Ask the user:\n"
        "- What events, appointments, or commitments are coming up in the next 2-4 weeks?\n"
        "- Any recurring weekly commitments (work, classes, regular plans)?\n"
        "- Any important personal dates or annual events?\n"
        "Save each confirmed item with save_calendar_item (date YYYY-MM-DD, title, details)."
    ),
    "gym": (
        "Ask the user:\n"
        "- What does your training week look like? (push/pull/legs, upper/lower, full body, etc.)\n"
        "- How many days per week and which days?\n"
        "- What is your main training goal? (strength, muscle, fat loss, endurance)\n"
        "- Any specific program or methodology you follow?\n"
        "Save the full weekly structure with save_gym_plan."
    ),
    "food": (
        "Ask the user:\n"
        "- What is your current nutrition approach? (high protein, deficit, maintenance, intuitive)\n"
        "- What does a typical day of eating look like?\n"
        "- Any dietary restrictions, intolerances, or foods you avoid?\n"
        "- What are your main nutrition goals?\n"
        "Save the structure and goal with save_food_program."
    ),
    "birthdays": (
        "Ask the user:\n"
        "- Whose birthdays do you want to remember? (family, partner, close friends)\n"
        "- For each person: name, relationship, and birth date (MM-DD is fine if year unknown)\n"
        "- Any anniversaries or recurring personal dates to add?\n"
        "Save each with save_calendar_item — include 'birthday' in the title."
    ),
    "trips": (
        "Ask the user:\n"
        "- Any travel planned in the next 3-6 months?\n"
        "- For each: destination, rough dates, and purpose (holiday, work, family visit)\n"
        "- Any regular travel patterns for work or personal reasons?\n"
        "Save each with save_calendar_item — include 'trip' or 'travel' in the details."
    ),
    "plans": (
        "Ask the user:\n"
        "- What are your 2-3 most important goals right now? (health, career, personal growth)\n"
        "- What is the next concrete step for each goal?\n"
        "- Is there anything you have been meaning to do or follow up on?\n"
        "- Any decisions you are sitting on?\n"
        "Save goals with track_goal (category, next_step). Save follow-ups with save_open_loop."
    ),
    "dating": (
        "Ask the user:\n"
        "- Are you currently in a relationship? If so, what context would help me support you?\n"
        "- Are you dating? Any patterns, preferences, or situations worth remembering?\n"
        "- Is there anything about your romantic life that affects your plans or emotional state?\n"
        "Save with save_relationship_profile (relationship type: partner/dating/etc.)."
    ),
    "family": (
        "Ask the user:\n"
        "- Who are the key people in your family I should know about?\n"
        "- For each: name, relationship (mother, brother, etc.), and any context?\n"
        "- Any family situations or ongoing dynamics worth remembering?\n"
        "Save each with save_relationship_profile."
    ),
    "habits": (
        "Ask the user:\n"
        "- What daily or weekly habits are you currently building or maintaining?\n"
        "- Which ones are non-negotiable in your routine?\n"
        "- Are there habits you want to start being consistent about?\n"
        "For each habit: ask name, category (health/sport/focus/nutrition), and frequency.\n"
        "Create each with create_habit."
    ),
    "body": (
        "Ask the user:\n"
        "- What is your current weight?\n"
        "- How many hours of sleep are you getting, and how is the quality?\n"
        "- How would you rate your energy, stress, and soreness this week (1 to 5)?\n"
        "- If you know it, what are your resting heart rate and daily steps looking like?\n"
        "Log with log_body_metrics."
    ),
}


@tool
def ask_about_chapter(
    chapter: str,
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Get a structured intake prompt to gather missing data for a specific chapter.

    Valid chapters: calendar, gym, food, birthdays, trips, plans, dating, family, habits, body.
    Use this when a chapter is empty or needs an update, then ask the user the listed questions.
    """
    key = chapter.strip().lower()
    prompt = _CHAPTER_INTAKE_PROMPTS.get(key)
    if not prompt:
        available = ", ".join(_CHAPTER_INTAKE_PROMPTS.keys())
        return f"Unknown chapter '{chapter}'. Valid options: {available}."
    return f"Intake prompt for '{key}':\n\n{prompt}"


@tool
def get_personal_context(
    topic: str,
    state: Annotated[AgentState, InjectedState] = None,
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
            parts.append("Open loops: " + "; ".join(l.get("topic", "") for l in loops) + ".")
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
        from datetime import date, timedelta
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


JUNE_TOOLS_CORE = [
    log_mood,
    save_journal_entry,
    track_goal,
    update_goal_status,
    save_open_loop,
    update_open_loop_status,
    save_calendar_item,
    list_calendar_items,
    update_calendar_item_status,
    save_user_preference,
    log_body_metrics,
    log_workout_session,
    log_habit_completion,
    create_habit,
    log_nutrition,
    log_water,
    set_ui_chapter,
    set_ui_focus,
    set_ui_checklist,
    clear_ui_workspace,
]

JUNE_TOOLS_GEMMA = [
    log_mood,
    get_mood_history,
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
    save_gym_plan,
    list_gym_plans,
    save_food_program,
    list_food_programs,
    log_workout_session,
    log_body_metrics,
    create_habit,
    log_habit_completion,
    get_habits_with_streaks,
    log_nutrition,
    log_water,
    get_today_summary,
    get_recovery_readiness_summary,
    get_active_commitments_summary,
    summarize_progress,
    set_ui_focus,
    set_ui_checklist,
    set_ui_layout,
    set_ui_chapter,
    clear_ui_workspace,
    check_chapter_completeness,
    ask_about_chapter,
    get_personal_context,
]

JUNE_TOOLS = [
    log_mood,
    get_mood_history,
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
    save_gym_plan,
    list_gym_plans,
    save_food_program,
    list_food_programs,
    log_workout_session,
    log_body_metrics,
    create_habit,
    log_habit_completion,
    get_habits_with_streaks,
    log_nutrition,
    log_water,
    get_today_summary,
    get_recovery_readiness_summary,
    get_active_commitments_summary,
    summarize_progress,
    get_runtime_privacy_status,
    preview_runtime_preset_switch,
    switch_runtime_preset,
    analyze_compatibility,
    generate_conversation_starters,
    draft_reply,
    plan_difficult_conversation,
    set_ui_focus,
    set_ui_checklist,
    set_ui_layout,
    set_ui_chapter,
    clear_ui_workspace,
    check_chapter_completeness,
    ask_about_chapter,
    get_personal_context,
]
