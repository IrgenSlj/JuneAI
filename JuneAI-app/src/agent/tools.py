"""JuneAI tools exposed to the LangGraph agent."""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from .memory import Memory

AgentState = dict
DEFAULT_UI_STATE = {
    "layout": "split",
    "focus_title": "Workspace",
    "focus_body": "June can pin structured notes, plans, and highlights here.",
    "checklist_title": "Next steps",
    "checklist_items": [],
    "notice": "",
}


def _memory_for_state(state: AgentState) -> Memory:
    """Resolve memory for the active user."""
    return Memory(state["user_id"])


def _merge_ui_state(current: dict | None, updates: dict) -> dict:
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
    return f"Mood '{mood}' logged at {entry['timestamp'][:16].replace('T', ' ')}."


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
    return f"Journal entry saved at {item['timestamp'][:16].replace('T', ' ')}."


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
    return f"Saved relationship context for {item['person']} ({item['relationship']})."


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
    return f"Saved goal '{goal['title']}' with status '{goal['status']}'."


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
    return f"Saved open loop '{item['topic']}' with status '{item['status']}'."


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
    return f"Saved preference '{item['category']}: {item['value']}'."


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
    return f"Saved calendar item '{item['title']}' on {item['date']}."


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
    return f"Saved {item['category']} recommendation '{item['title']}'."


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
    return f"Saved gym plan '{item['name']}'."


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
    return f"Saved food program '{item['name']}'."


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
    state: Annotated[AgentState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
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
    state: Annotated[AgentState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
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
    state: Annotated[AgentState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
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
def clear_ui_workspace(
    state: Annotated[AgentState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
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


JUNE_TOOLS = [
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
    save_user_preference,
    get_user_preferences,
    save_calendar_item,
    list_calendar_items,
    save_favorite_recommendation,
    list_favorites,
    save_gym_plan,
    list_gym_plans,
    save_food_program,
    list_food_programs,
    summarize_progress,
    analyze_compatibility,
    generate_conversation_starters,
    draft_reply,
    plan_difficult_conversation,
    set_ui_focus,
    set_ui_checklist,
    set_ui_layout,
    clear_ui_workspace,
]
