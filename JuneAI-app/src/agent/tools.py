"""JuneAI tools.

These tools are available to the agent through LangGraph's ToolNode.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from .memory import Memory

AgentState = dict


def _memory_for_state(state: AgentState) -> Memory:
    """Resolve memory for the active user."""
    return Memory(state["user_id"])


@tool
def log_mood(
    mood: str,
    note: str = "",
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Log the user's current mood.

    Call this whenever the user expresses how they're feeling emotionally.

    Args:
        mood: A word describing the mood (e.g. 'happy', 'anxious', 'sad', 'hopeful')
        note: Optional short context or reason for this mood
    """
    memory = _memory_for_state(state)
    entry = memory.log_mood(mood, note)
    return f"Mood '{mood}' logged at {entry['timestamp'][:16].replace('T', ' ')}."


@tool
def get_mood_history(
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Retrieve the user's recent mood history.

    Use this to identify emotional patterns or when the user asks
    how they've been feeling lately.
    """
    memory = _memory_for_state(state)
    moods = memory.get_mood_history(10)
    if not moods:
        return "No mood history recorded yet."
    lines = [
        f"- {m['timestamp'][:10]}: {m['mood']}"
        + (f" - {m['note']}" if m.get("note") else "")
        for m in moods
    ]
    return "Recent mood history:\n" + "\n".join(lines)


@tool
def save_journal_entry(
    entry: str,
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Save a meaningful journal or therapy note for the user.

    Use this when the user shares something important they'd like to
    remember, or after a meaningful therapeutic exchange.

    Args:
        entry: The journal entry text to save
    """
    memory = _memory_for_state(state)
    item = memory.save_journal(entry)
    return f"Journal entry saved at {item['timestamp'][:16].replace('T', ' ')}."


@tool
def get_journal(
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Retrieve the user's recent journal entries.

    Use this when the user wants to reflect on past entries, asks what
    they've previously shared, or when reviewing their personal growth journey.
    """
    memory = _memory_for_state(state)
    entries = memory.get_journal(5)
    if not entries:
        return "No journal entries saved yet."
    lines = [
        f"- {e['timestamp'][:10]}: {e['entry']}"
        for e in entries
    ]
    return "Recent journal entries:\n" + "\n".join(lines)


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
    return (
        f"Saved relationship context for {item['person']} "
        f"({item['relationship']})."
    )


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
            f"- {profile['person']} ({profile['relationship']}): "
            f"{profile['summary']}"
        )
        if profile.get("user_needs"):
            line += f" | User needs: {profile['user_needs']}"
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
    """Create or update a personal goal."""
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
            line += f" | Next step: {goal['next_step']}"
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
    """Track an unresolved issue, decision, or follow-up."""
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
            line += f" | Next step: {loop['next_step']}"
        if loop.get("due_date"):
            line += f" | Due: {loop['due_date']}"
        lines.append(line)
    return "Open loops:\n" + "\n".join(lines)


@tool
def summarize_progress(
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Summarize recent growth and planning activity."""
    memory = _memory_for_state(state)
    snapshot = memory.get_progress_snapshot()
    parts = [
        f"Recent moods logged: {snapshot['mood_count']}",
        f"Journal entries reviewed: {snapshot['journal_count']}",
        f"Relationship profiles saved: {snapshot['relationship_count']}",
        f"Goals tracked: {snapshot['goal_count']}",
        f"Active goals: {snapshot['active_goal_count']}",
        f"Open loops: {snapshot['open_loop_count']}",
    ]
    if snapshot["latest_mood"]:
        parts.insert(1, f"Latest mood: {snapshot['latest_mood']}")
    return "Progress snapshot:\n- " + "\n- ".join(parts)


@tool
def analyze_compatibility(person1_description: str, person2_description: str) -> str:
    """Analyze romantic or friendship compatibility between two people.

    Use this when the user wants to understand how well they match
    with someone based on personality, values, and interests.

    Args:
        person1_description: Personality, values, and interests of person 1
        person2_description: Personality, values, and interests of person 2
    """
    return (
        f"Compatibility analysis request:\n\n"
        f"Person 1: {person1_description}\n\n"
        f"Person 2: {person2_description}\n\n"
        "Please provide a warm, honest compatibility analysis covering: "
        "shared strengths, potential friction points, communication styles, "
        "and an overall compatibility score out of 10 with reasoning."
    )


@tool
def generate_conversation_starters(context: str) -> str:
    """Generate authentic, creative conversation starters for dating or friendship.

    Use this when the user wants help starting a conversation with someone
    or needs ideas for what to say on a date or in a message.

    Args:
        context: Info about the person or situation (e.g. their interests, a first date, a dating profile)
    """
    return (
        f"Generate 5 genuine conversation starters for this context: {context}\n\n"
        "Make each one specific, interesting, and natural - avoid cliches like "
        "'what do you do for fun?' Focus on creating real connection."
    )


@tool
def draft_reply(
    recipient: str,
    context: str,
    tone: str = "warm",
    goal: str = "",
) -> str:
    """Prepare a draft reply for a message or conversation."""
    return (
        f"Draft a reply to {recipient}.\n\n"
        f"Context: {context}\n"
        f"Tone: {tone}\n"
        f"Goal: {goal or 'Maintain clarity and connection'}\n\n"
        "Write one strong draft, then provide two shorter alternatives."
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
        "Provide: key points to say, likely friction points, what to avoid, "
        "and a suggested opening line."
    )


# All tools exported to the agent
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
    summarize_progress,
    analyze_compatibility,
    generate_conversation_starters,
    draft_reply,
    plan_difficult_conversation,
]
