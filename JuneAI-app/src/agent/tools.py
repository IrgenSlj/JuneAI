"""JuneAI tools — what June can *do* beyond just talking.

LangGraph uses these as callable functions that Gemini can invoke
during a conversation when it decides a tool is appropriate.

Each tool is decorated with @tool which turns it into a LangChain Tool
object with a name, description, and typed arguments.
"""

from typing import Optional

from langchain_core.tools import tool

from .memory import Memory

# Global memory instance — set per user session from app.py
_memory: Optional[Memory] = None


def set_memory(memory: Memory) -> None:
    """Inject the current user's memory into the tools."""
    global _memory
    _memory = memory


@tool
def log_mood(mood: str, note: str = "") -> str:
    """Log the user's current mood.

    Call this whenever the user expresses how they're feeling emotionally.

    Args:
        mood: A word describing the mood (e.g. 'happy', 'anxious', 'sad', 'hopeful')
        note: Optional short context or reason for this mood
    """
    if _memory:
        entry = _memory.log_mood(mood, note)
        return f"Mood '{mood}' logged at {entry['timestamp'][:16].replace('T', ' ')}."
    return f"Mood '{mood}' noted (memory unavailable)."


@tool
def get_mood_history() -> str:
    """Retrieve the user's recent mood history.

    Use this to identify emotional patterns or when the user asks
    how they've been feeling lately.
    """
    if _memory:
        moods = _memory.get_mood_history(10)
        if not moods:
            return "No mood history recorded yet."
        lines = [
            f"- {m['timestamp'][:10]}: **{m['mood']}**"
            + (f" — {m['note']}" if m.get("note") else "")
            for m in moods
        ]
        return "Recent mood history:\n" + "\n".join(lines)
    return "Mood history unavailable."


@tool
def save_journal_entry(entry: str) -> str:
    """Save a meaningful journal or therapy note for the user.

    Use this when the user shares something important they'd like to
    remember, or after a meaningful therapeutic exchange.

    Args:
        entry: The journal entry text to save
    """
    if _memory:
        item = _memory.save_journal(entry)
        return f"Journal entry saved at {item['timestamp'][:16].replace('T', ' ')}."
    return "Journal entry noted."


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
        "Make each one specific, interesting, and natural — avoid clichés like "
        "'what do you do for fun?' Focus on creating real connection."
    )


# All tools exported to the agent
JUNE_TOOLS = [
    log_mood,
    get_mood_history,
    save_journal_entry,
    analyze_compatibility,
    generate_conversation_starters,
]
