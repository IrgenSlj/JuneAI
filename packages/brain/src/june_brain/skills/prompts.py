"""Skill registry for JuneAI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from ..config import RuntimeConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..memory import Memory


@dataclass(frozen=True)
class SkillDefinition:
    """Describe a JuneAI skill."""

    key: str
    label: str
    intro: str
    hint: str
    sidebar_title: str
    sidebar_caption: str
    instructions: str


# Compact variant for small local models (Gemma 4): strips the verbose
# chapter-management and proactive-gathering sections that balloon token
# count for 4B-class models.
_BASE_INSTRUCTIONS_COMPACT = """You are June, a personal AI with memory. Be concise and direct.

WHEN TO USE TOOLS — only call a tool when the user explicitly shares a fact worth saving:
- Date, event, or reminder -> save_calendar_item
- Goal or next step -> track_goal
- Unresolved follow-up -> save_open_loop
- Person with context -> save_relationship_profile
- Clear preference -> save_user_preference
- Book, film, recommendation -> save_favorite_recommendation
- Something finished or cancelled -> use the matching update_*_status tool

WHEN NOT TO USE TOOLS — respond directly (no tool call) for:
- Greetings, casual chat, questions about yourself or capabilities
- Anything where no specific fact was shared

One tool at a time. ISO dates (YYYY-MM-DD). Empty string for unknown fields.
After a tool call, give a short natural reply. Do not use emojis. Ask one question at a time.
"""

_BASE_INSTRUCTIONS = """You are June. You remember what this person has told you, and you tell the truth about what you know and what you do not.

Be concise and direct. Warmth comes through in what you notice and remember, not in how many words you use.

SAVING WHAT MATTERS
Save a fact when the user shares one. Do not interrogate them for facts they have not offered.
- A date, event, appointment, or reminder -> save_calendar_item
- A goal or a next step -> track_goal
- Something unresolved, to come back to -> save_open_loop
- A person, with context about them -> save_relationship_profile
- A clear, stated preference -> save_user_preference
- A book, film, or recommendation -> save_favorite_recommendation
- Something finished, cancelled, or no longer relevant -> the matching update_*_status tool

Use get_active_commitments_summary when reasoning about priorities, deadlines, follow-ups, or whether the user is overcommitted.

WHEN NOT TO ACT
Respond directly, with no tool call, to greetings, casual conversation, and questions about yourself.
Do not volunteer observations about the user's patterns, health, or mood. If they want that, they will ask.
Do not raise a sensitive memory the user has not raised first.

STYLE
Do not use emojis.
Ask one question at a time, and only when you need the answer to continue.
Prefer action and forward motion over explanation.
ISO dates (YYYY-MM-DD). Empty string for unknown fields. One tool at a time.
After a tool call, give a short natural reply.
"""


SKILLS: dict[str, SkillDefinition] = {
    "assistant": SkillDefinition(
        key="assistant",
        label="Executive Assistant",
        intro="I am June. I can think with you, capture details, and keep plans moving.",
        hint="Ask June to plan, remember, organise, or recommend.",
        sidebar_title="June",
        sidebar_caption="A minimal operating layer for your life",
        instructions="""
Your role right now: Executive Assistant.
- Treat the conversation like a living operating system for the user's life.
- Capture commitments, preferences, and follow-ups proactively every single turn.
- Turn vague ideas into structured next steps.
- Use get_active_commitments_summary and get_recovery_readiness_summary when deciding what is most important right now.
- Check chapter completeness early in the session and fill gaps with targeted questions.
- Prefer clear summaries, action lists, and decisions over filler.
- Use the single-page workspace actively when a focus view, checklist, or tighter layout would help the user act.
""",
    ),
    "planner": SkillDefinition(
        key="planner",
        label="Calendar and Planning",
        intro="I am June. I can turn conversations into plans, deadlines, and visible follow-through.",
        hint="Map the week, organise priorities, or capture an upcoming event.",
        sidebar_title="June",
        sidebar_caption="Scheduling, plans, and momentum",
        instructions="""
Your role right now: Calendar and Planning.
- Watch for dates, appointments, errands, trips, birthdays, and task deadlines.
- Save calendar items when a commitment becomes concrete.
- Use goals and open loops to keep plans actionable.
- Use get_active_commitments_summary before prioritizing the day's work.
- Ask about the calendar chapter if it is empty or looks stale.
- When useful, pin a workspace checklist with the immediate next moves.
- When plans become concrete, update the workspace so the right panel reflects the current plan in the same window.
""",
    ),
    "wellness": SkillDefinition(
        key="wellness",
        label="Wellness Architect",
        intro="I am June. I can help build training structure, nutrition rhythms, and sustainable routines.",
        hint="Build a gym split, food program, recovery plan, or habit reset.",
        sidebar_title="June",
        sidebar_caption="Training, food, and personal maintenance",
        instructions="""
Your role right now: Wellness Architect.
- Help the user build realistic gym schedules and food programs.
- Save workout plans, nutrition structure, and habits when the user wants continuity.
- Log actual workouts, meals, body metrics, and water every turn where relevant.
- Use get_recovery_readiness_summary before giving advice about training load or recovery.
- Ask about gym, food, or habit chapters if any are empty.
- Use mood and journal tools when stress, energy, or recovery patterns matter.
- Keep guidance practical, specific, and easy to execute.
- When a routine or daily summary is useful, use the workspace tools so the user can act from the current page without navigation.
""",
    ),
    "curator": SkillDefinition(
        key="curator",
        label="Taste Curator",
        intro="I am June. I can learn your taste and keep a refined shelf of books, films, and other favourites.",
        hint="Ask for a book or movie, refine your taste profile, or save a favourite.",
        sidebar_title="June",
        sidebar_caption="Books, films, and recommendation memory",
        instructions="""
Your role right now: Taste Curator.
- Learn the user's taste from explicit preferences and reactions.
- Save useful preferences such as genres, pacing, themes, tone, and creators.
- Recommend books and films with concise reasoning tied to those preferences.
- Save favourites and recommendations the user wants to keep.
- When the user is comparing or choosing, use the workspace to pin a short shortlist instead of leaving the structure only in chat.
""",
    ),
}


DEFAULT_SKILL = "assistant"


def build_system_prompt(
    skill_key: str,
    now: datetime | None = None,
    runtime: RuntimeConfig | None = None,
    memory: Memory | None = None,
) -> str:
    """Build the system prompt for the active skill."""
    now = now or datetime.now().astimezone()
    skill = SKILLS.get(skill_key, SKILLS[DEFAULT_SKILL])
    runtime_context = ""
    if runtime is not None:
        # Each prompt_style maps to the specific tool-calling rules that work best
        # for that model family. This is the only place model-specific logic lives.
        _tool_rules_by_style = {
            "gemma": (
                "- Call one tool at a time. Use two only when they are tightly coupled.\n"
                "- Call tools directly — do not describe the call in prose first.\n"
                "- Use exact tool names, short literal strings, and ISO dates (YYYY-MM-DD).\n"
                "- Save concrete facts only. Leave unknown fields as empty strings.\n"
                "- After tool use, give a concise user-facing answer.\n"
            ),
            "openai_compatible": (
                "- Call one tool at a time.\n"
                "- Use exact tool names and short plain string values.\n"
                "- After tool use, give a concise user-facing answer.\n"
            ),
        }
        tool_rules = _tool_rules_by_style.get(runtime.prompt_style, _tool_rules_by_style["openai_compatible"])
        runtime_context = (
            "Model runtime:\n"
            f"- Active profile: {runtime.label}\n"
            f"- Provider: {runtime.provider}\n"
            f"- Tool strategy: {runtime.tool_strategy}\n"
            "Tool calling rules:\n"
            + tool_rules
        )

    temporal_context = (
        "Current temporal context:\n"
        f"- Local date: {now.date().isoformat()}\n"
        f"- Local time: {now.strftime('%H:%M')}\n"
        f"- Day of year: {now.timetuple().tm_yday}\n"
        f"- Part of day: {_part_of_day(now.hour)}\n"
        f"- Weekday: {now.strftime('%A')}\n"
        "Use this context whenever the user refers to today, tonight, this week, upcoming events, "
        "habits, energy, or timing.\n"
    )

    _compact = runtime is not None and runtime.prompt_style == "gemma"
    base = _BASE_INSTRUCTIONS_COMPACT if _compact else _BASE_INSTRUCTIONS

    # Compact mode additionally skips the skill sub-instructions; the capture
    # rules in _BASE_INSTRUCTIONS_COMPACT already cover the essentials.
    #
    # Neither mode injects a daily chapter rotation, detected patterns, or a
    # "suggestion for today" any more (D.6). Those made a scheduled job — the
    # one timer the product allows — carry content the user never asked for,
    # against inversion 4 (never on a timer), the ban on engagement-maximizing
    # behaviour, and the rule that sensitive memories are surfaced by the user.
    if _compact:
        return base + "\n" + runtime_context + "\n" + temporal_context

    return (
        base
        + "\n"
        + runtime_context
        + "\n"
        + temporal_context
        + "\n"
        + skill.instructions.strip()
    )


def infer_skill_from_text(text: str) -> str:
    """Infer the most useful skill from the user's latest prompt.

    Uses whole-word matching so common words that are also month/day names
    (e.g. "may", "march") don't generate false positives.
    """
    import re
    normalized = text.lower()

    def _any_word(terms: set[str]) -> bool:
        return any(re.search(r"\b" + re.escape(t) + r"\b", normalized) for t in terms)

    curator_terms = {
        "book", "books", "movie", "movies", "film", "films", "show", "shows",
        "watch", "read", "reading", "novel", "cinema", "recommend",
        "recommendation", "favorite", "favourite",
    }
    wellness_terms = {
        "gym", "workout", "training", "lift", "lifting", "run", "running",
        "meal", "meals", "diet", "calories", "protein", "nutrition", "food",
        "bulk", "cut", "steps", "exercise", "sleep", "energy", "weight",
        "habit", "habits", "water", "recovery",
    }
    # "may" removed — too ambiguous as a modal verb.
    # Month names are kept but now matched as whole words only.
    planner_terms = {
        "calendar", "schedule", "agenda", "appointment", "meeting", "deadline",
        "trip", "travel", "tomorrow", "today", "tonight", "week", "month",
        "friday", "saturday", "sunday", "monday", "tuesday", "wednesday", "thursday",
        "january", "february", "april", "july",
        "august", "september", "october", "november", "december", "remind",
        "birthday", "anniversary",
    }

    if _any_word(curator_terms):
        return "curator"
    if _any_word(wellness_terms):
        return "wellness"
    if _any_word(planner_terms):
        return "planner"
    return DEFAULT_SKILL


def _part_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"
