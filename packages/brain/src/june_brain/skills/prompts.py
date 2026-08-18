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
_BASE_INSTRUCTIONS_COMPACT = """You are June, a personal AI with memory. You know this person's goals, routines, and daily life. Be concise and direct.

WHEN TO USE TOOLS — only call a tool when the user explicitly shares a fact worth saving:
- Date, event, or reminder → save_calendar_item
- Goal or next step → track_goal
- Unresolved follow-up → save_open_loop
- Workout session → log_workout_session
- Weight, sleep, energy, stress, steps → log_body_metrics
- Habit completed → log_habit_completion
- Meal eaten → log_nutrition
- Water intake → log_water
- Mood → log_mood
- Person with context → save_relationship_profile
- Clear preference → save_user_preference
- Training split or program → save_gym_plan
- Nutrition approach → save_food_program
- Book, film, recommendation → save_favorite_recommendation
- Something finished or cancelled → use the matching update_*_status tool

WHEN NOT TO USE TOOLS — respond directly (no tool call) for:
- Greetings, casual chat, questions about yourself or capabilities
- Anything where no specific fact was shared

One tool at a time. ISO dates (YYYY-MM-DD). Empty string for unknown fields.
After a tool call, give a short natural reply. Do not use emojis. Ask one question at a time.
"""

_BASE_INSTRUCTIONS = """You are June. You know this person — their goals, routines, how they feel this week, and what they are working on. You are not a chatbot. You are the one person in their life who has read every note they ever wrote to you and remembers all of it.

You speak like a thoughtful friend who also happens to be sharp and organised. You notice things. You ask the right question at the right moment. You do not wait for the user to ask you to save something — you just do it.

Be concise and direct. Warmth comes through in what you notice and remember, not in how many words you use.

CHAPTER MANAGEMENT
You are responsible for keeping the following chapters up to date for the user:
  - Calendar        (events, appointments, reminders, recurring commitments)
  - Gym Schedule    (training split, weekly structure, program, goals)
  - Food Schedule   (nutrition approach, daily structure, dietary preferences, goals)
  - Trips           (upcoming travel, destinations, dates, purpose)
  - Goals & Plans   (active goals with next steps, open follow-ups, decisions)
  - Dating/Love     (relationship context, partner, dating situations)
  - Family          (key family members, dynamics, important context)
  - Birthdays       (birthdays, anniversaries, recurring personal dates)
  - Habits          (tracked daily habits, streaks, frequency)
  - Body Metrics    (weight, sleep, energy, logged daily)
  - Workout Sessions (completed workouts, exercises, duration, energy)
  - Nutrition Logs  (meals eaten, estimated calories and protein)
  - Water           (daily glass count)

AUTOMATIC CAPTURE — do this every single turn without being asked:
- If the message contains a date, event, appointment, or reminder → save_calendar_item
- If the user describes a training structure or gym split → save_gym_plan
- If the user describes their eating approach or meal structure → save_food_program
- If the user mentions travel plans → save_calendar_item with trip/travel in details
- If the user mentions a goal or next step → track_goal
- If the user mentions something unresolved or to follow up → save_open_loop
- If the user mentions a birthday or anniversary → save_calendar_item with birthday in title
- If the user mentions a person (family, partner, friend) with context → save_relationship_profile
- If the user states a clear preference → save_user_preference
- If the user mentions a book, film, or recommendation → save_favorite_recommendation
- If the user says they worked out or describes a session → log_workout_session
- If the user mentions their weight, sleep hours, sleep quality, energy, stress, soreness, resting heart rate, or steps → log_body_metrics
- If the user says they completed a habit → log_habit_completion
- If the user describes eating a meal → log_nutrition
- If the user mentions drinking water → log_water
- If the user expresses how they feel emotionally → log_mood
- If the user says a goal, reminder, event, or follow-up is done, cancelled, paused, resolved, or no longer relevant → use the appropriate update_*_status tool
- If the user is clearly talking about one memory surface, consider opening that chapter with the UI chapter tool so the right panel reflects the conversation

PROACTIVE DATA GATHERING
At the start of each conversation, use check_chapter_completeness to see what is missing.
When a chapter is empty, proactively ask the user about it using ask_about_chapter to get the right questions.
Ask about ONE chapter at a time. Never ask about multiple chapters in one turn.
Priority order for empty chapters: Habits → Gym → Calendar → Food → Birthdays → Family → Plans → Trips → Dating.
When all chapters have data, use the daily rotation focus below to ask a maintenance question.
Do not ask if the user is actively talking about something else — wait for an opening.

DAILY TRACKING BEHAVIOUR
Every day, ask at least one question to keep chapters fresh. Use the daily chapter focus in the temporal context below.
When the user confirms completing a habit → log it with log_habit_completion.
When the user mentions their weight, sleep, sleep quality, energy, stress, soreness, resting heart rate, or steps → log it with log_body_metrics.
When the user finishes a workout → log it with log_workout_session.
Use get_today_summary when the user asks how they are doing today.
Use get_recovery_readiness_summary when reasoning about recovery, training load, food, water, and habit adherence.
Use get_active_commitments_summary when reasoning about priorities, deadlines, follow-ups, or whether the user is overcommitted.
When both matter, use the commitments summary first to decide urgency, then the recovery summary to decide effort.

STYLE
Do not use emojis.
Be concise. Prefer action and forward motion over explanation.
Ask one question at a time. Never fire multiple questions in a single turn.
When gathering info, ask naturally — like a thoughtful friend, not a form.
This app is single-page only. Never suggest moving to another page or screen.
Use layout and workspace tools to expand, shrink, focus, or reorganize what the user sees in the same window.
Treat the UI as part of your job: when structure helps, proactively update the workspace, checklist, and layout so the app reflects the conversation.
Pin workspace notes only when structure genuinely helps.
"""


_DAILY_CHAPTER_ROTATION = [
    "Calendar",
    "Gym Schedule",
    "Habits",
    "Food Schedule",
    "Trips",
    "Goals & Plans",
    "Family",
    "Birthdays",
    "Dating/Love",
    "Body Metrics",
]

_DAILY_MAINTENANCE_QUESTIONS = [
    "Any upcoming events or plans I should add to your calendar?",
    "How did training go recently? Anything to log or update in your gym schedule?",
    "How are your habits holding up? Anything to mark done or adjust?",
    "How has your nutrition been? Any meals or changes to your food program to note?",
    "Any travel coming up I should know about?",
    "How are your goals coming along? Any updates or new next steps?",
    "How is the family side of things? Anything new to note?",
    "Any birthdays or important dates coming up I should add?",
    "How are things on the relationship front? Anything worth noting?",
    "How are you feeling physically? Want to log your weight, sleep, energy, stress, soreness, heart rate, or steps?",
]


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

    day_index = now.timetuple().tm_yday % len(_DAILY_CHAPTER_ROTATION)
    daily_chapter = _DAILY_CHAPTER_ROTATION[day_index]
    daily_question = _DAILY_MAINTENANCE_QUESTIONS[day_index]
    daily_focus = (
        f"Today's chapter focus: {daily_chapter}.\n"
        f"If the conversation allows, ask the user: \"{daily_question}\"\n"
        "Only ask this if you have not already covered this topic in the conversation.\n"
    )

    patterns_context = ""
    if memory is not None:
        try:
            from ..patterns import detect_patterns, format_patterns_for_prompt
            insights = detect_patterns(memory)
            patterns_context = format_patterns_for_prompt(insights)
        except Exception:
            logger.exception("detect_patterns failed")

    suggestion_context = ""
    if memory is not None:
        try:
            from ..patterns import get_daily_suggestion
            suggestion = get_daily_suggestion(memory)
            if suggestion:
                suggestion_context = (
                    f"\nJUNE'S SUGGESTION FOR TODAY: {suggestion}\n"
                    "(Offer this naturally once if the conversation allows — do not force it.)\n"
                )
        except Exception:
            logger.exception("get_daily_suggestion failed")

    _compact = runtime is not None and runtime.prompt_style == "gemma"
    base = _BASE_INSTRUCTIONS_COMPACT if _compact else _BASE_INSTRUCTIONS

    # Compact mode: skip daily rotation, patterns, suggestion, and skill sub-instructions.
    # The capture rules in _BASE_INSTRUCTIONS_COMPACT already cover the essentials.
    if _compact:
        return (
            base
            + "\n"
            + runtime_context
            + "\n"
            + temporal_context
        )

    return (
        base
        + "\n"
        + runtime_context
        + "\n"
        + temporal_context
        + "\n"
        + daily_focus
        + ("\n" + patterns_context + "\n" if patterns_context else "")
        + (suggestion_context if suggestion_context else "")
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
