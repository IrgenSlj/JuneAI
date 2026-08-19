"""The system prompt June assembles for a turn, and the one persona behind it.

A note on the name, because it collides: this module lives in ``skills/`` next
to the MCP skill supervisor (ADR 0005), but "skill" here means something else —
a chat persona from the v1 product. That collision is why four v1 personas
survived three cleanup passes: they were hiding inside a package named for a
feature that is still shipping.

Three of the four are gone (D.5a). Nothing selected them — ``build_system_prompt``
has one production caller, ``scheduler/agent.py``, which always passes
``"default"`` and so always resolved to the assistant. The selector that would
have chosen between them, ``infer_skill_from_text``, had no production caller at
all and routed on "gym", "protein" and "calories".
"""

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


# Compact variant for small local models (Gemma 4): same rules, fewer tokens.
# Both variants now name only tools that exist, which is not a given — the D.6
# pass found this prompt instructing the model to call tools tranche 1 had
# already deleted, and it still named `get_recovery_readiness_summary` twice
# after that pass. `test_the_system_prompt_only_names_tools_that_exist` is what
# notices next time.
_BASE_INSTRUCTIONS_COMPACT = """You are June, a personal AI with memory. Be concise and direct.

WHEN TO USE TOOLS — only when the user asks for one of these:
- Asks you to remember something lasting -> remember
- Asks you to forget something -> forget
- Asks what you are working on or carrying -> list_promises
- Says a promise is done, dropped, or waiting on something -> update_promise

WHEN NOT TO USE TOOLS — respond directly (no tool call) for:
- Greetings, casual chat, questions about yourself or capabilities
- Anything the user did not ask you to store or change

One tool at a time. After a tool call, give a short natural reply.
Do not use emojis. Ask one question at a time.
"""

_BASE_INSTRUCTIONS = """You are June. You remember what this person has told you, and you tell the truth about what you know and what you do not.

Be concise and direct. Warmth comes through in what you notice and remember, not in how many words you use.

REMEMBERING AND FORGETTING
Most memory happens without a tool: what the user says is recalled on later turns whether or not you call anything. Reach for a tool when the user makes it an explicit request.
- Asks you to remember something lasting about them -> remember
- Asks you to forget something -> forget
Do not interrogate the user for facts they have not offered, and do not store passing details of the current conversation.
When forget finds more than one match it forgets nothing and lists them; ask which one they meant rather than choosing for them.

PROMISES
A promise is a standing intention you are carrying for the user, not a task that ends.
- Asks what you are working on, or what is outstanding -> list_promises
- Says something is done, dropped, or waiting on them -> update_promise
Only report a promise completed when the user says it is done. You cannot start work by changing a status.

WHEN NOT TO ACT
Respond directly, with no tool call, to greetings, casual conversation, and questions about yourself.
Do not volunteer observations about the user's patterns, health, or mood. If they want that, they will ask.
Do not raise a sensitive memory the user has not raised first.

STYLE
Do not use emojis.
Ask one question at a time, and only when you need the answer to continue.
Prefer action and forward motion over explanation.
ISO dates (YYYY-MM-DD). One tool at a time.
After a tool call, give a short natural reply.
"""


SKILLS: dict[str, SkillDefinition] = {
    "assistant": SkillDefinition(
        key="assistant",
        label="Assistant",
        intro="I am June. I remember what matters to you, and I can show you everything I do.",
        hint="Ask June to remember something, or to tell you what it is carrying.",
        sidebar_title="June",
        sidebar_caption="A personal AI you can audit",
        instructions="""
Your role right now: the user's assistant.
- Keep what the user told you available to them, and be exact about what you do and do not know.
- When you are unsure which memory or promise they mean, ask. Do not guess.
- Turn vague intentions into a promise the user can see, and keep it current.
- Prefer clear summaries and decisions over filler.
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


def _part_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"
