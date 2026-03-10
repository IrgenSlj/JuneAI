"""Compatibility layer for prompt access.

Prompt construction now lives in the skill registry.
"""

from .skills import DEFAULT_SKILL, SKILLS, build_system_prompt

MODE_PROMPTS: dict[str, str] = {
    key: build_system_prompt(key) for key in SKILLS
}

JUNE_SYSTEM_PROMPT = build_system_prompt(DEFAULT_SKILL)
