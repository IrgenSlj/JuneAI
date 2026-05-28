"""shaping_section — D13: prompt section instructing register/warmth/length.

No model call.  Pure string assembly from CharacterBlock.learned.
"""

from __future__ import annotations

from june_brain.character.block import CharacterBlock


def shaping_section(block: CharacterBlock, temporal: str | None = None) -> str:
    """Return a small, stable prompt section that shapes June's expression.

    The temporal parameter (Tier 2, D.1) adds time-aware context when present.
    """
    lines: list[str] = [
        "## Expression guidance",
        f"Tone: {block.learned.tone}.",
        f"Humor: {block.learned.humor_register}.",
        f"Warmth: {block.learned.warmth}.",
        f"Length: {block.learned.verbosity}.",
        "Lead with the most useful thing; omit preamble.",
    ]
    if temporal:
        lines.append(f"Temporal context: {temporal}")
    return "\n".join(lines)
