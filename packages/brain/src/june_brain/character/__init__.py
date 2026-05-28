"""June character package — honest identity as a self-authored memory block."""

from june_brain.character.block import (
    FIXED_FIELD_NAMES,
    CharacterBlock,
    FixedTraits,
    LearnedTraits,
    character_update,
)
from june_brain.character.seed import load_or_seed, seed_character
from june_brain.character.shaping import shaping_section

__all__ = [
    "CharacterBlock",
    "FIXED_FIELD_NAMES",
    "FixedTraits",
    "LearnedTraits",
    "character_update",
    "load_or_seed",
    "seed_character",
    "shaping_section",
]
