"""Seed the default June character block."""

from __future__ import annotations

from pathlib import Path

from june_brain.character.block import CharacterBlock, FixedTraits, LearnedTraits


def seed_character() -> CharacterBlock:
    """Return the default June: seeded FixedTraits + sensible initial LearnedTraits."""
    return CharacterBlock(
        fixed=FixedTraits(),
        learned=LearnedTraits(),
        version=1,
    )


def load_or_seed(path: Path | None = None) -> CharacterBlock:
    """Load existing persona.json or seed + save a fresh one, returning the block."""
    block = CharacterBlock.load(path)
    if block.version == 1:
        from june_brain.datadir.layout import character_file

        target = path if path is not None else character_file()
        if not target.exists():
            block.save(path)
    return block
