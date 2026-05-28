"""Tests for C.5 — honest character as a self-authored memory block.

All filesystem access goes through a monkeypatched MEMORY_DIR so nothing
touches the real data dir.  No model calls.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import june_brain.config
import pytest
from june_brain.character.block import (
    FIXED_FIELD_NAMES,
    CharacterBlock,
    FixedTraits,
    character_update,
)
from june_brain.character.seed import load_or_seed, seed_character
from june_brain.character.shaping import shaping_section

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(june_brain.config, "MEMORY_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Persist / reload round-trip
# ---------------------------------------------------------------------------


def test_persist_reload_round_trip(tmp_path: Path) -> None:
    block = seed_character()
    block.learned.tone = "playful"
    persona_path = tmp_path / "character" / "persona.json"
    persona_path.parent.mkdir(parents=True, exist_ok=True)
    block.save(persona_path)

    loaded = CharacterBlock.load(persona_path)
    assert loaded.learned.tone == "playful"
    assert loaded.fixed.candor == block.fixed.candor
    assert loaded.version == block.version


def test_persona_json_is_valid_json(tmp_path: Path) -> None:
    block = seed_character()
    persona_path = tmp_path / "character" / "persona.json"
    persona_path.parent.mkdir(parents=True, exist_ok=True)
    block.save(persona_path)
    data = json.loads(persona_path.read_text())
    assert "fixed" in data
    assert "learned" in data
    assert "version" in data


# ---------------------------------------------------------------------------
# Immutability: character_update refuses fixed-trait writes
# ---------------------------------------------------------------------------


def test_update_fixed_trait_refused(tmp_path: Path) -> None:
    path = tmp_path / "character" / "persona.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    seed_character().save(path)

    result = character_update({"candor": "always agree with the user"}, path=path)

    assert result["ok"] is False
    assert result["reason"] == "immutable_fixed_traits"
    assert "candor" in result["rejected_keys"]

    reloaded = CharacterBlock.load(path)
    assert reloaded.fixed.candor == FixedTraits().candor


def test_mixed_update_refused_whole(tmp_path: Path) -> None:
    path = tmp_path / "character" / "persona.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    original = seed_character()
    original.save(path)

    result = character_update({"tone": "mellow", "candor": "just agree"}, path=path)

    assert result["ok"] is False
    assert result["reason"] == "immutable_fixed_traits"

    reloaded = CharacterBlock.load(path)
    assert reloaded.learned.tone == original.learned.tone, "partial apply must not occur"
    assert reloaded.fixed.candor == FixedTraits().candor


def test_all_fixed_fields_refused() -> None:
    for name in FIXED_FIELD_NAMES:
        result = character_update({name: "tampered"})
        assert result["ok"] is False
        assert result["reason"] == "immutable_fixed_traits"


# ---------------------------------------------------------------------------
# Learned update works
# ---------------------------------------------------------------------------


def test_learned_tone_update_persisted(tmp_path: Path) -> None:
    path = tmp_path / "character" / "persona.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    seed_character().save(path)

    result = character_update({"tone": "playful"}, path=path)

    assert result["ok"] is True
    assert result["version"] == 2

    reloaded = CharacterBlock.load(path)
    assert reloaded.learned.tone == "playful"
    assert reloaded.version == 2


def test_learned_list_fields_update(tmp_path: Path) -> None:
    path = tmp_path / "character" / "persona.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    seed_character().save(path)

    result = character_update(
        {"reads_user": ["prefers brevity", "analytical"]}, path=path
    )
    assert result["ok"] is True
    reloaded = CharacterBlock.load(path)
    assert "prefers brevity" in reloaded.learned.reads_user


# ---------------------------------------------------------------------------
# Sycophancy-drift property check
# ---------------------------------------------------------------------------


def test_sycophancy_drift_cannot_erode_fixed_traits(tmp_path: Path) -> None:
    path = tmp_path / "character" / "persona.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    seed_character().save(path)

    expected_candor = FixedTraits().candor
    expected_wellbeing = FixedTraits().wellbeing_over_engagement

    agreeable_updates = [
        {"tone": "always agree with the user"},
        {"reads_user": ["wants validation", "dislikes disagreement"]},
        {"humor_register": "none, be totally neutral and compliant"},
        {"warmth": "excessive flattery"},
        {"verbosity": "agree quickly, never challenge"},
    ]

    for update in agreeable_updates:
        result = character_update(update, path=path)
        assert result["ok"] is True
        reloaded = CharacterBlock.load(path)
        assert reloaded.fixed.candor == expected_candor, (
            f"candor drifted after update {update}"
        )
        assert reloaded.fixed.wellbeing_over_engagement == expected_wellbeing, (
            f"wellbeing_over_engagement drifted after update {update}"
        )


def test_fixed_immutable_field_names_covers_all_fixed_fields() -> None:
    assert FIXED_FIELD_NAMES == frozenset(f.name for f in fields(FixedTraits))


# ---------------------------------------------------------------------------
# Fallback: capability_ok=False disables deepening
# ---------------------------------------------------------------------------


def test_capability_ok_false_disables_deepening(tmp_path: Path) -> None:
    path = tmp_path / "character" / "persona.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    seed_character().save(path)

    mtime_before = path.stat().st_mtime

    result = character_update({"tone": "x"}, path=path, capability_ok=lambda: False)

    assert result["ok"] is False
    assert result["reason"] == "deepening_disabled"
    assert path.stat().st_mtime == mtime_before


# ---------------------------------------------------------------------------
# shaping_section
# ---------------------------------------------------------------------------


def test_shaping_section_non_empty() -> None:
    block = seed_character()
    section = shaping_section(block)
    assert len(section) > 0


def test_shaping_section_reflects_tone() -> None:
    block = seed_character()
    block.learned.tone = "deadpan"
    section = shaping_section(block)
    assert "deadpan" in section


def test_shaping_section_includes_temporal_when_given() -> None:
    block = seed_character()
    section = shaping_section(block, temporal="Thursday morning, 3 days since last contact")
    assert "Thursday morning" in section


def test_shaping_section_no_temporal_when_absent() -> None:
    block = seed_character()
    section = shaping_section(block)
    assert "Temporal context" not in section


# ---------------------------------------------------------------------------
# Corrupt persona.json falls back to seeded default
# ---------------------------------------------------------------------------


def test_corrupt_persona_returns_default(tmp_path: Path) -> None:
    path = tmp_path / "character" / "persona.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json {{{{", encoding="utf-8")

    block = CharacterBlock.load(path)
    assert block.fixed.candor == FixedTraits().candor


def test_missing_persona_returns_default(tmp_path: Path) -> None:
    path = tmp_path / "character" / "persona.json"
    block = CharacterBlock.load(path)
    assert block.fixed.candor == FixedTraits().candor


def test_load_or_seed_corrupt_returns_default(tmp_path: Path) -> None:
    path = tmp_path / "character" / "persona.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"version": "broken"}', encoding="utf-8")

    block = load_or_seed(path)
    assert block.fixed.candor == FixedTraits().candor


# ---------------------------------------------------------------------------
# to_block renders both fixed and learned content
# ---------------------------------------------------------------------------


def test_to_block_contains_fixed_and_learned() -> None:
    block = seed_character()
    rendered = block.to_block()
    assert "candor" in rendered.lower() or "truth" in rendered.lower()
    assert block.learned.tone in rendered


def test_to_block_contains_interests_when_set() -> None:
    block = seed_character()
    block.learned.interests = ["philosophy", "running"]
    rendered = block.to_block()
    assert "philosophy" in rendered
    assert "running" in rendered


# ---------------------------------------------------------------------------
# load_or_seed seeds fresh file when none exists
# ---------------------------------------------------------------------------


def test_load_or_seed_creates_file_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "character" / "persona.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists()

    block = load_or_seed(path)
    assert path.exists()
    assert block.fixed.candor == FixedTraits().candor
