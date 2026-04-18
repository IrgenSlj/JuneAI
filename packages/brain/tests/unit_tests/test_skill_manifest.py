"""Tests for the skills manifest — on-disk config for enabled MCP servers."""

from __future__ import annotations

from pathlib import Path

from june_brain.skills.manifest import (
    DEFAULT_MANIFEST,
    SkillManifest,
    SkillManifestEntry,
    load_manifest,
    save_manifest,
)


def test_load_creates_default_manifest_when_missing(tmp_path: Path) -> None:
    target = tmp_path / "skills.toml"
    manifest = load_manifest(target)
    assert target.exists(), "missing manifest should be materialized"
    assert set(manifest.entries.keys()) == set(DEFAULT_MANIFEST.entries.keys())
    # Defaults ship enabled so a fresh user sees all five skills.
    assert all(entry.enabled for entry in manifest.entries.values())


def test_save_then_load_round_trips_toggle(tmp_path: Path) -> None:
    target = tmp_path / "skills.toml"
    manifest = load_manifest(target)
    manifest.set_enabled("research", False)
    save_manifest(manifest, target)

    reloaded = load_manifest(target)
    assert reloaded.entries["research"].enabled is False
    # Other skills remain enabled.
    assert reloaded.entries["calendar"].enabled is True


def test_load_recovers_missing_default_entries(tmp_path: Path) -> None:
    """A pre-existing manifest with only a subset of skills should pick up new ones."""
    target = tmp_path / "skills.toml"
    partial = SkillManifest(
        entries={
            "calendar": SkillManifestEntry(
                key="calendar",
                enabled=False,
                command="python",
                args=["-m", "june_skill_calendar"],
                description="Calendar events.",
            ),
        }
    )
    save_manifest(partial, target)

    manifest = load_manifest(target)
    assert set(manifest.entries.keys()) == set(DEFAULT_MANIFEST.entries.keys())
    # User's explicit toggle wins over the default.
    assert manifest.entries["calendar"].enabled is False
    # Newly added defaults land enabled.
    assert manifest.entries["research"].enabled is True


def test_load_tolerates_malformed_toml(tmp_path: Path) -> None:
    target = tmp_path / "skills.toml"
    target.write_text("this is not = valid [toml\n", encoding="utf-8")
    manifest = load_manifest(target)
    # Falls back to the default set rather than exploding.
    assert set(manifest.entries.keys()) == set(DEFAULT_MANIFEST.entries.keys())


def test_set_enabled_returns_none_for_unknown_key() -> None:
    manifest = SkillManifest()
    assert manifest.set_enabled("nonexistent", True) is None
