"""Tests for the skills manifest — on-disk config for enabled MCP servers."""

from __future__ import annotations

from pathlib import Path

from june_brain.skills.manifest import (
    DEFAULT_MANIFEST,
    SkillManifest,
    SkillManifestEntry,
    check_scope_drift,
    load_manifest,
    save_manifest,
)


def test_load_creates_default_manifest_when_missing(tmp_path: Path) -> None:
    target = tmp_path / "skills.toml"
    manifest = load_manifest(target)
    assert target.exists(), "missing manifest should be materialized"
    assert set(manifest.entries.keys()) == set(DEFAULT_MANIFEST.entries.keys())
    # Defaults ship enabled so a fresh user sees all skills.
    # Daemon skills (e.g. telegram) are disabled by default until configured.
    for entry in manifest.entries.values():
        if entry.daemon:
            continue
        assert entry.enabled, f"non-daemon skill {entry.key} should be enabled"


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


# ---------------------------------------------------------------------------
# check_scope_drift — pure drift detection
# ---------------------------------------------------------------------------


def test_drift_no_declared_always_clean() -> None:
    """When no scopes are declared the function returns has_drift=False regardless of derived."""
    drift = check_scope_drift(frozenset(), frozenset({"read_network", "write_local"}))
    assert not drift.has_drift
    assert drift.undeclared == frozenset()
    assert drift.unused == frozenset()


def test_drift_exact_match_no_drift() -> None:
    declared = frozenset({"read_local", "write_local"})
    derived = frozenset({"read_local", "write_local"})
    drift = check_scope_drift(declared, derived)
    assert not drift.has_drift
    assert drift.undeclared == frozenset()
    assert drift.unused == frozenset()


def test_drift_undeclared_detected() -> None:
    """Skill uses write_network but only declared read_local — violation."""
    declared = frozenset({"read_local"})
    derived = frozenset({"read_local", "write_network"})
    drift = check_scope_drift(declared, derived)
    assert drift.has_drift
    assert drift.undeclared == frozenset({"write_network"})
    assert drift.unused == frozenset()


def test_drift_unused_detected() -> None:
    """Skill declared execute but no tool maps to it — over-declaration."""
    declared = frozenset({"read_local", "execute"})
    derived = frozenset({"read_local"})
    drift = check_scope_drift(declared, derived)
    assert drift.has_drift
    assert drift.undeclared == frozenset()
    assert drift.unused == frozenset({"execute"})


def test_drift_both_undeclared_and_unused() -> None:
    declared = frozenset({"read_local", "execute"})
    derived = frozenset({"read_local", "write_network"})
    drift = check_scope_drift(declared, derived)
    assert drift.has_drift
    assert drift.undeclared == frozenset({"write_network"})
    assert drift.unused == frozenset({"execute"})


def test_declared_scopes_round_trips_through_toml(tmp_path: Path) -> None:
    """declared_scopes persists through save/load."""
    target = tmp_path / "skills.toml"
    manifest = load_manifest(target)
    manifest.entries["research"].declared_scopes = ["read_network"]
    save_manifest(manifest, target)

    reloaded = load_manifest(target)
    assert reloaded.entries["research"].declared_scopes == ["read_network"]


def test_invalid_declared_scopes_dropped_on_load(tmp_path: Path) -> None:
    """Unknown scope strings are silently dropped during load."""
    target = tmp_path / "skills.toml"
    target.write_text(
        '[skill.research]\nenabled = true\ncommand = "python"\nargs = ["-m", "june_skill_research"]'
        '\ndeclared_scopes = ["read_network", "fly_to_mars"]\n',
        encoding="utf-8",
    )
    manifest = load_manifest(target)
    assert manifest.entries["research"].declared_scopes == ["read_network"]
