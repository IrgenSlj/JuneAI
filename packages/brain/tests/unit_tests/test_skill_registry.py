"""Tests for the MCP registry connector (ADR 0010, Sprint 1.6)."""

from __future__ import annotations

from pathlib import Path

import pytest
from june_brain.skills.manifest import load_manifest
from june_brain.skills.registry import (
    install_from_registry,
    installed_keys,
    load_registry,
    resolve_secret,
    uninstall_from_manifest,
)


def test_load_registry_returns_entries_from_bundled_index() -> None:
    registry = load_registry()
    assert registry.schema_version == 1
    assert len(registry.entries) >= 3
    # Every entry must have a key, command, and a model policy.
    for entry in registry.entries:
        assert entry.key
        assert entry.install.command
        assert entry.model_policy in {"local_only", "cloud_allowed", "cloud_required"}


def test_registry_find_returns_matching_entry() -> None:
    registry = load_registry()
    target = registry.entries[0]
    assert registry.find(target.key) is target
    assert registry.find("never-registered") is None


def test_install_from_registry_writes_manifest_entry(tmp_path: Path) -> None:
    target = tmp_path / "skills.toml"
    # Bootstrap the manifest so the supervisor's default skills exist.
    load_manifest(target)

    registry = load_registry()
    pick = registry.entries[0]

    entry = install_from_registry(pick.key, registry=registry, manifest_path=target)
    assert entry.key == pick.key
    assert entry.command == pick.install.command
    assert entry.args == pick.install.args
    assert entry.model_policy == pick.model_policy

    # Confirm it lives in the on-disk manifest.
    reloaded = load_manifest(target)
    assert pick.key in reloaded.entries
    assert reloaded.entries[pick.key].command == pick.install.command


def test_install_from_registry_rejects_unknown_key(tmp_path: Path) -> None:
    target = tmp_path / "skills.toml"
    load_manifest(target)
    with pytest.raises(KeyError):
        install_from_registry("never-registered", manifest_path=target)


def test_install_from_registry_rejects_duplicate(tmp_path: Path) -> None:
    target = tmp_path / "skills.toml"
    load_manifest(target)
    registry = load_registry()
    pick = registry.entries[0]

    install_from_registry(pick.key, registry=registry, manifest_path=target)
    with pytest.raises(ValueError):
        install_from_registry(pick.key, registry=registry, manifest_path=target)


def test_uninstall_from_manifest_removes_entry(tmp_path: Path) -> None:
    target = tmp_path / "skills.toml"
    load_manifest(target)
    registry = load_registry()
    pick = registry.entries[0]

    install_from_registry(pick.key, registry=registry, manifest_path=target)
    assert uninstall_from_manifest(pick.key, manifest_path=target) is True
    reloaded = load_manifest(target)
    assert pick.key not in reloaded.entries
    # Second uninstall is a no-op.
    assert uninstall_from_manifest(pick.key, manifest_path=target) is False


def test_installed_keys_includes_first_party_and_registry(tmp_path: Path) -> None:
    target = tmp_path / "skills.toml"
    manifest = load_manifest(target)
    # First-party defaults should already be there.
    first_party_count = len(manifest.entries)
    assert first_party_count >= 1

    registry = load_registry()
    pick = registry.entries[0]
    install_from_registry(pick.key, registry=registry, manifest_path=target)

    reloaded = load_manifest(target)
    keys = installed_keys(reloaded)
    assert pick.key in keys
    assert len(keys) == first_party_count + 1


# ---------------------------------------------------------------------------
# resolve_secret — secret resolution precedence
# ---------------------------------------------------------------------------


def test_resolve_secret_keychain_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keychain value takes priority over a matching env var."""
    import june_brain.secret_store as ss

    monkeypatch.setattr(ss, "load_secret", lambda name: "keychain-value")
    monkeypatch.setenv("GITHUB_PERSONAL_ACCESS_TOKEN", "env-value")
    assert resolve_secret("GITHUB_PERSONAL_ACCESS_TOKEN") == "keychain-value"


def test_resolve_secret_env_fallback_when_keychain_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the keychain has no entry, the env var is returned."""
    import june_brain.secret_store as ss

    monkeypatch.setattr(ss, "load_secret", lambda name: None)
    monkeypatch.setenv("GITHUB_PERSONAL_ACCESS_TOKEN", "env-value")
    assert resolve_secret("GITHUB_PERSONAL_ACCESS_TOKEN") == "env-value"


def test_resolve_secret_empty_when_neither_source_has_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty string is returned when both keychain and env are absent."""
    import june_brain.secret_store as ss

    monkeypatch.setattr(ss, "load_secret", lambda name: None)
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
    assert resolve_secret("GITHUB_PERSONAL_ACCESS_TOKEN") == ""


def test_install_from_registry_env_uses_keychain_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """install_from_registry builds env dict using keychain-first resolution."""
    import june_brain.secret_store as ss

    keychain = {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_keychain123"}
    monkeypatch.setattr(ss, "load_secret", lambda name: keychain.get(name))
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)

    target = tmp_path / "skills.toml"
    load_manifest(target)

    registry = load_registry()
    github_entry = registry.find("github-mcp")
    assert github_entry is not None, "github-mcp must be in the bundled index"

    entry = install_from_registry("github-mcp", registry=registry, manifest_path=target)
    assert entry.env.get("GITHUB_PERSONAL_ACCESS_TOKEN") == "ghp_keychain123"


def test_install_from_registry_env_falls_back_to_plain_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """install_from_registry falls back to env var when keychain returns None."""
    import june_brain.secret_store as ss

    monkeypatch.setattr(ss, "load_secret", lambda name: None)
    monkeypatch.setenv("GITHUB_PERSONAL_ACCESS_TOKEN", "ghp_plainenv456")

    target = tmp_path / "skills.toml"
    load_manifest(target)

    registry = load_registry()
    entry = install_from_registry("github-mcp", registry=registry, manifest_path=target)
    assert entry.env.get("GITHUB_PERSONAL_ACCESS_TOKEN") == "ghp_plainenv456"
