"""Trust Ledger scope-review entry written at skill install (Track 4, item 3).

Verifies that ``install_from_registry`` appends a ``system`` ledger entry
recording the scope review, and that a ledger failure never breaks the install.
The global conftest ``_isolate_data_dir`` fixture provides MEMORY_DIR isolation
for every test in this file automatically.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from june_brain.skills.manifest import load_manifest
from june_brain.skills.registry import (
    Registry,
    RegistryEntry,
    RegistryInstallSpec,
    install_from_registry,
)
from june_brain.trust import LedgerReader


def _mock_registry(
    *,
    key: str = "test-skill",
    publisher: str = "test-publisher",
    verified: bool = True,
    model_policy: str = "local_only",
    tools_preview: list[str] | None = None,
) -> Registry:
    """Build a one-entry mock Registry for testing (no disk I/O)."""
    entry = RegistryEntry(
        key=key,
        name="Test Skill",
        description="A skill for testing.",
        homepage="https://example.com",
        publisher=publisher,
        verified=verified,
        model_policy=model_policy,
        install=RegistryInstallSpec(
            kind="npx",
            package="@test/test-skill",
            command="npx",
            args=["-y", "@test/test-skill"],
        ),
        tools_preview=tools_preview if tools_preview is not None else ["tool_a", "tool_b"],
    )
    return Registry(entries=[entry])


# ---------------------------------------------------------------------------
# Scope-review entry is written to the ledger on install
# ---------------------------------------------------------------------------


def test_install_records_one_system_scope_review_entry(tmp_path: Path) -> None:
    """A fresh install appends exactly one 'system' ledger entry of kind skill_scope_review."""
    target = tmp_path / "skills.toml"
    load_manifest(target)
    registry = _mock_registry()

    install_from_registry("test-skill", registry=registry, manifest_path=target)

    reader = LedgerReader()
    system_entries = [e for e in reader.page(limit=100) if e.kind == "system"]
    assert len(system_entries) == 1
    assert system_entries[0].actor == "june"
    assert reader.verify_chain().ok is True


def test_install_scope_review_payload_contains_skill_key(tmp_path: Path) -> None:
    target = tmp_path / "skills.toml"
    load_manifest(target)

    install_from_registry("test-skill", registry=_mock_registry(), manifest_path=target)

    reader = LedgerReader()
    entries = [e for e in reader.page(limit=100) if e.kind == "system"]
    payload = entries[0].payload
    assert payload["event"] == "skill_scope_review"
    assert payload["skill_key"] == "test-skill"


def test_install_scope_review_payload_declared_scopes_empty_at_install(
    tmp_path: Path,
) -> None:
    """declared_scopes is [] at install: RegistryEntry carries none and tools are
    not discovered until first spawn.  The ledger records what is honestly known."""
    target = tmp_path / "skills.toml"
    load_manifest(target)

    install_from_registry("test-skill", registry=_mock_registry(), manifest_path=target)

    reader = LedgerReader()
    entries = [e for e in reader.page(limit=100) if e.kind == "system"]
    assert entries[0].payload["declared_scopes"] == []


def test_install_scope_review_payload_registry_metadata(tmp_path: Path) -> None:
    """publisher, verified, model_policy, and tools_preview are all recorded."""
    target = tmp_path / "skills.toml"
    load_manifest(target)
    registry = _mock_registry(
        publisher="acme-corp",
        verified=False,
        model_policy="cloud_allowed",
        tools_preview=["search", "fetch"],
    )

    install_from_registry("test-skill", registry=registry, manifest_path=target)

    reader = LedgerReader()
    entries = [e for e in reader.page(limit=100) if e.kind == "system"]
    payload = entries[0].payload
    assert payload["publisher"] == "acme-corp"
    assert payload["verified"] is False
    assert payload["model_policy"] == "cloud_allowed"
    assert payload["tools_preview"] == ["search", "fetch"]


# ---------------------------------------------------------------------------
# Best-effort guard: ledger failure must never break skill install
# ---------------------------------------------------------------------------


def test_install_succeeds_when_ledger_append_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ledger write failure is swallowed; the manifest entry is still created."""
    import june_brain.trust as trust_mod

    def _raising_get_writer():
        raise RuntimeError("simulated ledger failure")

    monkeypatch.setattr(trust_mod, "get_writer", _raising_get_writer)

    target = tmp_path / "skills.toml"
    load_manifest(target)
    registry = _mock_registry()

    entry = install_from_registry("test-skill", registry=registry, manifest_path=target)

    assert entry.key == "test-skill"
    reloaded = load_manifest(target)
    assert "test-skill" in reloaded.entries
