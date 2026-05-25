"""MCP registry — third-party MCP servers June can install as skills.

Per ADR 0010, the registry positions June as the personal-MCP home base.
Users browse a curated list, install any entry in one click, and the
supervisor runs it the same way it runs first-party skills.

This module owns the registry data shape and a one-click install that
writes the entry into the user's ``skills.toml``. The supervisor's existing
lifecycle (start, restart-on-crash, agent reload) picks up the new skill on
the next reload.

Real signing and a hosted index land later. For now the registry is a JSON
file shipped with the brain package, and installs carry an ``unsigned``
flag in the manifest so the UI can show the appropriate warning.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from .manifest import (
    DEFAULT_MODEL_POLICY,
    SkillManifest,
    SkillManifestEntry,
    load_manifest,
    save_manifest,
)


@dataclass
class RegistryInstallSpec:
    """How to launch a registry entry as a subprocess."""

    kind: str  # "npx", "uvx", "python", "binary"
    package: str
    command: str
    args: list[str] = field(default_factory=list)
    env_required: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "package": self.package,
            "command": self.command,
            "args": list(self.args),
            "env_required": list(self.env_required),
        }


@dataclass
class RegistryEntry:
    """One entry in the MCP registry."""

    key: str
    name: str
    description: str
    homepage: str
    publisher: str
    verified: bool
    model_policy: str
    install: RegistryInstallSpec
    tools_preview: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "homepage": self.homepage,
            "publisher": self.publisher,
            "verified": self.verified,
            "model_policy": self.model_policy,
            "install": self.install.to_dict(),
            "tools_preview": list(self.tools_preview),
        }


@dataclass
class Registry:
    """Parsed registry index."""

    schema_version: int = 1
    source: str = "first-party-curated"
    updated_at: str = ""
    entries: list[RegistryEntry] = field(default_factory=list)

    def find(self, key: str) -> RegistryEntry | None:
        for entry in self.entries:
            if entry.key == key:
                return entry
        return None


_INDEX_FILENAME = "registry_index.json"


def load_registry() -> Registry:
    """Load the curated registry shipped with the brain package."""
    try:
        text = resources.files(__package__).joinpath(_INDEX_FILENAME).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError):
        return Registry(entries=[])
    return _parse_registry(text)


def _parse_registry(text: str) -> Registry:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return Registry(entries=[])
    raw_entries = data.get("entries") or []
    entries: list[RegistryEntry] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            continue
        install = raw.get("install") or {}
        entries.append(
            RegistryEntry(
                key=str(raw.get("key") or ""),
                name=str(raw.get("name") or raw.get("key") or ""),
                description=str(raw.get("description") or ""),
                homepage=str(raw.get("homepage") or ""),
                publisher=str(raw.get("publisher") or "unknown"),
                verified=bool(raw.get("verified", False)),
                model_policy=str(raw.get("model_policy") or DEFAULT_MODEL_POLICY),
                install=RegistryInstallSpec(
                    kind=str(install.get("kind") or "npx"),
                    package=str(install.get("package") or ""),
                    command=str(install.get("command") or ""),
                    args=[str(a) for a in (install.get("args") or [])],
                    env_required=[str(e) for e in (install.get("env_required") or [])],
                ),
                tools_preview=[str(t) for t in (raw.get("tools_preview") or [])],
            )
        )
    return Registry(
        schema_version=int(data.get("schema_version") or 1),
        source=str(data.get("source") or "first-party-curated"),
        updated_at=str(data.get("updated_at") or ""),
        entries=[e for e in entries if e.key],
    )


def install_from_registry(
    key: str,
    *,
    registry: Registry | None = None,
    manifest_path: Path | None = None,
) -> SkillManifestEntry:
    """Materialize a registry entry into the on-disk skill manifest.

    Returns the created ``SkillManifestEntry``. The caller is responsible for
    reloading the supervisor so the next chat turn sees the new tool surface.
    """
    reg = registry or load_registry()
    entry = reg.find(key)
    if entry is None:
        raise KeyError(f"no registry entry with key '{key}'")

    manifest = load_manifest(manifest_path)
    if key in manifest.entries:
        raise ValueError(f"skill '{key}' is already installed; uninstall first")

    new_entry = SkillManifestEntry(
        key=key,
        enabled=True,
        command=entry.install.command,
        args=list(entry.install.args),
        description=entry.description,
        env={var: os.environ.get(var, "") for var in entry.install.env_required},
        model_policy=entry.model_policy,
    )
    manifest.entries[key] = new_entry
    save_manifest(manifest, manifest_path)
    return new_entry


def uninstall_from_manifest(
    key: str,
    *,
    manifest_path: Path | None = None,
) -> bool:
    """Remove an installed registry skill from the manifest."""
    manifest = load_manifest(manifest_path)
    if key not in manifest.entries:
        return False
    del manifest.entries[key]
    save_manifest(manifest, manifest_path)
    return True


def installed_keys(manifest: SkillManifest | None = None) -> set[str]:
    """Set of skill keys currently in the manifest (first-party + registry)."""
    manifest = manifest or load_manifest()
    return set(manifest.entries.keys())
