"""Versioned, self-describing manifest for June's data directory.

On startup call load_or_init() to get (manifest, fresh). The fresh flag is True
when the manifest was missing, corrupt, or unreadable — the caller surfaces that
to the UI so the user knows the data dir was just created or recovered.
"""

from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .layout import EXPECTED_CONTENTS, ensure_layout, manifest_path

SCHEMA_VERSION = 1


@dataclass
class Manifest:
    schema_version: int
    created_at: str
    june_version: str
    contents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "june_version": self.june_version,
            "contents": list(self.contents),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Manifest:
        return cls(
            schema_version=int(data["schema_version"]),  # type: ignore[arg-type]
            created_at=str(data["created_at"]),
            june_version=str(data["june_version"]),
            contents=list(data.get("contents", [])),  # type: ignore[arg-type]
        )


def june_version() -> str:
    """Return the installed june-brain package version, or '0.0.0' if undetectable."""
    try:
        return importlib.metadata.version("june-brain")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def read_manifest() -> Manifest | None:
    """Read manifest.json and return a Manifest, or None if missing/unparseable."""
    path = manifest_path()
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            return None
        return Manifest.from_dict(data)
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def write_manifest(manifest: Manifest) -> Path:
    """Write manifest.json atomically and return its path."""
    ensure_layout()
    path = manifest_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def init_data_dir() -> Manifest:
    """Create the layout, write a fresh manifest, and return it."""
    ensure_layout()
    now = datetime.now(UTC).isoformat()
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        created_at=now,
        june_version=june_version(),
        contents=list(EXPECTED_CONTENTS),
    )
    write_manifest(manifest)
    return manifest


# Migration registry — keyed by the schema_version the migration upgrades FROM.
# To add a real migration: _MIGRATIONS[old_version] = fn(manifest) -> manifest
_MIGRATIONS: dict[int, Callable[[Manifest], Manifest]] = {}


def _migrate(manifest: Manifest) -> Manifest:
    """Apply all registered migrations in order and bump schema_version."""
    for version in range(manifest.schema_version, SCHEMA_VERSION):
        migration = _MIGRATIONS.get(version)
        if migration is not None:
            manifest = migration(manifest)
    manifest.schema_version = SCHEMA_VERSION
    return manifest


def load_or_init() -> tuple[Manifest, bool]:
    """Load the manifest, migrating or reinitialising as needed.

    Returns (manifest, fresh) where fresh=True means the data dir was just
    created or recovered from a corrupt/missing manifest.
    """
    existing = read_manifest()

    if existing is None:
        return init_data_dir(), True

    if existing.schema_version < SCHEMA_VERSION:
        migrated = _migrate(existing)
        write_manifest(migrated)
        return migrated, False

    return existing, False
