"""Single source of truth for all June data-directory paths.

All paths are resolved lazily at call time by reading june_brain.config.MEMORY_DIR,
so tests that monkeypatch that attribute get the right paths without any import-time
capture.
"""

from __future__ import annotations

from pathlib import Path

# Canonical subdirectories that make up a well-formed data dir.
# manifest.py reads EXPECTED_CONTENTS to populate the manifest's contents field.
EXPECTED_CONTENTS: list[str] = [
    "memory/",
    "character/",
    "skills/",
    "tasks/",
    "config/",
]


def _root() -> Path:
    # Re-import on every call so monkeypatching june_brain.config.MEMORY_DIR works.
    import june_brain.config as _cfg  # noqa: PLC0415

    return Path(_cfg.MEMORY_DIR)


def data_dir() -> Path:
    """Root of June's data directory."""
    return _root()


def manifest_path() -> Path:
    """Path to the versioned manifest file."""
    return _root() / "manifest.json"


def memory_dir() -> Path:
    """Directory for SQLite db, ChromaDB store, and graph."""
    return _root() / "memory"


def character_dir() -> Path:
    """Directory for the character/persona files."""
    return _root() / "character"


def character_file() -> Path:
    """Path to the canonical persona JSON file."""
    return _root() / "character" / "persona.json"


def skills_dir() -> Path:
    """Directory for installed skill configs."""
    return _root() / "skills"


def tasks_dir() -> Path:
    """Directory for task-related files including the ledger."""
    return _root() / "tasks"


def ledger_path() -> Path:
    """Path to the append-only task/event ledger."""
    return _root() / "tasks" / "ledger.jsonl"


def config_dir() -> Path:
    """Directory for providers.toml, privacy mode, salience weights, thresholds."""
    return _root() / "config"


def ensure_layout() -> Path:
    """Create all required subdirectories under data_dir() and return data_dir().

    Safe to call repeatedly — uses exist_ok=True throughout.
    """
    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    for subdir in ("memory", "character", "skills", "tasks", "config"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return root
