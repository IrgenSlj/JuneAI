"""Public facade for the MCP skills subsystem.

Thin wrapper over :mod:`.supervisor` and :mod:`.manifest` so callers can
``from june_brain.skills.loader import load_skill_tools`` without poking at
supervisor internals. Names here are stable; supervisor internals are not.
"""

from __future__ import annotations

from ..tools_base import Tool
from .manifest import (
    SkillManifest,
    SkillManifestEntry,
    load_manifest,
    manifest_path,
    save_manifest,
)
from .supervisor import (
    SkillProcess,
    SkillStatus,
    SkillSupervisor,
    get_supervisor,
    load_skill_tools,
    reset_supervisor_for_tests,
)

__all__ = [
    "SkillManifest",
    "SkillManifestEntry",
    "SkillProcess",
    "SkillStatus",
    "SkillSupervisor",
    "get_supervisor",
    "load_manifest",
    "load_skill_tools",
    "manifest_path",
    "reset_supervisor_for_tests",
    "save_manifest",
]


def list_status() -> list[dict[str, object]]:
    """Snapshot of every skill's lifecycle state for /skills endpoints."""
    return get_supervisor().list_status()


def set_skill_enabled(key: str, enabled: bool) -> SkillProcess | None:
    """Toggle a skill and persist the change to ~/Library/Application Support/June/skills.toml."""
    return get_supervisor().set_enabled(key, enabled)


def set_skill_tool_enabled(
    key: str, tool_name: str, enabled: bool
) -> SkillProcess | None:
    """Toggle one tool of a skill on/off and persist the change."""
    return get_supervisor().set_tool_enabled(key, tool_name, enabled)


def reload_skills() -> None:
    """Re-read the manifest from disk and reconcile running processes."""
    get_supervisor().reload()


def available_tools() -> list[Tool]:
    """Alias of :func:`load_skill_tools` for symmetry with ``available_*`` helpers."""
    return load_skill_tools()


def call_skill_tool(skill_key: str, tool_name: str, arguments: dict[str, object]) -> str:
    """Invoke a single tool on a skill subprocess and return its stringified result.

    Used by the /skills playground endpoint. Crashes are caught and reported
    via the returned string — same surface as the supervisor's bridged tool.
    """
    return get_supervisor().call_tool(skill_key, tool_name, dict(arguments))
