"""Skill manifest — lists installed MCP skill servers and their enabled state.

The manifest lives at ``~/Library/Application Support/June/skills.toml`` on
macOS (and the XDG equivalent elsewhere). If the file is missing, a default
manifest is materialized on first read so users never have to hand-author it.

Each entry is:

    [skill.research]
    enabled = true
    command = "python"
    args = ["-m", "june_skill_research"]
    env = { BRAVE_SEARCH_API_KEY = "..." }

``command`` + ``args`` tell the supervisor how to spawn the MCP stdio server.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


DEFAULT_MODEL_POLICY = "cloud_allowed"
_VALID_MODEL_POLICIES = frozenset({"local_only", "cloud_allowed", "cloud_required"})

# Default per-JSON-RPC-call timeout. Skills that need longer (research,
# browser, anything hitting slow networks) can override per-entry in
# skills.toml via ``response_timeout_seconds``.
DEFAULT_RESPONSE_TIMEOUT_SECONDS = 30.0


@dataclass
class SkillManifestEntry:
    """One skill's configuration in the manifest."""

    key: str
    enabled: bool = True
    daemon: bool = False
    command: str = sys.executable
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""
    disabled_tools: list[str] = field(default_factory=list)
    model_policy: str = DEFAULT_MODEL_POLICY
    response_timeout_seconds: float = DEFAULT_RESPONSE_TIMEOUT_SECONDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "daemon": self.daemon,
            "command": self.command,
            "args": list(self.args),
            "env": dict(self.env),
            "description": self.description,
            "disabled_tools": list(self.disabled_tools),
            "model_policy": self.model_policy,
            "response_timeout_seconds": self.response_timeout_seconds,
        }

    def policy_enum(self):  # type: ignore[no-untyped-def]
        """Return the ``SkillModelPolicy`` enum equivalent of ``model_policy``."""
        from ..routing import SkillModelPolicy

        try:
            return SkillModelPolicy(self.model_policy)
        except ValueError:
            return SkillModelPolicy.CLOUD_ALLOWED


@dataclass
class SkillManifest:
    """Parsed skills.toml."""

    entries: dict[str, SkillManifestEntry] = field(default_factory=dict)

    def enabled_entries(self) -> list[SkillManifestEntry]:
        return [e for e in self.entries.values() if e.enabled]

    def get(self, key: str) -> SkillManifestEntry | None:
        return self.entries.get(key)

    def set_enabled(self, key: str, enabled: bool) -> SkillManifestEntry | None:
        entry = self.entries.get(key)
        if entry is None:
            return None
        entry.enabled = enabled
        return entry

    def set_tool_enabled(
        self, key: str, tool_name: str, enabled: bool
    ) -> SkillManifestEntry | None:
        entry = self.entries.get(key)
        if entry is None:
            return None
        tool_name = tool_name.strip()
        if not tool_name:
            return entry
        if enabled:
            entry.disabled_tools = [t for t in entry.disabled_tools if t != tool_name]
        elif tool_name not in entry.disabled_tools:
            entry.disabled_tools = [*entry.disabled_tools, tool_name]
        return entry


# The five Week-5 skills. Each assumes the skill package is installed in the
# same venv as the brain, so we invoke it as "python -m <module>".
DEFAULT_MANIFEST: SkillManifest = SkillManifest(
    entries={
        "calendar": SkillManifestEntry(
            key="calendar",
            enabled=True,
            command=sys.executable,
            args=["-m", "june_skill_calendar"],
            description="Calendar events, reminders, and birthdays.",
        ),
        "health": SkillManifestEntry(
            key="health",
            enabled=True,
            command=sys.executable,
            args=["-m", "june_skill_health"],
            description="Body metrics, workouts, water, and habits.",
        ),
        "research": SkillManifestEntry(
            key="research",
            enabled=True,
            command=sys.executable,
            args=["-m", "june_skill_research"],
            description="Web search via Brave Search or DuckDuckGo.",
            # Network fetches can be slow; give the research skill more headroom.
            response_timeout_seconds=60.0,
        ),
        "files": SkillManifestEntry(
            key="files",
            enabled=True,
            command=sys.executable,
            args=["-m", "june_skill_files"],
            description=(
                "Sandboxed filesystem under your HOME: list directories, read text "
                "files and PDFs, search by filename or content. Also extracts clean "
                "text from webpages."
            ),
            model_policy="local_only",
        ),
        "daily": SkillManifestEntry(
            key="daily",
            enabled=True,
            command=sys.executable,
            args=["-m", "june_skill_daily"],
            description="Journaling, moods, goals, and chapter intake.",
        ),
        "telegram": SkillManifestEntry(
            key="telegram",
            enabled=False,
            daemon=True,
            command=sys.executable,
            args=["-m", "june_skill_telegram"],
            description="Telegram bridge — send/receive messages from your bot. Requires JUNE_TELEGRAM_BOT_TOKEN.",
        ),
    }
)


def _default_config_root() -> Path:
    """Return the user's June configuration root."""
    env_override = os.environ.get("JUNE_CONFIG_ROOT")
    if env_override:
        return Path(env_override)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "June"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "june"


def manifest_path(root: Path | None = None) -> Path:
    """Return the manifest file path, creating the parent directory if needed."""
    root = root or _default_config_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / "skills.toml"


def _serialize(manifest: SkillManifest) -> str:
    """Hand-written TOML writer (avoids adding tomli-w as a dep)."""
    lines: list[str] = [
        "# June skills manifest — edit to enable/disable skills or change launch args.",
        "# Documented at docs/decisions/0005-skills-as-mcp.md.",
        "",
    ]
    for key, entry in manifest.entries.items():
        lines.append(f"[skill.{key}]")
        lines.append(f"enabled = {str(entry.enabled).lower()}")
        lines.append(f'command = "{entry.command}"')
        args = ", ".join(f'"{a}"' for a in entry.args)
        lines.append(f"args = [{args}]")
        if entry.description:
            lines.append(f'description = "{entry.description}"')
        if entry.daemon:
            lines.append("daemon = true")
        if entry.env:
            env_inline = ", ".join(f'{k} = "{v}"' for k, v in entry.env.items())
            lines.append(f"env = {{ {env_inline} }}")
        if entry.disabled_tools:
            tools_inline = ", ".join(f'"{t}"' for t in entry.disabled_tools)
            lines.append(f"disabled_tools = [{tools_inline}]")
        if entry.model_policy and entry.model_policy != DEFAULT_MODEL_POLICY:
            lines.append(f'model_policy = "{entry.model_policy}"')
        if entry.response_timeout_seconds != DEFAULT_RESPONSE_TIMEOUT_SECONDS:
            lines.append(f"response_timeout_seconds = {entry.response_timeout_seconds}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_manifest(path: Path | None = None) -> SkillManifest:
    """Load the manifest, materializing a default file if missing.

    Missing-entry recovery: if the on-disk manifest lacks one of the five
    DEFAULT_MANIFEST skills, the missing entry is added with its default
    (enabled) config. Existing entries are never overwritten.
    """
    target = path or manifest_path()
    if not target.exists():
        save_manifest(DEFAULT_MANIFEST, target)
        return SkillManifest(
            entries={k: _copy_entry(v) for k, v in DEFAULT_MANIFEST.entries.items()}
        )

    try:
        with target.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception:
        logger.exception("Failed to parse %s; falling back to default manifest.", target)
        return SkillManifest(
            entries={k: _copy_entry(v) for k, v in DEFAULT_MANIFEST.entries.items()}
        )

    manifest = SkillManifest()
    skill_block = data.get("skill") or {}
    for key, raw in skill_block.items():
        if not isinstance(raw, dict):
            continue
        default = DEFAULT_MANIFEST.entries.get(key)
        raw_policy = str(raw.get("model_policy") or "").strip().lower()
        if raw_policy not in _VALID_MODEL_POLICIES:
            raw_policy = default.model_policy if default else DEFAULT_MODEL_POLICY
        raw_timeout = raw.get("response_timeout_seconds")
        try:
            timeout = float(raw_timeout) if raw_timeout is not None else None
        except (TypeError, ValueError):
            timeout = None
        if timeout is None or timeout <= 0:
            timeout = (
                default.response_timeout_seconds
                if default
                else DEFAULT_RESPONSE_TIMEOUT_SECONDS
            )
        manifest.entries[key] = SkillManifestEntry(
            key=key,
            enabled=bool(raw.get("enabled", True)),
            daemon=bool(raw.get("daemon", default.daemon if default else False)),
            command=str(raw.get("command") or (default.command if default else sys.executable)),
            args=list(raw.get("args") or (default.args if default else [])),
            env=dict(raw.get("env") or {}),
            description=str(raw.get("description") or (default.description if default else "")),
            disabled_tools=[str(t) for t in (raw.get("disabled_tools") or [])],
            model_policy=raw_policy,
            response_timeout_seconds=timeout,
        )

    for key, default_entry in DEFAULT_MANIFEST.entries.items():
        if key not in manifest.entries:
            manifest.entries[key] = _copy_entry(default_entry)

    return manifest


def save_manifest(manifest: SkillManifest, path: Path | None = None) -> Path:
    """Write the manifest to disk and return the path."""
    target = path or manifest_path()
    target.write_text(_serialize(manifest), encoding="utf-8")
    return target


def _copy_entry(entry: SkillManifestEntry) -> SkillManifestEntry:
    return SkillManifestEntry(
        key=entry.key,
        enabled=entry.enabled,
        daemon=entry.daemon,
        command=entry.command,
        args=list(entry.args),
        env=dict(entry.env),
        description=entry.description,
        disabled_tools=list(entry.disabled_tools),
        model_policy=entry.model_policy,
        response_timeout_seconds=entry.response_timeout_seconds,
    )
