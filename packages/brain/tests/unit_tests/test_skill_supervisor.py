"""Tests for the MCP skill supervisor.

Uses a tiny in-tree fake skill written on demand to ``tmp_path`` so we
never spawn the real june_skill_* servers from these tests. The fake is
a standalone script that uses :class:`MCPStdioServer`, so we exercise
the real stdio handshake + tools/list + tools/call paths.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
from june_brain.skills.manifest import SkillManifest, SkillManifestEntry
from june_brain.skills.supervisor import (
    SkillStatus,
    SkillSupervisor,
    resolve_spawn_argv,
)


def _write_fake_skill(tmp_path: Path, *, crash_after: int = 0) -> Path:
    """Write a minimal MCP skill script and return its path.

    When ``crash_after`` > 0 the skill exits after the Nth tools/call so
    we can verify the supervisor restarts on crash.
    """
    script = tmp_path / "fake_skill.py"
    # Import june_brain.skills.server directly from this repo's src tree so
    # the fake does not require any package install step.
    src_root = Path(__file__).resolve().parents[2] / "src"
    script.write_text(
        textwrap.dedent(
            f"""
            import os, sys
            sys.path.insert(0, {str(src_root)!r})
            from june_brain.skills.server import MCPStdioServer

            server = MCPStdioServer(name="fake", version="0.0.1")

            _calls = {{"echo": 0}}
            _crash_after = {crash_after}

            @server.tool(
                name="echo",
                description="Echo back the input value.",
                input_schema={{
                    "type": "object",
                    "properties": {{
                        "value": {{"type": "string"}},
                    }},
                    "required": ["value"],
                }},
            )
            def echo(value: str) -> str:
                _calls["echo"] += 1
                if _crash_after and _calls["echo"] >= _crash_after:
                    os._exit(1)
                return f"echoed:{{value}}"

            server.run()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return script


def _manifest_for(script: Path) -> SkillManifest:
    return SkillManifest(
        entries={
            "fake": SkillManifestEntry(
                key="fake",
                enabled=True,
                command=sys.executable,
                args=[str(script)],
                description="test skill",
            )
        }
    )


@pytest.fixture
def no_autostart(monkeypatch):
    """Prevent the module-level supervisor from auto-starting real skills."""
    monkeypatch.setenv("JUNE_SKILLS_DISABLED", "1")
    yield


def test_start_spawns_and_lists_tools(tmp_path, no_autostart):
    script = _write_fake_skill(tmp_path)
    supervisor = SkillSupervisor(_manifest_for(script))
    try:
        supervisor.start()
        statuses = supervisor.list_status()
        assert len(statuses) == 1
        entry = statuses[0]
        assert entry["key"] == "fake"
        assert entry["status"] == SkillStatus.RUNNING.value
        assert [t["name"] for t in entry["tools"]] == ["echo"]
    finally:
        supervisor.stop()


def test_enabled_tools_returns_bridged_tool(tmp_path, no_autostart):
    script = _write_fake_skill(tmp_path)
    supervisor = SkillSupervisor(_manifest_for(script))
    try:
        supervisor.start()
        tools = supervisor.enabled_tools()
        assert [t.name for t in tools] == ["echo"]
        result = tools[0].invoke({"value": "hello"})
        assert "echoed:hello" in result
    finally:
        supervisor.stop()


def test_crashed_skill_is_restarted_on_next_call(tmp_path, no_autostart):
    # Crash on the second call so the first call succeeds, the crash
    # happens, and the third call forces a restart + retry path.
    script = _write_fake_skill(tmp_path, crash_after=2)
    supervisor = SkillSupervisor(_manifest_for(script))
    try:
        supervisor.start()
        tool = supervisor.enabled_tools()[0]

        first = tool.invoke({"value": "a"})
        assert "echoed:a" in first

        # Second call triggers os._exit before returning; supervisor's
        # retry spawns a fresh subprocess and the retry succeeds.
        second = tool.invoke({"value": "b"})
        assert "echoed:b" in second

        assert supervisor._skills["fake"].status == SkillStatus.RUNNING
    finally:
        supervisor.stop()


def test_set_enabled_flips_state_and_persists(tmp_path, no_autostart, monkeypatch):
    # Route manifest_path() at the supervisor's save step to a tmp file.
    monkeypatch.setenv("JUNE_CONFIG_ROOT", str(tmp_path))

    script = _write_fake_skill(tmp_path)
    supervisor = SkillSupervisor(_manifest_for(script))
    try:
        supervisor.start()
        assert supervisor._skills["fake"].status == SkillStatus.RUNNING

        skill = supervisor.set_enabled("fake", False)
        assert skill is not None
        assert skill.entry.enabled is False
        assert skill.status == SkillStatus.DISABLED
        # After disabling, the bridged tool list drops this skill.
        assert supervisor.enabled_tools() == []

        # Persisted manifest reflects the flip.
        persisted = (tmp_path / "skills.toml").read_text(encoding="utf-8")
        assert "enabled = false" in persisted
    finally:
        supervisor.stop()


def test_set_enabled_unknown_key_returns_none(no_autostart):
    supervisor = SkillSupervisor(SkillManifest())
    assert supervisor.set_enabled("nope", True) is None


# --------------------------------------------------------------- frozen spawn argv


def _first_party_entry() -> SkillManifestEntry:
    return SkillManifestEntry(
        key="calendar",
        command=sys.executable,
        args=["-m", "june_skill_calendar"],
    )


def test_resolve_spawn_argv_dev_mode_leaves_first_party_unchanged(monkeypatch):
    # Not a frozen build: the portable `-m june_skill_<key>` form is used as-is.
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    entry = _first_party_entry()
    assert resolve_spawn_argv(entry) == [sys.executable, "-m", "june_skill_calendar"]


def test_resolve_spawn_argv_frozen_translates_first_party(monkeypatch):
    # Simulate the frozen sidecar: sys.frozen True and sys.executable is the
    # frozen june-api binary. First-party `-m` form becomes `--run-skill <key>`,
    # with the key taken from the module name (not entry.command).
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/Frozen/june-api", raising=False)
    entry = SkillManifestEntry(
        key="calendar",
        command="/some/original/python",
        args=["-m", "june_skill_calendar"],
    )
    assert resolve_spawn_argv(entry) == ["/Frozen/june-api", "--run-skill", "calendar"]


def test_resolve_spawn_argv_frozen_leaves_third_party_unchanged(monkeypatch):
    # A third-party skill (or any non `-m june_skill_*` form) is never rewritten,
    # even in a frozen build.
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/Frozen/june-api", raising=False)
    entry = SkillManifestEntry(
        key="weather",
        command="/usr/local/bin/weather-mcp",
        args=["--stdio", "--port", "0"],
    )
    assert resolve_spawn_argv(entry) == ["/usr/local/bin/weather-mcp", "--stdio", "--port", "0"]


def test_resolve_spawn_argv_frozen_leaves_bare_script_unchanged(monkeypatch):
    # A `python script.py` style entry (no `-m`) is not a first-party module launch.
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    entry = SkillManifestEntry(
        key="custom",
        command=sys.executable,
        args=["/opt/custom_skill.py"],
    )
    assert resolve_spawn_argv(entry) == [sys.executable, "/opt/custom_skill.py"]
