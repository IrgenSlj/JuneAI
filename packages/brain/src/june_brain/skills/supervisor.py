"""MCP skill supervisor — spawns skill subprocesses and bridges their tools to LangGraph.

The supervisor speaks MCP over stdio (newline-delimited JSON-RPC 2.0). It:

1. **Spawns** each enabled skill as a subprocess (``command`` + ``args`` from
   the manifest).
2. **Handshakes** — sends ``initialize``, then ``notifications/initialized``.
3. **Discovers** each skill's tools via ``tools/list``.
4. **Executes** tool calls via ``tools/call`` with a single retry after
   respawn if the subprocess has died between calls.
5. **Bridges** each MCP tool to a LangChain ``StructuredTool`` so the
   existing ``ToolNode`` in ``graph.py`` can dispatch to it like any other
   tool. ``user_id`` is injected from the agent state — skills that touch
   memory receive it as a regular argument.

We deliberately do not use the ``mcp`` SDK here. The protocol surface we
need (initialize + tools/list + tools/call) is small, and a sync,
dep-free client keeps the supervisor easy to reason about and test.

Crash isolation: if a tool call fails because the subprocess died, the
supervisor logs the crash, restarts the skill, re-lists tools, and retries
the call once. If the retry also fails, it returns an error string — the
agent sees a normal tool error and keeps going. The chat loop never dies
because one skill did.
"""

from __future__ import annotations

import json
import logging
import os
import select
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Callable

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field, create_model

from .manifest import (
    DEFAULT_RESPONSE_TIMEOUT_SECONDS,
    SkillManifest,
    SkillManifestEntry,
    load_manifest,
)

logger = logging.getLogger(__name__)


# Fallback response timeout for any skill that didn't declare its own. Skills
# can override per-entry in skills.toml via ``response_timeout_seconds`` — the
# research skill, for example, needs more headroom for network fetches.
_RESPONSE_TIMEOUT_SECONDS = DEFAULT_RESPONSE_TIMEOUT_SECONDS


class SkillStatus(str, Enum):
    """Lifecycle state reported to the /skills endpoint."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    CRASHED = "crashed"
    DISABLED = "disabled"


@dataclass
class _MCPTool:
    """Tool metadata returned from ``tools/list``."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class SkillProcess:
    """Runtime state for one skill."""

    entry: SkillManifestEntry
    status: SkillStatus = SkillStatus.STOPPED
    proc: subprocess.Popen[bytes] | None = None
    tools: list[_MCPTool] = field(default_factory=list)
    error: str = ""
    _next_id: int = 1
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def key(self) -> str:
        return self.entry.key

    @property
    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


class SkillSupervisor:
    """Manages the lifecycle of all MCP skill subprocesses.

    Thread safety: per-skill calls serialize on each ``SkillProcess._lock``.
    Skill-set-level mutations (start/stop/reload/toggle) serialize on
    ``_mutation_lock``. Read-only snapshots (``list_status``, ``list_tools``)
    are safe without locks because they copy data.
    """

    def __init__(self, manifest: SkillManifest | None = None) -> None:
        self._manifest = manifest or load_manifest()
        self._skills: dict[str, SkillProcess] = {}
        self._mutation_lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Spawn every enabled skill in parallel. Errors on one skill never block the rest."""
        with self._mutation_lock:
            for entry in self._manifest.entries.values():
                self._skills.setdefault(entry.key, SkillProcess(entry=entry))
            for key, skill in self._skills.items():
                if not skill.entry.enabled:
                    skill.status = SkillStatus.DISABLED

        threads = []
        for skill in list(self._skills.values()):
            if not skill.entry.enabled:
                continue
            t = threading.Thread(target=self._spawn_worker, args=(skill,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def stop(self) -> None:
        """Terminate every running skill."""
        with self._mutation_lock:
            for skill in self._skills.values():
                self._terminate_locked(skill)

    def reload(self) -> None:
        """Re-read the manifest from disk and reconcile running state.

        - Newly enabled skills are spawned.
        - Newly disabled skills are terminated.
        - Unchanged skills keep their existing subprocess.
        """
        with self._mutation_lock:
            new_manifest = load_manifest()
            self._manifest = new_manifest
            for key, entry in new_manifest.entries.items():
                skill = self._skills.setdefault(key, SkillProcess(entry=entry))
                skill.entry = entry
                if entry.enabled and not skill.is_alive:
                    self._spawn_locked(skill)
                elif not entry.enabled and skill.is_alive:
                    self._terminate_locked(skill)
                    skill.status = SkillStatus.DISABLED

    def set_tool_enabled(
        self, key: str, tool_name: str, enabled: bool
    ) -> SkillProcess | None:
        """Toggle one tool of a skill on/off; persist; no respawn.

        Per-tool gates apply at agent-build time: the skill subprocess
        keeps running and continues to advertise the tool, but the
        supervisor filters it out of ``enabled_tools()`` so the agent
        never binds it. The caller is responsible for rebuilding the
        agent (e.g. ``brain_graph.reload_agent()``) after a successful
        flip.
        """
        entry = self._manifest.set_tool_enabled(key, tool_name, enabled)
        if entry is None:
            return None

        from .manifest import save_manifest

        save_manifest(self._manifest)
        with self._mutation_lock:
            skill = self._skills.setdefault(key, SkillProcess(entry=entry))
            skill.entry = entry
        return self._skills[key]

    def set_enabled(self, key: str, enabled: bool) -> SkillProcess | None:
        """Flip a skill's enabled flag, persist to disk, and reconcile."""
        entry = self._manifest.set_enabled(key, enabled)
        if entry is None:
            return None

        from .manifest import save_manifest

        save_manifest(self._manifest)
        with self._mutation_lock:
            skill = self._skills.setdefault(key, SkillProcess(entry=entry))
            skill.entry = entry
            if enabled and not skill.is_alive:
                self._spawn_locked(skill)
            elif not enabled and skill.is_alive:
                self._terminate_locked(skill)
                skill.status = SkillStatus.DISABLED
        return self._skills[key]

    # ------------------------------------------------------------------ introspection

    def list_status(self) -> list[dict[str, Any]]:
        """Snapshot suitable for the /skills endpoint."""
        out: list[dict[str, Any]] = []
        for skill in self._skills.values():
            disabled = set(skill.entry.disabled_tools or [])
            out.append(
                {
                    "key": skill.key,
                    "description": skill.entry.description,
                    "enabled": skill.entry.enabled,
                    "status": skill.status.value,
                    "error": skill.error,
                    "tools": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "enabled": t.name not in disabled,
                            "input_schema": dict(t.input_schema or {}),
                        }
                        for t in skill.tools
                    ],
                }
            )
        return out

    def enabled_tools(self) -> list[StructuredTool]:
        """Return LangChain tools for every running, enabled skill.

        Per-tool gates from the manifest's ``disabled_tools`` are honored
        here: a tool whose name appears in its skill's disabled list is
        excluded from the agent's bound tools, even if the skill itself
        is enabled and running.
        """
        tools: list[StructuredTool] = []
        for skill in self._skills.values():
            if not skill.entry.enabled or skill.status != SkillStatus.RUNNING:
                continue
            disabled = set(skill.entry.disabled_tools or [])
            for mcp_tool in skill.tools:
                if mcp_tool.name in disabled:
                    continue
                tools.append(self._bridge_tool(skill, mcp_tool))
        return tools

    # ------------------------------------------------------------------ spawning

    def _spawn_locked(self, skill: SkillProcess) -> None:
        """Start the skill subprocess and discover its tools. Caller holds the mutation lock."""
        if skill.is_alive:
            return
        skill.status = SkillStatus.STARTING
        skill.error = ""
        try:
            env = os.environ.copy()
            env.update(skill.entry.env)
            # Break the import recursion: a skill subprocess does
            # ``from june_brain.memory import Memory`` which re-enters the
            # brain package, which without this flag would try to build a new
            # agent and spawn five more skill subprocesses — a fork bomb that
            # can OOM a laptop in seconds. Both the subprocess guard and the
            # disabled supervisor are needed: one stops agent construction,
            # the other stops any code that still reaches the supervisor.
            env["JUNE_IS_SKILL_SUBPROCESS"] = "1"
            env["JUNE_SKILLS_DISABLED"] = "1"
            skill.proc = subprocess.Popen(  # noqa: S603 - command comes from user-owned manifest
                [skill.entry.command, *skill.entry.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                env=env,
            )
            self._handshake_locked(skill)
            skill.tools = self._list_tools_locked(skill)
            skill.status = SkillStatus.RUNNING
            logger.info(
                "Skill %r started (%d tools): %s",
                skill.key,
                len(skill.tools),
                ", ".join(t.name for t in skill.tools),
            )
        except Exception as exc:  # noqa: BLE001
            skill.status = SkillStatus.CRASHED
            skill.error = f"spawn failed: {exc}"
            logger.warning("Skill %r failed to start: %s", skill.key, exc)
            self._cleanup_proc(skill)

    def _spawn_worker(self, skill: SkillProcess) -> None:
        """Thread worker for parallel skill spawning. Errors are caught internally.

        We intentionally do NOT hold ``_mutation_lock`` during spawning: the
        subprocess handshake blocks on I/O (~2s per skill) and holding a lock
        would serialize all skills. Per-skill ``_lock`` guards against
        concurrent access from ``_call_tool`` during init.
        """
        with skill._lock:
            skill.status = SkillStatus.STARTING
            skill.error = ""
            try:
                env = os.environ.copy()
                env.update(skill.entry.env)
                env["JUNE_IS_SKILL_SUBPROCESS"] = "1"
                env["JUNE_SKILLS_DISABLED"] = "1"
                skill.proc = subprocess.Popen(
                    [skill.entry.command, *skill.entry.args],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    env=env,
                )
                self._handshake_locked(skill)
                skill.tools = self._list_tools_locked(skill)
                skill.status = SkillStatus.RUNNING
                logger.info(
                    "Skill %r started (%d tools): %s",
                    skill.key,
                    len(skill.tools),
                    ", ".join(t.name for t in skill.tools),
                )
            except Exception as exc:  # noqa: BLE001
                skill.status = SkillStatus.CRASHED
                skill.error = f"spawn failed: {exc}"
                logger.warning("Skill %r failed to start: %s", skill.key, exc)
                self._cleanup_proc(skill)

    def _terminate_locked(self, skill: SkillProcess) -> None:
        """Stop a running skill. Caller holds the mutation lock."""
        if skill.proc is None:
            skill.status = SkillStatus.STOPPED
            return
        try:
            skill.proc.terminate()
            try:
                skill.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                skill.proc.kill()
                skill.proc.wait(timeout=1.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error terminating skill %r: %s", skill.key, exc)
        finally:
            self._cleanup_proc(skill)
            skill.status = SkillStatus.STOPPED
            skill.tools = []

    def _cleanup_proc(self, skill: SkillProcess) -> None:
        if skill.proc is not None:
            for pipe in (skill.proc.stdin, skill.proc.stdout, skill.proc.stderr):
                try:
                    if pipe is not None:
                        pipe.close()
                except Exception:  # noqa: BLE001
                    pass
        skill.proc = None

    # ------------------------------------------------------------------ JSON-RPC

    def _handshake_locked(self, skill: SkillProcess) -> None:
        """Send initialize + notifications/initialized. Raises on protocol error."""
        init_result = self._request(
            skill,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "june-brain", "version": "0.1.0"},
            },
        )
        if not isinstance(init_result, dict) or "protocolVersion" not in init_result:
            raise RuntimeError(f"bad initialize response from {skill.key}: {init_result!r}")
        self._notify(skill, "notifications/initialized", {})

    def _list_tools_locked(self, skill: SkillProcess) -> list[_MCPTool]:
        result = self._request(skill, "tools/list", {})
        raw_tools = (result or {}).get("tools") if isinstance(result, dict) else []
        tools: list[_MCPTool] = []
        for raw in raw_tools or []:
            if not isinstance(raw, dict) or not raw.get("name"):
                continue
            tools.append(
                _MCPTool(
                    name=str(raw["name"]),
                    description=str(raw.get("description", "")),
                    input_schema=dict(raw.get("inputSchema") or {}),
                )
            )
        return tools

    def _request(self, skill: SkillProcess, method: str, params: dict[str, Any]) -> Any:
        """Send a JSON-RPC request and return the ``result`` field, or raise."""
        proc = skill.proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError(f"skill {skill.key} not running")
        message_id = skill._next_id
        skill._next_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": message_id,
            "method": method,
            "params": params,
        }
        line = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
        try:
            proc.stdin.write(line)
            proc.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(f"skill {skill.key} closed stdin: {exc}") from exc

        # Read response. Ignore any notification lines while waiting for our id.
        timeout_seconds = (
            skill.entry.response_timeout_seconds
            if skill.entry.response_timeout_seconds > 0
            else _RESPONSE_TIMEOUT_SECONDS
        )
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                err_tail = self._drain_stderr(skill)
                exit_code = proc.poll()
                raise RuntimeError(
                    f"skill {skill.key} timed out after {timeout_seconds}s "
                    f"waiting for {method} response. "
                    f"exit={exit_code} stderr tail: {err_tail[-400:]!r}"
                )
            r, _, _ = select.select([proc.stdout], [], [], remaining)
            if not r:
                continue  # spurious wakeup; re-check deadline
            raw = proc.stdout.readline()
            if not raw:
                err_tail = self._drain_stderr(skill)
                exit_code = proc.poll()
                raise RuntimeError(
                    f"skill {skill.key} closed stdout before responding to {method}. "
                    f"exit={exit_code} stderr tail: {err_tail[-400:]!r}"
                )
            try:
                response = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                logger.debug("skill %r emitted non-JSON on stdout: %r", skill.key, raw)
                continue
            if not isinstance(response, dict):
                continue
            if response.get("id") != message_id:
                # notification or response for a different id; ignore
                continue
            if "error" in response:
                raise RuntimeError(
                    f"skill {skill.key} {method} error: {response['error']}"
                )
            return response.get("result")

    def _notify(self, skill: SkillProcess, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        proc = skill.proc
        if proc is None or proc.stdin is None:
            return
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        line = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
        try:
            proc.stdin.write(line)
            proc.stdin.flush()
        except BrokenPipeError:
            pass

    def _drain_stderr(self, skill: SkillProcess) -> str:
        if skill.proc is None or skill.proc.stderr is None:
            return ""
        try:
            return skill.proc.stderr.read(4096).decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            logger.warning("_drain_stderr failed for skill %r: %s", skill.key, exc)
            return ""

    # ------------------------------------------------------------------ tool bridging

    def call_tool(self, skill_key: str, tool_name: str, arguments: dict[str, Any]) -> str:
        """Public-facing tool invocation used by /skills playground and tests."""
        return self._call_tool(skill_key, tool_name, arguments)

    def _call_tool(self, skill_key: str, tool_name: str, arguments: dict[str, Any]) -> str:
        """Invoke a skill's tool. Restart + retry once on crash."""
        skill = self._skills.get(skill_key)
        if skill is None:
            return f"Unknown skill '{skill_key}'."
        if not skill.entry.enabled:
            return f"Skill '{skill_key}' is disabled."

        with skill._lock:
            for attempt in (1, 2):
                if not skill.is_alive:
                    with self._mutation_lock:
                        self._spawn_locked(skill)
                    if skill.status != SkillStatus.RUNNING:
                        return f"Skill '{skill_key}' failed to start: {skill.error}"
                try:
                    result = self._request(
                        skill,
                        "tools/call",
                        {"name": tool_name, "arguments": arguments},
                    )
                    return _stringify_tool_result(result)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Skill %r tool %r failed (attempt %d): %s",
                        skill_key,
                        tool_name,
                        attempt,
                        exc,
                    )
                    skill.status = SkillStatus.CRASHED
                    skill.error = str(exc)
                    with self._mutation_lock:
                        self._cleanup_proc(skill)
                    if attempt == 2:
                        return f"Skill '{skill_key}' tool '{tool_name}' failed: {exc}"
        return f"Skill '{skill_key}' tool '{tool_name}' failed unexpectedly."

    def _bridge_tool(self, skill: SkillProcess, mcp_tool: _MCPTool) -> StructuredTool:
        """Wrap an MCP tool as a LangChain StructuredTool with user_id injected from state."""
        args_model = _build_args_model(
            f"{skill.key}_{mcp_tool.name}_Args", mcp_tool.input_schema
        )
        wants_user_id = "user_id" in (mcp_tool.input_schema.get("properties") or {})
        skill_key = skill.key
        tool_name = mcp_tool.name
        supervisor_ref = self

        def _run(
            state: Annotated[Any, InjectedState] = None,
            **kwargs: Any,
        ) -> str:
            if wants_user_id and state is not None:
                user_id = state.get("user_id") if isinstance(state, dict) else None
                if user_id and "user_id" not in kwargs:
                    kwargs["user_id"] = user_id
            return supervisor_ref._call_tool(skill_key, tool_name, kwargs)

        return StructuredTool.from_function(
            func=_run,
            name=mcp_tool.name,
            description=mcp_tool.description or f"Tool from skill '{skill.key}'.",
            args_schema=args_model,
        )


def _stringify_tool_result(result: Any) -> str:
    """Flatten an MCP ``tools/call`` result into a single string for LangGraph."""
    if result is None:
        return ""
    if isinstance(result, dict):
        if result.get("isError"):
            return f"Error: {_content_text(result.get('content'))}".strip()
        return _content_text(result.get("content")) or json.dumps(result, default=str)
    return str(result)


def _content_text(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
    return "\n\n".join(parts)


_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _build_args_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Turn a JSON Schema object into a pydantic model LangChain can use."""
    properties = schema.get("properties") if isinstance(schema, dict) else None
    required = set(schema.get("required") or []) if isinstance(schema, dict) else set()
    if not isinstance(properties, dict):
        return create_model(name)

    fields: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        if prop_name == "user_id":
            # injected from agent state — don't expose to the model
            continue
        py_type = _JSON_TO_PY.get(
            str(prop_schema.get("type")) if isinstance(prop_schema, dict) else "",
            str,
        )
        description = (
            prop_schema.get("description", "") if isinstance(prop_schema, dict) else ""
        )
        if prop_name in required:
            fields[prop_name] = (py_type, Field(..., description=description))
        else:
            default = (
                prop_schema.get("default") if isinstance(prop_schema, dict) else None
            )
            fields[prop_name] = (py_type, Field(default=default, description=description))
    return create_model(name, **fields)  # type: ignore[call-overload]


# ---------------------------------------------------------------------- public API


_singleton_lock = threading.Lock()
_singleton: SkillSupervisor | None = None


def get_supervisor() -> SkillSupervisor:
    """Return the process-wide supervisor, creating + starting it on first use.

    Auto-start is skipped when ``JUNE_SKILLS_DISABLED=1`` or
    ``JUNE_IS_SKILL_SUBPROCESS=1`` — the latter is set by the supervisor in
    every child env so a skill importing ``june_brain`` cannot trigger its own
    supervisor to spawn a fresh set of skill children.
    """
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = SkillSupervisor()
            autostart = (
                os.environ.get("JUNE_SKILLS_DISABLED") != "1"
                and os.environ.get("JUNE_IS_SKILL_SUBPROCESS") != "1"
            )
            if autostart:
                _singleton.start()
    return _singleton


def reset_supervisor_for_tests() -> None:
    """Clear the module-level supervisor. Used by tests that need isolation."""
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            _singleton.stop()
        _singleton = None


def load_skill_tools() -> list[StructuredTool]:
    """Convenience: start (if needed) and return all enabled-skill tools."""
    return get_supervisor().enabled_tools()


# Suppress ruff's unused-import warning: shlex is kept in the reserved
# import block so future env-var expansion of command/args (shell-quoted
# strings from TOML) lands in one place.
_ = shlex
