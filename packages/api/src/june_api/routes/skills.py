"""Skill registry routes — list MCP skills and toggle their enabled state.

The supervisor in ``june_brain.skills.loader`` owns the lifecycle of each
skill subprocess. These routes are a thin HTTP facade over that
supervisor plus an agent reload after any toggle so the next chat turn
actually sees the updated tool surface.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from june_brain import graph as brain_graph
from june_brain.memory import VectorStore
from june_brain.skills.loader import (
    call_skill_tool,
    list_status,
    set_skill_enabled,
    set_skill_tool_enabled,
)
from june_brain.skills.registry import (
    install_from_registry,
    installed_keys,
    load_registry,
    uninstall_from_manifest,
)

from ..schemas import (
    RegistryEntry,
    RegistryInstallResponse,
    RegistryInstallSpec,
    RegistryResponse,
    RegistryUninstallResponse,
    SkillInfo,
    SkillsResponse,
    SkillToggleRequest,
    SkillToggleResponse,
    SkillToolInfo,
    SkillToolInvokeRequest,
    SkillToolInvokeResponse,
    SkillToolToggleRequest,
    SkillToolToggleResponse,
    SkillWriteRecord,
    SkillWritesResponse,
)

router = APIRouter(tags=["skills"])


def _status_to_info(payload: dict) -> SkillInfo:
    return SkillInfo(
        key=str(payload.get("key", "")),
        description=str(payload.get("description", "")),
        enabled=bool(payload.get("enabled", False)),
        status=str(payload.get("status", "stopped")),
        error=str(payload.get("error", "")),
        tools=[
            SkillToolInfo(
                name=str(tool.get("name", "")),
                description=str(tool.get("description", "")),
                enabled=bool(tool.get("enabled", True)),
                input_schema=dict(tool.get("input_schema") or {}),
            )
            for tool in payload.get("tools", []) or []
        ],
    )


@router.get("/skills", response_model=SkillsResponse)
def list_skills() -> SkillsResponse:
    """List every MCP skill with its current enabled state and tools."""
    snapshots = list_status()
    infos = [_status_to_info(snap) for snap in snapshots]
    return SkillsResponse(skills=infos, count=len(infos))


@router.get(
    "/skills/{key}/writes/{user_id}",
    response_model=SkillWritesResponse,
)
def list_skill_writes(key: str, user_id: str, limit: int = 30) -> SkillWritesResponse:
    """List paraphrased facts this skill has written for a user.

    Filters the vector store's shadow table by ``source LIKE 'skill:<key>:%'``,
    so it picks up writes from every tool the skill exposes. Each record
    carries the prefixed memory ref so the UI can deep-link into ``/memory``
    or hand the ref to ``forget``.
    """
    if not any(snap.get("key") == key for snap in list_status()):
        raise HTTPException(status_code=404, detail=f"Unknown skill: {key!r}")
    facts = VectorStore(user_id).list_facts(
        limit=max(1, min(limit, 200)),
        source_prefix=f"skill:{key}:",
    )
    records = [
        SkillWriteRecord(
            ref=f"semantic:{f['fact_id']}",
            text=str(f.get("text", "")),
            source=str(f.get("source", "")),
            tool=str(f.get("source", "")).split(":", 2)[2]
            if str(f.get("source", "")).count(":") >= 2
            else "",
            created_at=str(f.get("created_at", "")),
        )
        for f in facts
    ]
    return SkillWritesResponse(
        skill=key,
        user_id=user_id,
        writes=records,
        count=len(records),
    )


@router.post(
    "/skills/{key}/tools/{tool}/toggle",
    response_model=SkillToolToggleResponse,
)
def toggle_skill_tool(
    key: str, tool: str, body: SkillToolToggleRequest
) -> SkillToolToggleResponse:
    """Enable or disable a single tool inside a skill.

    The skill subprocess keeps running and continues to advertise this
    tool, but the supervisor filters it out of the agent's bound tools
    at the next reload. Persists to ``skills.toml`` so the gate
    survives restart.
    """
    skill = set_skill_tool_enabled(key, tool, body.enabled)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Unknown skill: {key!r}")

    # Rebuild the agent so its bound tool list reflects the new state.
    brain_graph.reload_agent()

    return SkillToolToggleResponse(
        key=skill.entry.key,
        tool=tool,
        enabled=tool not in (skill.entry.disabled_tools or []),
    )


@router.post("/skills/{key}/toggle", response_model=SkillToggleResponse)
def toggle_skill(key: str, body: SkillToggleRequest) -> SkillToggleResponse:
    """Enable or disable a skill.

    Persists to ``~/Library/Application Support/June/skills.toml`` and
    rebuilds the agent so the next chat turn picks up the new tool set.
    """
    skill = set_skill_enabled(key, body.enabled)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Unknown skill: {key!r}")

    # Rebuild the agent so its bound tool list reflects the new state.
    # Done after the toggle so a failed reload doesn't block the flip.
    brain_graph.reload_agent()

    return SkillToggleResponse(
        key=skill.entry.key,
        enabled=skill.entry.enabled,
        status=skill.status.value,
        error=skill.error,
    )


# ---------------------------------------------------------------------------
# MCP registry (Sprint 1.6)
# ---------------------------------------------------------------------------


def _registry_to_response() -> RegistryResponse:
    registry = load_registry()
    installed = installed_keys()
    entries = [
        RegistryEntry(
            key=e.key,
            name=e.name,
            description=e.description,
            homepage=e.homepage,
            publisher=e.publisher,
            verified=e.verified,
            model_policy=e.model_policy,
            install=RegistryInstallSpec(**e.install.to_dict()),
            tools_preview=list(e.tools_preview),
            installed=e.key in installed,
        )
        for e in registry.entries
    ]
    return RegistryResponse(
        schema_version=registry.schema_version,
        source=registry.source,
        updated_at=registry.updated_at,
        entries=entries,
        count=len(entries),
    )


@router.get("/skills/registry", response_model=RegistryResponse)
def list_registry() -> RegistryResponse:
    """List installable third-party MCP servers from the curated registry."""
    return _registry_to_response()


@router.post("/skills/registry/{key}/install", response_model=RegistryInstallResponse)
def install_registry_entry(key: str) -> RegistryInstallResponse:
    """Materialize a registry entry into the user's skill manifest."""
    try:
        entry = install_from_registry(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    brain_graph.reload_agent()
    registry = load_registry()
    reg_entry = registry.find(key)
    return RegistryInstallResponse(
        key=entry.key,
        installed=True,
        requires_env=list(reg_entry.install.env_required) if reg_entry else [],
    )


@router.delete("/skills/registry/{key}", response_model=RegistryUninstallResponse)
def uninstall_registry_entry(key: str) -> RegistryUninstallResponse:
    """Remove an installed registry skill from the manifest."""
    removed = uninstall_from_manifest(key)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Skill '{key}' is not installed.")
    brain_graph.reload_agent()
    return RegistryUninstallResponse(key=key, uninstalled=True)


# ---------------------------------------------------------------------------
# Skill playground (Batch 1)
# ---------------------------------------------------------------------------


@router.post(
    "/skills/{key}/tools/{tool}/invoke",
    response_model=SkillToolInvokeResponse,
)
def invoke_skill_tool(
    key: str, tool: str, body: SkillToolInvokeRequest
) -> SkillToolInvokeResponse:
    """Invoke a skill's tool with arbitrary args. Used by the /skills playground.

    Behaves like the agent's bridged call: returns the stringified MCP result,
    and surfaces subprocess crashes as readable error text rather than raising.
    """
    import time

    if not any(snap.get("key") == key for snap in list_status()):
        raise HTTPException(status_code=404, detail=f"Unknown skill: {key!r}")

    started = time.monotonic()
    try:
        result_text = call_skill_tool(key, tool, body.arguments or {})
        latency_ms = int((time.monotonic() - started) * 1000)
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.monotonic() - started) * 1000)
        return SkillToolInvokeResponse(
            skill=key, tool=tool, ok=False, error=str(exc), latency_ms=latency_ms,
        )

    text = result_text or ""
    # The supervisor returns "Skill 'x' tool 'y' failed: ..." on crash; surface
    # that as ok=False so the UI can highlight it appropriately.
    is_failure = text.startswith(("Skill ", "Unknown skill"))
    return SkillToolInvokeResponse(
        skill=key,
        tool=tool,
        ok=not is_failure,
        result="" if is_failure else text,
        error=text if is_failure else "",
        latency_ms=latency_ms,
    )
