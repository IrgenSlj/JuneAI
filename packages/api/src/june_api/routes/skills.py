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
    list_status,
    set_skill_enabled,
    set_skill_tool_enabled,
)

from ..schemas import (
    SkillInfo,
    SkillsResponse,
    SkillToggleRequest,
    SkillToggleResponse,
    SkillToolInfo,
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
