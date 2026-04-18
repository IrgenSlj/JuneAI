"""Skill registry routes — list MCP skills and toggle their enabled state.

The supervisor in ``june_brain.skills.loader`` owns the lifecycle of each
skill subprocess. These routes are a thin HTTP facade over that
supervisor plus an agent reload after any toggle so the next chat turn
actually sees the updated tool surface.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from june_brain import graph as brain_graph
from june_brain.skills.loader import list_status, set_skill_enabled

from ..schemas import (
    SkillInfo,
    SkillsResponse,
    SkillToggleRequest,
    SkillToggleResponse,
    SkillToolInfo,
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
