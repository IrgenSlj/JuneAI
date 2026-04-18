"""GET /skills — tools exposed to the agent for the current runtime."""

from __future__ import annotations

from fastapi import APIRouter

from june_brain.config import resolve_runtime_config
from june_brain.tools import JUNE_TOOLS, JUNE_TOOLS_GEMMA

from ..schemas import SkillInfo, SkillsResponse

router = APIRouter(tags=["skills"])


def _active_toolset() -> list:
    """Return the tool list the agent will actually use for the active runtime.

    Mirrors ``_select_tools_for_runtime`` in june_brain.graph: Gemma gets
    the trimmed set (small local models do better with fewer tool
    schemas), Gemini gets the full set.
    """
    runtime = resolve_runtime_config()
    if runtime.preset_key == "gemma":
        return JUNE_TOOLS_GEMMA
    return JUNE_TOOLS


@router.get("/skills", response_model=SkillsResponse)
def list_skills() -> SkillsResponse:
    """Enumerate the tools June can call right now."""
    tools = _active_toolset()
    infos = [
        SkillInfo(
            name=getattr(tool, "name", ""),
            description=getattr(tool, "description", "") or "",
            category="core",
        )
        for tool in tools
    ]
    return SkillsResponse(skills=infos, count=len(infos))
