"""Skill / tool listing schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SkillInfo(BaseModel):
    """A single tool exposed to the agent.

    Week 2 maps one-to-one onto the current LangChain tools. Week 5
    groups these by skill (calendar, health, research, ...) once the
    MCP decomposition lands.
    """

    name: str = Field(..., description="Tool identifier used by the agent when calling it.")
    description: str = Field(default="", description="Human-readable description.")
    category: str = Field(default="core", description="Group label; Week 5 maps this to MCP skill server.")


class SkillsResponse(BaseModel):
    skills: list[SkillInfo] = Field(default_factory=list)
    count: int = Field(default=0)
