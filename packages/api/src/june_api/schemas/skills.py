"""Skill listing + toggle schemas.

Week 5 reshaped this endpoint to describe MCP skill servers (calendar,
health, research, files, daily) rather than individual tools. Each skill
carries its current lifecycle status and the tools it exposes so the
/skills page can render both the toggle and the impact of toggling.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SkillToolInfo(BaseModel):
    """A single tool inside a skill — shown under the skill in the UI."""

    name: str = Field(..., description="Tool identifier the agent uses to call it.")
    description: str = Field(default="", description="Human-readable description.")


class SkillInfo(BaseModel):
    """One MCP skill server with its current lifecycle state."""

    key: str = Field(..., description="Stable skill id, e.g. 'calendar' or 'research'.")
    description: str = Field(default="", description="What the skill does.")
    enabled: bool = Field(default=True, description="Manifest toggle state.")
    status: str = Field(
        default="stopped",
        description="Lifecycle state: stopped, starting, running, crashed, or disabled.",
    )
    error: str = Field(
        default="",
        description="Last error message when status is 'crashed', empty otherwise.",
    )
    tools: list[SkillToolInfo] = Field(
        default_factory=list,
        description="Tools this skill contributes to the agent when running.",
    )


class SkillsResponse(BaseModel):
    """GET /skills payload."""

    skills: list[SkillInfo] = Field(default_factory=list)
    count: int = Field(default=0)


class SkillToggleRequest(BaseModel):
    """POST /skills/{key}/toggle body."""

    enabled: bool = Field(..., description="New enabled state.")


class SkillToggleResponse(BaseModel):
    """Result of a toggle — callers refresh their UI from this."""

    key: str
    enabled: bool
    status: str
    error: str = ""
