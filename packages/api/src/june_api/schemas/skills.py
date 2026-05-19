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
    enabled: bool = Field(
        default=True,
        description=(
            "Whether the agent currently binds this tool. A disabled tool stays "
            "advertised by the skill subprocess but is filtered out at "
            "agent-build time, so the model can no longer call it."
        ),
    )


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


class SkillToolToggleRequest(BaseModel):
    """POST /skills/{key}/tools/{tool}/toggle body."""

    enabled: bool = Field(..., description="New enabled state for this tool.")


class SkillToolToggleResponse(BaseModel):
    """Result of a per-tool toggle."""

    key: str
    tool: str
    enabled: bool


class SkillWriteRecord(BaseModel):
    """One paraphrased fact a skill has written into the vector store."""

    ref: str = Field(..., description='Memory ref, e.g. "semantic:abc...".')
    text: str = Field(..., description="Paraphrased fact text the recall pipeline sees.")
    source: str = Field(..., description='Full source tag, e.g. "skill:daily:track_goal".')
    tool: str = Field(default="", description="Tool name extracted from the source tag.")
    created_at: str = Field(default="", description="ISO timestamp when the write landed.")


class SkillWritesResponse(BaseModel):
    """GET /skills/{key}/writes payload."""

    skill: str
    user_id: str
    writes: list[SkillWriteRecord] = Field(default_factory=list)
    count: int = Field(default=0)


# ---------------------------------------------------------------------------
# MCP registry (Sprint 1.6) — third-party skill catalog
# ---------------------------------------------------------------------------


class RegistryInstallSpec(BaseModel):
    """How a registry entry launches as a subprocess."""

    kind: str = Field(..., description="Launcher kind: 'npx', 'uvx', 'python', or 'binary'.")
    package: str = Field(..., description="Package identifier (npm name, PyPI name, etc.).")
    command: str = Field(..., description="Executable used to launch the MCP server.")
    args: list[str] = Field(default_factory=list, description="Args passed to the command.")
    env_required: list[str] = Field(
        default_factory=list,
        description="Env vars the user must set before this skill can run.",
    )


class RegistryEntry(BaseModel):
    """One installable third-party MCP server."""

    key: str = Field(..., description="Stable identifier; also the manifest key after install.")
    name: str
    description: str = ""
    homepage: str = ""
    publisher: str = "unknown"
    verified: bool = False
    model_policy: str = Field(
        default="cloud_allowed",
        description="One of 'local_only', 'cloud_allowed', 'cloud_required'.",
    )
    install: RegistryInstallSpec
    tools_preview: list[str] = Field(default_factory=list)
    installed: bool = Field(default=False, description="True if this key already exists in skills.toml.")


class RegistryResponse(BaseModel):
    """GET /skills/registry payload."""

    schema_version: int = 1
    source: str = "first-party-curated"
    updated_at: str = ""
    entries: list[RegistryEntry] = Field(default_factory=list)
    count: int = 0


class RegistryInstallResponse(BaseModel):
    """POST /skills/registry/{key}/install payload."""

    key: str
    installed: bool
    requires_env: list[str] = Field(default_factory=list)


class RegistryUninstallResponse(BaseModel):
    """DELETE /skills/registry/{key} payload."""

    key: str
    uninstalled: bool
