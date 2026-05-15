"""Obsidian vault export schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ObsidianFile(BaseModel):
    """One file to write into an Obsidian vault."""

    path: str = Field(..., description="Vault-relative POSIX path.")
    content: str = Field(..., description="UTF-8 file contents.")


class ObsidianExportResponse(BaseModel):
    """A vault-shaped export of June memory, skills, and architecture."""

    user_id: str
    files: list[ObsidianFile] = Field(default_factory=list)
    count: int = Field(default=0, description="Number of files in the export.")
