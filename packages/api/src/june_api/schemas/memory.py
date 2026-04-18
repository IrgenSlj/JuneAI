"""Memory-inspection schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MemoryFact(BaseModel):
    """One structured row read from memory."""

    kind: str = Field(..., description="Category of fact, e.g. 'goal', 'calendar_item', 'preference'.")
    title: str = Field(default="", description="Short human-readable label.")
    body: str = Field(default="", description="Longer description or detail, when present.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured fields specific to this fact kind.",
    )


class MemorySnapshot(BaseModel):
    """Summary of what June remembers about a user.

    Week 2 returns structured highlights from the SQLite store. Week 4
    will extend this with vector-recalled facts and graph neighbors.
    """

    user_id: str
    goals: list[MemoryFact] = Field(default_factory=list)
    open_loops: list[MemoryFact] = Field(default_factory=list)
    calendar: list[MemoryFact] = Field(default_factory=list)
    recent_messages: int = Field(default=0, description="Chat messages stored for this user.")
