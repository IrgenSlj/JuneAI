"""Memory-inspection schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MemoryFact(BaseModel):
    """One structured row read from memory.

    ``ref`` is a stable opaque string that DELETE /memory/{user}/fact/{ref}
    understands. It carries the source tag (``semantic:``, ``node:``,
    ``edge:``, ``goal:``…) so the API can route the delete to the right
    store without a second schema lookup.
    """

    kind: str = Field(..., description="Category of fact, e.g. 'goal', 'semantic', 'entity:person'.")
    title: str = Field(default="", description="Short human-readable label.")
    body: str = Field(default="", description="Longer description or detail, when present.")
    ref: str = Field(default="", description="Opaque identifier for delete/edit operations.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured fields specific to this fact kind.",
    )


class MemorySnapshot(BaseModel):
    """Summary of what June remembers about a user.

    The SQLite structured tables, the vector semantic store, and the
    knowledge graph are all represented. UI can show them grouped or
    merged in a timeline.
    """

    user_id: str
    goals: list[MemoryFact] = Field(default_factory=list)
    open_loops: list[MemoryFact] = Field(default_factory=list)
    calendar: list[MemoryFact] = Field(default_factory=list)
    semantic_facts: list[MemoryFact] = Field(
        default_factory=list,
        description="Facts extracted from conversation and embedded for semantic recall.",
    )
    entities: list[MemoryFact] = Field(
        default_factory=list,
        description="People, places, projects and concepts June has mapped for this user.",
    )
    recent_messages: int = Field(default=0, description="Chat messages stored for this user.")


class MemoryDeleteResponse(BaseModel):
    """Result of DELETE /memory/{user_id}/fact/{ref}."""

    user_id: str
    ref: str
    removed: bool


class MemoryUpdateRequest(BaseModel):
    """Body of PATCH /memory/{user_id}/fact/{ref}.

    All fields are optional strings; missing keys preserve current values.
    The set of accepted keys depends on the ref kind:
      - goal: title, category, target_date, next_step, status
      - open_loop: topic, next_step, due_date, status
      - calendar: title, date, time, details, status, source
    """

    fields: dict[str, str] = Field(
        default_factory=dict,
        description="Map of field name to new value. Unset fields keep current values.",
    )


class MemoryUpdateResponse(BaseModel):
    """Result of PATCH /memory/{user_id}/fact/{ref}.

    ``ref`` may be a *new* opaque identifier when the primary key changed
    (e.g. a goal renamed). Clients should use the returned ref for any
    subsequent edit/delete on this fact.
    """

    user_id: str
    ref: str
    updated: bool
    fact: MemoryFact | None = None
