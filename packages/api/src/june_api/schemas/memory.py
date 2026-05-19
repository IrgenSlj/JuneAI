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
    journal: list[MemoryFact] = Field(
        default_factory=list,
        description="Recent journal entries written by the user or by the daily skill.",
    )
    body_metrics: list[MemoryFact] = Field(
        default_factory=list,
        description="Body metrics recorded by the health skill (one row per day).",
    )
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


class MemoryStoreCount(BaseModel):
    """Counts for one logical bucket inside a memory store."""

    kind: str = Field(..., description="Bucket name, e.g. 'goals' or 'semantic_facts'.")
    store: str = Field(..., description="Backing store: 'sqlite', 'vector', or 'graph'.")
    count: int


class MemoryStats(BaseModel):
    """GET /memory/{user_id}/stats payload.

    A compact at-a-glance view: per-bucket counts grouped by backing store, the
    total across all stores, the most recent write timestamp seen in any
    structured table, and a small sample of the most-recent semantic facts so
    the UI can show 'June recently learned…' as proof of life.
    """

    user_id: str
    total: int = 0
    buckets: list[MemoryStoreCount] = Field(default_factory=list)
    last_write: str = Field(default="", description="ISO timestamp; empty if no writes yet.")
    recent_messages: int = 0
    recent_facts: list[MemoryFact] = Field(
        default_factory=list,
        description="Up to five most-recent semantic facts, newest first.",
    )


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


class MemoryFeedbackRequest(BaseModel):
    """Body of POST /memory/{user_id}/feedback.

    ``vote`` is one of:
      - ``"up"`` — this memory was useful; rank it higher next time.
      - ``"down"`` — this memory was wrong or irrelevant; rank it lower.
      - ``"clear"`` — remove any prior vote on this ref.
    """

    ref: str = Field(..., description="Opaque memory identifier (same scheme as /memory).")
    vote: str = Field(..., description='One of "up", "down", or "clear".')


class MemoryFeedbackResponse(BaseModel):
    """Result of POST /memory/{user_id}/feedback."""

    user_id: str
    ref: str
    vote: str = Field(default="", description='"up", "down", or "" when cleared / unrecorded.')


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


class MemoryWriteRequest(BaseModel):
    """Body of POST /memory/{user_id}/fact.

    ``kind`` selects the write handler and ``fields`` are passed
    to the corresponding ``MemoryManager.write()`` handler.
    """

    kind: str = Field(..., description="Memory kind: goal, fact, entity, calendar, journal, open_loop, body_metric, mood, relation")
    fields: dict[str, Any] = Field(default_factory=dict, description="Fields for the chosen memory kind.")


class MemoryWriteResponse(BaseModel):
    """Result of POST /memory/{user_id}/fact."""

    written: bool
    kind: str
    ref: str | None = None
    stores: list[str] = Field(default_factory=list)
