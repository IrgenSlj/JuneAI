"""Memory inspection + editing routes.

``GET /memory/{user_id}`` returns the union of the three memory stores
(structured, semantic, graph). ``DELETE /memory/{user_id}/fact/{ref}``
propagates a delete through ``MemoryManager.forget`` so the next turn's
recall no longer surfaces it.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from june_brain.memory import KnowledgeGraph, Memory, MemoryManager, VectorStore

from ..schemas import (
    MemoryDeleteResponse,
    MemoryFact,
    MemoryFeedbackRequest,
    MemoryFeedbackResponse,
    MemorySnapshot,
    MemoryUpdateRequest,
    MemoryUpdateResponse,
    MemoryWriteRequest,
    MemoryWriteResponse,
)

router = APIRouter(tags=["memory"])


def _goal_to_fact(row: dict) -> MemoryFact:
    return MemoryFact(
        kind="goal",
        title=row.get("title", ""),
        body=row.get("next_step", ""),
        ref=f"goal:{row.get('title', '')}",
        metadata={
            "category": row.get("category", ""),
            "target_date": row.get("target_date", ""),
            "status": row.get("status", ""),
            "updated_at": row.get("updated_at", ""),
        },
    )


def _loop_to_fact(row: dict) -> MemoryFact:
    return MemoryFact(
        kind="open_loop",
        title=row.get("topic", ""),
        body=row.get("next_step", ""),
        ref=f"open_loop:{row.get('topic', '')}",
        metadata={
            "due_date": row.get("due_date", ""),
            "status": row.get("status", ""),
            "updated_at": row.get("updated_at", ""),
        },
    )


def _calendar_to_fact(row: dict) -> MemoryFact:
    title = row.get("title", "")
    date = row.get("date", "")
    time = row.get("time", "")
    return MemoryFact(
        kind="calendar_item",
        title=title,
        body=row.get("details", ""),
        ref=f"calendar:{title}|{date}|{time}",
        metadata={
            "date": date,
            "time": time,
            "status": row.get("status", ""),
            "source": row.get("source", ""),
        },
    )


def _journal_to_fact(row: dict) -> MemoryFact:
    return MemoryFact(
        kind="journal",
        title="",
        body=row.get("entry", ""),
        ref=f"journal:{row.get('id', '')}",
        metadata={
            "timestamp": row.get("timestamp", ""),
        },
    )


def _body_metric_to_fact(row: dict) -> MemoryFact:
    weight = row.get("weight_kg") or 0
    sleep = row.get("sleep_hours") or 0
    body_parts: list[str] = []
    if weight:
        body_parts.append(f"weight {weight}kg")
    if sleep:
        body_parts.append(f"sleep {sleep}h")
    metadata: dict[str, object] = {"date": row.get("date", "")}
    for key in ("sleep_quality", "energy", "stress", "soreness", "resting_hr", "steps"):
        value = row.get(key)
        if value:
            metadata[key] = value
    if row.get("notes"):
        metadata["notes"] = row["notes"]
    if row.get("timestamp"):
        metadata["timestamp"] = row["timestamp"]
    return MemoryFact(
        kind="body_metric",
        title=row.get("date", ""),
        body=", ".join(body_parts),
        ref=f"body_metric:{row.get('date', '')}",
        metadata=metadata,
    )


def _semantic_to_fact(row: dict) -> MemoryFact:
    return MemoryFact(
        kind="semantic",
        title=row.get("text", "")[:80],
        body=row.get("text", ""),
        ref=f"semantic:{row.get('fact_id', '')}",
        metadata={
            "source": row.get("source", ""),
            "created_at": row.get("created_at", ""),
            **(row.get("metadata") or {}),
        },
    )


def _entity_to_fact(node: dict) -> MemoryFact:
    props = node.get("props") or {}
    return MemoryFact(
        kind=f"entity:{node.get('kind', 'entity')}",
        title=node.get("label", ""),
        body=str(props.get("description", "")),
        ref=f"node:{node.get('node_id', '')}",
        metadata={
            "node_id": node.get("node_id", ""),
            "updated_at": node.get("updated_at", ""),
            **{k: v for k, v in props.items() if k != "description"},
        },
    )


@router.get("/memory/{user_id}", response_model=MemorySnapshot)
def get_memory(user_id: str) -> MemorySnapshot:
    """Return everything June remembers about a user, across all three stores."""
    mem = Memory(user_id)
    vector = VectorStore(user_id)
    graph = KnowledgeGraph(user_id)
    return MemorySnapshot(
        user_id=user_id,
        goals=[_goal_to_fact(g) for g in mem.get_goals(limit=20)],
        open_loops=[_loop_to_fact(loop) for loop in mem.get_open_loops(status="", limit=20)],
        calendar=[_calendar_to_fact(item) for item in mem.get_calendar_items(limit=20)],
        journal=[_journal_to_fact(j) for j in mem.get_journal(limit=20)],
        body_metrics=[_body_metric_to_fact(b) for b in mem.get_body_metrics(days=14)],
        semantic_facts=[_semantic_to_fact(f) for f in vector.list_facts(limit=30)],
        entities=[_entity_to_fact(n) for n in graph.find_nodes(limit=30)],
        recent_messages=len(mem.load_chat()),
    )


def _row_to_memory_fact(ref_kind: str, row: dict) -> MemoryFact:
    """Re-encode a freshly-updated SQLite row as a MemoryFact for the response."""
    if ref_kind == "goal":
        return _goal_to_fact(row)
    if ref_kind == "open_loop":
        return _loop_to_fact(row)
    if ref_kind == "calendar":
        return _calendar_to_fact(row)
    return MemoryFact(kind=ref_kind, title="", body="", ref="")


@router.patch("/memory/{user_id}/fact/{ref:path}", response_model=MemoryUpdateResponse)
def update_memory_fact(
    user_id: str,
    ref: str,
    payload: MemoryUpdateRequest,
) -> MemoryUpdateResponse:
    """Patch a structured fact by ref.

    Only the SQLite ref kinds are supported (``goal:``, ``open_loop:``,
    ``calendar:``). Editing semantic facts or graph nodes is not yet
    exposed here; those have their own paths.

    The returned ``ref`` may differ from the request ``ref`` when the
    primary key changed (a goal renamed, a calendar item rescheduled).
    Clients should use the returned ref for any subsequent edit or
    delete operation on the same fact.
    """
    ref_kind = ref.split(":", 1)[0] if ":" in ref else ""
    if ref_kind not in {"goal", "open_loop", "calendar"}:
        raise HTTPException(
            status_code=400,
            detail=f"PATCH not supported for ref kind '{ref_kind or '<unknown>'}'",
        )
    manager = MemoryManager(user_id)
    row = manager.update(ref, payload.fields, source="api")
    if row is None:
        raise HTTPException(status_code=404, detail=f"Fact not found: {ref}")
    fact = _row_to_memory_fact(ref_kind, row)
    return MemoryUpdateResponse(
        user_id=user_id,
        ref=fact.ref,
        updated=True,
        fact=fact,
    )


@router.post("/memory/{user_id}/feedback", response_model=MemoryFeedbackResponse)
def set_memory_feedback(
    user_id: str,
    payload: MemoryFeedbackRequest,
) -> MemoryFeedbackResponse:
    """Record a thumbs-up / thumbs-down vote on a recalled memory.

    The vote nudges the recall ranker for future turns:
      - ``up`` halves the score (lower is more relevant for distance-based hits).
      - ``down`` doubles the score.
      - ``clear`` removes the vote.

    Voting on a fact that does not exist is allowed — the feedback row stands
    alone and applies the next time a hit with this ref is recalled. This
    keeps the API simple at the cost of orphan rows; ``clear`` always wins
    if the user changes their mind.
    """
    ref = payload.ref.strip()
    vote = payload.vote.strip().lower()
    if not ref:
        raise HTTPException(status_code=400, detail="ref is required")
    if vote not in {"up", "down", "clear"}:
        raise HTTPException(
            status_code=400,
            detail="vote must be 'up', 'down', or 'clear'",
        )
    manager = MemoryManager(user_id)
    if vote == "clear":
        manager.clear_feedback(ref)
        return MemoryFeedbackResponse(user_id=user_id, ref=ref, vote="")
    manager.set_feedback(ref, vote)
    return MemoryFeedbackResponse(user_id=user_id, ref=ref, vote=vote)


@router.delete("/memory/{user_id}/fact/{ref:path}", response_model=MemoryDeleteResponse)
def delete_memory_fact(user_id: str, ref: str) -> MemoryDeleteResponse:
    """Remove a fact by its opaque ref.

    Supports refs returned from ``GET /memory``:
      - ``semantic:<fact_id>`` removes a semantic fact from vector + shadow
      - ``node:<node_id>`` removes a graph entity and its edges
      - ``edge:<src>|<dst>|<kind>`` removes a single edge
      - ``goal:<title>`` removes a structured goal
      - ``open_loop:<topic>`` removes a structured open loop
      - ``calendar:<title>|<date>|<time>`` removes a structured calendar item
      - ``journal:<id>`` removes a journal entry
      - ``body_metric:<date>`` removes a body metric row
    """
    manager = MemoryManager(user_id)
    removed = manager.forget(ref)
    return MemoryDeleteResponse(user_id=user_id, ref=ref, removed=removed)


@router.post("/memory/{user_id}/fact", response_model=MemoryWriteResponse)
def write_memory_fact(
    user_id: str,
    payload: MemoryWriteRequest,
) -> MemoryWriteResponse:
    """Write a fact to memory.

    ``kind`` selects the store:

      - ``"goal"`` / ``"open_loop"`` / ``"calendar"`` / ``"journal"`` /
        ``"body_metric"`` / ``"mood"`` → SQLite structured row + vector paraphrase
      - ``"fact"`` → vector store (semantic)
      - ``"entity"`` → knowledge graph node
      - ``"relation"`` → knowledge graph edge

    ``fields`` are passed to the corresponding write handler (see
    ``MemoryManager.write()`` for the field schema per kind).
    """
    manager = MemoryManager(user_id)
    result = manager.write({"kind": payload.kind, "fields": payload.fields}, source="api")
    return MemoryWriteResponse(**result)
