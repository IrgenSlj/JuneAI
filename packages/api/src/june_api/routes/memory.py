"""Memory inspection + editing routes.

``GET /memory/{user_id}`` returns the union of the three memory stores
(structured, semantic, graph). ``DELETE /memory/{user_id}/fact/{ref}``
propagates a delete through ``MemoryManager.forget`` so the next turn's
recall no longer surfaces it.
"""

from __future__ import annotations

from fastapi import APIRouter

from june_brain.memory import KnowledgeGraph, Memory, MemoryManager, VectorStore

from ..schemas import MemoryDeleteResponse, MemoryFact, MemorySnapshot

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
        semantic_facts=[_semantic_to_fact(f) for f in vector.list_facts(limit=30)],
        entities=[_entity_to_fact(n) for n in graph.find_nodes(limit=30)],
        recent_messages=len(mem.load_chat()),
    )


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
    """
    manager = MemoryManager(user_id)
    removed = manager.forget(ref)
    return MemoryDeleteResponse(user_id=user_id, ref=ref, removed=removed)
