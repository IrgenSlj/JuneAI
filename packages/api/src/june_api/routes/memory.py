"""Memory inspection + editing routes.

``GET /memory/{user_id}`` returns the union of the three memory stores
(structured, semantic, graph). ``DELETE /memory/{user_id}/fact/{ref}``
propagates a delete through ``MemoryManager.forget`` so the next turn's
recall no longer surfaces it.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from june_brain.memory import KnowledgeGraph, Memory, MemoryManager, VectorStore

from ..schemas import (
    ForgottenListResponse,
    ForgottenMemory,
    MemoryDeleteResponse,
    MemoryFact,
    MemoryFeedbackRequest,
    MemoryFeedbackResponse,
    MemoryRestoreRequest,
    MemoryRestoreResponse,
    MemorySnapshot,
    MemoryStats,
    MemoryStoreCount,
    MemoryUpdateRequest,
    MemoryUpdateResponse,
    MemoryWriteRequest,
    MemoryWriteResponse,
)

router = APIRouter(tags=["memory"])


# Defence-in-depth: refs flow into multiple stores and into the response body.
# A ref with a newline or NUL injected by a misbehaving skill would corrupt
# logs and the JSON payload; an oversized ref is never legitimate either.
_MAX_REF_LENGTH = 512
_SEARCH_SCAN_LIMIT = 10_000
_GOAL_LIMIT = 20
_OPEN_LOOP_LIMIT = 20
_CALENDAR_LIMIT = 20
_JOURNAL_LIMIT = 20
_BODY_METRIC_LIMIT = 14
_SEMANTIC_FACT_LIMIT = 30
_ENTITY_LIMIT = 30


def _validate_ref(ref: str) -> str:
    """Return the trimmed ref, or 400 if it's malformed."""
    cleaned = (ref or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="ref is required")
    if len(cleaned) > _MAX_REF_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"ref exceeds {_MAX_REF_LENGTH} characters",
        )
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in cleaned):
        raise HTTPException(
            status_code=400,
            detail="ref contains control characters",
        )
    return cleaned


def _search_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        terms: list[str] = []
        for key in sorted(value, key=str):
            terms.append(str(key))
            terms.extend(_search_terms(value[key]))
        return terms
    if isinstance(value, (list, tuple, set)):
        items = sorted(value, key=str) if isinstance(value, set) else value
        terms: list[str] = []
        for item in items:
            terms.extend(_search_terms(item))
        return terms
    return [str(value)]


def _fact_matches_query(fact: MemoryFact, query: str) -> bool:
    needle = query.casefold()
    haystack = [fact.kind, fact.title, fact.body, fact.ref]
    haystack.extend(_search_terms(fact.metadata))
    return any(needle in term.casefold() for term in haystack)


def _filter_facts(facts: list[MemoryFact], query: str) -> list[MemoryFact]:
    if not query:
        return facts
    return [fact for fact in facts if _fact_matches_query(fact, query)]


def _latest_chronological(facts: list[MemoryFact], limit: int, query: str) -> list[MemoryFact]:
    if not query:
        return facts
    return facts[-limit:]


def _first_matches(facts: list[MemoryFact], limit: int, query: str) -> list[MemoryFact]:
    if not query:
        return facts
    return facts[:limit]


def _message_matches_query(message: dict, query: str) -> bool:
    needle = query.casefold()
    return needle in str(message.get("content", "")).casefold()


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
def get_memory(
    user_id: str,
    q: str = Query(default="", description="Optional case-insensitive substring filter."),
) -> MemorySnapshot:
    """Return everything June remembers about a user, across all three stores.

    When ``q`` is supplied, the snapshot is filtered locally across visible
    fact fields and metadata. Search mode scans larger per-store limits, then
    caps each section back to the normal snapshot size.
    """
    query = q.strip()
    structured_limit = _SEARCH_SCAN_LIMIT if query else None
    mem = Memory(user_id)
    vector = VectorStore(user_id)
    graph = KnowledgeGraph(user_id)
    goals = _filter_facts(
        [_goal_to_fact(g) for g in mem.get_goals(limit=structured_limit or _GOAL_LIMIT)],
        query,
    )
    open_loops = _filter_facts(
        [
            _loop_to_fact(loop)
            for loop in mem.get_open_loops(
                status="",
                limit=structured_limit or _OPEN_LOOP_LIMIT,
            )
        ],
        query,
    )
    calendar = _filter_facts(
        [
            _calendar_to_fact(item)
            for item in mem.get_calendar_items(limit=structured_limit or _CALENDAR_LIMIT)
        ],
        query,
    )
    journal = _filter_facts(
        [_journal_to_fact(j) for j in mem.get_journal(limit=structured_limit or _JOURNAL_LIMIT)],
        query,
    )
    body_metrics = _filter_facts(
        [
            _body_metric_to_fact(b)
            for b in mem.get_body_metrics(days=structured_limit or _BODY_METRIC_LIMIT)
        ],
        query,
    )
    semantic_facts = _filter_facts(
        [
            _semantic_to_fact(f)
            for f in vector.list_facts(limit=structured_limit or _SEMANTIC_FACT_LIMIT)
        ],
        query,
    )
    entities = _filter_facts(
        [_entity_to_fact(n) for n in graph.find_nodes(limit=structured_limit or _ENTITY_LIMIT)],
        query,
    )
    messages = mem.load_chat()
    return MemorySnapshot(
        user_id=user_id,
        goals=_latest_chronological(goals, _GOAL_LIMIT, query),
        open_loops=_latest_chronological(open_loops, _OPEN_LOOP_LIMIT, query),
        calendar=_first_matches(calendar, _CALENDAR_LIMIT, query),
        journal=_latest_chronological(journal, _JOURNAL_LIMIT, query),
        body_metrics=_latest_chronological(body_metrics, _BODY_METRIC_LIMIT, query),
        semantic_facts=_first_matches(semantic_facts, _SEMANTIC_FACT_LIMIT, query),
        entities=_first_matches(entities, _ENTITY_LIMIT, query),
        recent_messages=(
            sum(1 for message in messages if _message_matches_query(message, query))
            if query
            else len(messages)
        ),
    )


@router.get("/memory/{user_id}/stats", response_model=MemoryStats)
def get_memory_stats(user_id: str) -> MemoryStats:
    """Compact per-store totals plus a 'last write' timestamp for the /memory header."""
    mem = Memory(user_id)
    vector = VectorStore(user_id)
    graph = KnowledgeGraph(user_id)

    goals = mem.get_goals(limit=10_000)
    open_loops = mem.get_open_loops(status="", limit=10_000)
    calendar = mem.get_calendar_items(limit=10_000)
    journal = mem.get_journal(limit=10_000)
    body_metrics = mem.get_body_metrics(days=10_000)
    semantic = vector.list_facts(limit=10_000)
    entities = graph.find_nodes(limit=10_000)

    buckets = [
        MemoryStoreCount(kind="goals", store="sqlite", count=len(goals)),
        MemoryStoreCount(kind="open_loops", store="sqlite", count=len(open_loops)),
        MemoryStoreCount(kind="calendar", store="sqlite", count=len(calendar)),
        MemoryStoreCount(kind="journal", store="sqlite", count=len(journal)),
        MemoryStoreCount(kind="body_metrics", store="sqlite", count=len(body_metrics)),
        MemoryStoreCount(kind="semantic_facts", store="vector", count=len(semantic)),
        MemoryStoreCount(kind="entities", store="graph", count=len(entities)),
    ]
    total = sum(b.count for b in buckets)

    last_write = ""
    for row_list in (goals, open_loops, calendar, journal, body_metrics, semantic):
        for row in row_list:
            ts = str(row.get("created_at") or row.get("updated_at") or row.get("date") or "")
            if ts > last_write:
                last_write = ts

    recent_facts = [_semantic_to_fact(f) for f in semantic[:5]]

    return MemoryStats(
        user_id=user_id,
        total=total,
        buckets=buckets,
        last_write=last_write,
        recent_messages=len(mem.load_chat()),
        recent_facts=recent_facts,
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
    ref = _validate_ref(ref)
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
    ref = _validate_ref(payload.ref)
    vote = payload.vote.strip().lower()
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
    ref = _validate_ref(ref)
    manager = MemoryManager(user_id)
    removed = manager.forget(ref)
    return MemoryDeleteResponse(user_id=user_id, ref=ref, removed=removed)


@router.get("/memory/{user_id}/forgotten", response_model=ForgottenListResponse)
def list_forgotten_memories(user_id: str, limit: int = 50) -> ForgottenListResponse:
    """Recently forgotten memories that can still be restored.

    Forgetting is conservative and reversible: a forgotten memory is archived to
    a trash bin (never recalled) rather than destroyed, so the user can undo.
    Covers semantic facts and graph entities.
    """
    limit = max(1, min(int(limit), 200))
    rows = MemoryManager(user_id).list_forgotten(limit=limit)
    memories = [
        ForgottenMemory(
            ref=str(r.get("ref", "")),
            kind=str(r.get("kind", "")),
            text=str(r.get("text", "")),
            created_at=str(r.get("created_at", "")),
            forgotten_at=str(r.get("forgotten_at", "")),
        )
        for r in rows
    ]
    return ForgottenListResponse(user_id=user_id, memories=memories, count=len(memories))


@router.post("/memory/{user_id}/forgotten/restore", response_model=MemoryRestoreResponse)
def restore_forgotten_memory(
    user_id: str, payload: MemoryRestoreRequest
) -> MemoryRestoreResponse:
    """Restore a forgotten memory to its live store with its original id."""
    ref = _validate_ref(payload.ref)
    restored = MemoryManager(user_id).restore(ref)
    if restored is None:
        raise HTTPException(status_code=404, detail="forgotten memory not found")
    return MemoryRestoreResponse(user_id=user_id, ref=ref, restored=True)


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
