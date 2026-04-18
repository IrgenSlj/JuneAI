"""GET /memory/{user_id} — structured memory snapshot."""

from __future__ import annotations

from fastapi import APIRouter

from june_brain.memory import Memory

from ..schemas import MemoryFact, MemorySnapshot

router = APIRouter(tags=["memory"])


def _goal_to_fact(row: dict) -> MemoryFact:
    return MemoryFact(
        kind="goal",
        title=row.get("title", ""),
        body=row.get("next_step", ""),
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
        metadata={
            "due_date": row.get("due_date", ""),
            "status": row.get("status", ""),
            "updated_at": row.get("updated_at", ""),
        },
    )


def _calendar_to_fact(row: dict) -> MemoryFact:
    return MemoryFact(
        kind="calendar_item",
        title=row.get("title", ""),
        body=row.get("details", ""),
        metadata={
            "date": row.get("date", ""),
            "time": row.get("time", ""),
            "status": row.get("status", ""),
            "source": row.get("source", ""),
        },
    )


@router.get("/memory/{user_id}", response_model=MemorySnapshot)
def get_memory(user_id: str) -> MemorySnapshot:
    """Return a structured highlight reel of what June remembers about a user.

    Week 2: SQLite-only. Week 4 will fan out to vector + graph stores
    and merge the rankings (ADR 0004).
    """
    mem = Memory(user_id)
    return MemorySnapshot(
        user_id=user_id,
        goals=[_goal_to_fact(g) for g in mem.get_goals(limit=20)],
        open_loops=[_loop_to_fact(loop) for loop in mem.get_open_loops(status="", limit=20)],
        calendar=[_calendar_to_fact(item) for item in mem.get_calendar_items(limit=20)],
        recent_messages=len(mem.load_chat()),
    )
