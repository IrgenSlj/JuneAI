"""MemoryManager composes the three memory stores behind one interface.

Today it only delegates to the SQLite store. In Week 4 (ADR 0004) the
recall/extract loop will fan out to the vector store and knowledge graph
as well, then merge and rank the results. Keeping the public surface
here now means callers don't have to change when the other stores come
online.
"""

from __future__ import annotations

from typing import Any

from .sqlite import Memory


class MemoryManager:
    """Single entry point for all memory reads/writes for a user."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.sqlite = Memory(user_id)

    def recall(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k relevant facts for a query.

        Week 1: returns an empty list; the SQLite store handles structured
        reads directly via ``self.sqlite``. Week 4 replaces this with a
        fan-out across sqlite + vector + graph and a merged ranking.
        """
        return []

    def extract(self, exchange: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract new facts from a conversation exchange.

        Week 1: no-op. Week 4 runs an LLM-driven extractor and writes
        results to all three stores.
        """
        return []
