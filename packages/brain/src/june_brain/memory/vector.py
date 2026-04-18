"""Vector store for semantic recall.

Scheduled for Week 4 (see ADR 0004). The plan is to use ChromaDB in
embedded mode with `all-MiniLM-L6-v2` as the local embedder. This stub
exists so `MemoryManager` and the rest of the brain can start wiring in
the interface without waiting for the implementation.
"""

from __future__ import annotations

from typing import Any


class VectorStore:
    """Stub vector store. All operations are no-ops until Week 4."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id

    def upsert(self, fact_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        raise NotImplementedError("VectorStore lands in Week 4 — see docs/decisions/0004-memory-architecture.md")

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        return []

    def delete(self, fact_id: str) -> None:
        return None
