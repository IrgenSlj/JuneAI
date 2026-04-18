"""Knowledge graph for entity relationships.

Scheduled for Week 4 (see ADR 0004). Nodes are people, places, projects,
and recurring concepts; edges are typed relationships. Storage starts in
SQLite (nodes + edges tables) and can graduate to a dedicated graph DB
later if scale demands it. This stub keeps the interface shape so the
manager can reference it.
"""

from __future__ import annotations

from typing import Any


class KnowledgeGraph:
    """Stub graph store. All operations are no-ops until Week 4."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id

    def add_node(self, node_id: str, kind: str, props: dict[str, Any] | None = None) -> None:
        raise NotImplementedError("KnowledgeGraph lands in Week 4 — see docs/decisions/0004-memory-architecture.md")

    def add_edge(self, src: str, dst: str, kind: str, props: dict[str, Any] | None = None) -> None:
        raise NotImplementedError("KnowledgeGraph lands in Week 4 — see docs/decisions/0004-memory-architecture.md")

    def neighbors(self, node_id: str, kind: str | None = None) -> list[dict[str, Any]]:
        return []
