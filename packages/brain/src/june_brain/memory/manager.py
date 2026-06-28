"""MemoryManager composes the three memory stores behind one interface.

Per ADR 0004, every June turn runs a recall/extract loop:

1. **Recall** — fan out to all three stores with the incoming message,
   merge and rank the hits, inject them into the system prompt.
2. **Generate** — the LLM speaks with that context in hand.
3. **Extract** — a small LLM pass pulls durable facts, entities, and
   relationships from the exchange and writes them back to the
   appropriate store(s).

Consumers (agent, API, tools) only ever touch ``MemoryManager``. The
three backing stores are implementation details.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from . import extractor, writers
from .graph import KnowledgeGraph
from .recall import gather_hits, sqlite_keyword_hits
from .sqlite import Memory
from .vector import VectorStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """Single entry point for all memory reads/writes for a user."""

    def __init__(
        self,
        user_id: str,
        *,
        vector: VectorStore | None = None,
        graph: KnowledgeGraph | None = None,
        sqlite: Memory | None = None,
    ) -> None:
        self.user_id = user_id
        self.sqlite = sqlite or Memory(user_id)
        self.vector = vector if vector is not None else VectorStore(user_id)
        self.graph = graph if graph is not None else KnowledgeGraph(user_id)

    # ------------------------------------------------------------------
    # Recall — read path
    # ------------------------------------------------------------------

    def recall(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return up to ``k`` relevant memories across all three stores.

        Each hit is a dict with:
          - source: one of "vector", "graph", "sqlite"
          - text:   human-readable snippet to inject
          - kind:   sub-type for UI / filtering
          - ref:    stable identifier the caller can use to delete
          - score:  loose relevance score (lower = more relevant for
                    distance-based sources, higher = more relevant for
                    keyword hits).
        """
        return gather_hits(self.vector, self.graph, self.sqlite, self.user_id, query, k)

    # ------------------------------------------------------------------
    # Feedback pass-through (B.4)
    # ------------------------------------------------------------------

    def set_feedback(self, ref: str, vote: str) -> dict | None:
        """Record an up/down vote on a memory by ref."""
        return self.sqlite.set_feedback(ref, vote)

    def clear_feedback(self, ref: str) -> bool:
        return self.sqlite.clear_feedback(ref)

    def _sqlite_keyword_hits(self, query: str, k: int) -> list[dict[str, Any]]:
        """Scan structured tables for rows that overlap with the query.

        Retained as a method so the keyword path stays a documented recall
        fallback surface; delegates to ``recall.sqlite_keyword_hits``.
        """
        return sqlite_keyword_hits(self.sqlite, query, k)

    def format_for_prompt(self, hits: list[dict[str, Any]]) -> str:
        """Render recall hits as a compact block for the system prompt."""
        if not hits:
            return ""
        lines = ["Relevant memories (most relevant first):"]
        for h in hits:
            prefix = {
                "vector": "• (semantic)",
                "graph": "• (graph)",
                "sqlite": "• (structured)",
            }.get(h["source"], "•")
            lines.append(f"{prefix} {h['text']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Write — single entry point for all memory writes
    # ------------------------------------------------------------------

    def write(self, payload: dict[str, Any], source: str = "manual") -> dict[str, Any]:
        """Persist a memory across whichever stores apply.

        ``payload`` is ``{"kind": str, "fields": dict}``. ``kind`` selects
        the handler:

          - ``"fact"``        → vector store (text + metadata)
          - ``"entity"``      → graph node
          - ``"relation"``    → graph edge
          - ``"goal"`` / ``"open_loop"`` / ``"calendar"`` /
            ``"journal"`` / ``"body_metric"`` / ``"mood"`` →
              SQLite row + paraphrase upserted to the vector store so
              the structured row also feeds recall

        ``source`` becomes the vector-store ``source`` tag, e.g.
        ``"extraction"`` for facts pulled from chat or
        ``"skill:daily:save_journal_entry"`` for skill writes. The tag
        lets the UI explain where a recalled memory came from and lets
        future feature work filter by writer.

        Returns ``{"written": bool, "kind": str, "ref": str|None,
        "stores": list[str]}``. ``ref`` uses the same prefix scheme as
        ``/memory`` so callers can hand it to ``forget`` or render it as
        a deep link.
        """
        kind = str(payload.get("kind", "")).strip()
        fields = payload.get("fields") or {}
        if not isinstance(fields, dict):
            return {"written": False, "kind": kind, "ref": None, "stores": []}

        handler = writers.WRITE_HANDLERS.get(kind)
        if handler is None:
            return {"written": False, "kind": kind, "ref": None, "stores": []}
        try:
            return handler(self, fields, source)
        except Exception as exc:  # noqa: BLE001
            logger.exception("memory.write: %s handler failed", kind)
            return {"written": False, "kind": kind, "ref": None, "stores": [], "error": str(exc)}

    # ------------------------------------------------------------------
    # Extract — write path
    # ------------------------------------------------------------------

    def extract(
        self,
        exchange: dict[str, Any],
        llm_call: Callable[[str], str] | None = None,
    ) -> dict[str, Any]:
        """Pull durable facts/entities/relations from one exchange; write them.

        ``exchange`` expects at minimum ``{"user": str, "assistant": str}``.
        ``llm_call`` takes a prompt string and returns the raw model text —
        injected so tests can run the full extraction logic without any
        real model. Without an injected callable, extraction uses a local-only
        provider from the registry and skips gracefully if none is available.
        """
        return extractor.extract(self, exchange, llm_call)

    # ------------------------------------------------------------------
    # Delete — propagate to all stores
    # ------------------------------------------------------------------

    def forget(self, ref: str) -> bool:
        """Remove a fact by its ``ref`` (as returned from recall hits).

        Refs are prefixed with the source so we know which store to hit:
          - "semantic:<fact_id>" → vector + shadow table
          - "node:<node_id>"     → graph node (and all its edges)
          - "edge:<src>|<dst>|<kind>" → single graph edge
          - "goal:<title>"       → SQLite goals row
          - "open_loop:<topic>"  → SQLite open_loops row
          - "calendar:<title>" or "calendar:<title>|<date>|<time>"
                                 → SQLite calendar_items row
          - "journal:<id>"       → SQLite journal row
          - "body_metric:<date>" → SQLite body_metrics row
          - anything else falls through to the vector store by raw id
            (so the memory-browser UI can pass a fact_id directly).
        """
        return writers.forget(self, ref)

    # ------------------------------------------------------------------
    # Reversible forget — the trash bin for semantic facts
    # ------------------------------------------------------------------

    def list_forgotten(self, limit: int = 50) -> list[dict[str, Any]]:
        """Recently forgotten memories that can still be restored.

        Merges the per-store trash bins (semantic facts + graph entities) into
        one ref-keyed list, most recently forgotten first. Each entry carries a
        ``ref`` (the same scheme ``forget`` accepts), a short ``kind``, the
        display ``text``, and ``forgotten_at``.
        """
        entries: list[dict[str, Any]] = []
        for f in self.vector.list_forgotten(limit=limit):
            entries.append(
                {
                    "ref": f"semantic:{f['fact_id']}",
                    "kind": "fact",
                    "text": f["text"],
                    "created_at": f.get("created_at", ""),
                    "forgotten_at": f.get("forgotten_at", ""),
                }
            )
        for n in self.graph.list_forgotten_nodes(limit=limit):
            entries.append(
                {
                    "ref": f"node:{n['node_id']}",
                    "kind": n.get("kind") or "entity",
                    "text": n["label"],
                    "created_at": n.get("updated_at", ""),
                    "forgotten_at": n.get("forgotten_at", ""),
                }
            )
        entries.sort(key=lambda e: e["forgotten_at"], reverse=True)
        return entries[:limit]

    def restore(self, ref: str) -> dict[str, Any] | None:
        """Restore a forgotten memory by its ref; None if not in any trash bin."""
        ref = ref.strip()
        if ref.startswith("node:"):
            return self.graph.restore_node(ref.removeprefix("node:"))
        fact_id = ref.removeprefix("semantic:") if ref.startswith("semantic:") else ref
        return self.vector.restore(fact_id)

    def purge_forgotten(self) -> int:
        """Permanently empty every trash bin. Returns the total rows removed."""
        return self.vector.purge_forgotten() + self.graph.purge_forgotten_nodes()

    # ------------------------------------------------------------------
    # Update — patch a structured row by ref
    # ------------------------------------------------------------------

    def update(
        self,
        ref: str,
        fields: dict[str, str],
        source: str = "manual",
    ) -> dict | None:
        """Patch a structured-row memory by ``ref``.

        Returns the updated row dict (with the *new* fields), or ``None`` if
        the row was not found. Only the SQLite ref kinds are supported here:
        semantic facts and graph nodes have their own edit paths (re-upsert
        and add_node respectively).
        """
        return writers.update(self, ref, fields, source)
