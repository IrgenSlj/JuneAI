"""June's memory system — three complementary stores behind one facade.

Per ADR 0004 (vector backend updated by ADR 0019 — one SQLite engine):
- ``Memory`` — SQLite, structured domain tables (goals, journal, body, …)
- ``VectorStore`` — sqlite-vec (vec0) index in the same june.db, semantic
  recall via Ollama-served embeddings
- ``KnowledgeGraph`` — SQLite graph tables (nodes + edges), entity/relation memory
- ``MemoryManager`` — the facade that recalls across all three and
  extracts new memory from each exchange

Callers should prefer ``MemoryManager``. ``Memory`` is still re-exported
because a lot of legacy brain code speaks directly to SQLite tables.
"""

from __future__ import annotations

from ..config import MEMORY_DIR  # re-exported so tests can patch the data root here
from .graph import KnowledgeGraph
from .manager import MemoryManager
from .sqlite import Memory, db_path
from .vector import VectorStore

__all__ = [
    "MEMORY_DIR",
    "Memory",
    "db_path",
    "MemoryManager",
    "VectorStore",
    "KnowledgeGraph",
]
