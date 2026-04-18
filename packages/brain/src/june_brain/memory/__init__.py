"""June's memory system.

Today this is a single SQLite store. Per ADR 0004, two more stores land
in Week 4: a vector store for semantic recall (ChromaDB) and a
knowledge graph for entity relationships. `MemoryManager` will compose
all three behind one interface.

Until then, `Memory` (the SQLite store) is the sole public class and
re-exported here for backward compatibility with `from june_brain.memory
import Memory`.
"""

from __future__ import annotations

from ..config import MEMORY_DIR  # re-exported so tests can patch it here
from .manager import MemoryManager
from .sqlite import Memory

__all__ = ["MEMORY_DIR", "Memory", "MemoryManager"]
