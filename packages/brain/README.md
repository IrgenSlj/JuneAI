# june-brain

June's intelligence layer. A Python package built around a hand-written harness
loop (one engine, no agent framework), a three-store memory system, the Promises
runtime, an MCP skills loader, and the Gemma/Gemini model providers.

## Install

```bash
pip install june-brain
```

## Use

```python
from june_brain import create_june_agent
from june_brain.memory import MemoryManager

agent = create_june_agent()

# Recall is automatic inside the agent. To inspect or mutate memory
# directly, go through MemoryManager (per-user).
memory = MemoryManager(user_id="me")
hits = memory.recall("what did I say about ramen", k=5)
```

The three stores (`Memory` for SQLite rows, `VectorStore` for the sqlite-vec index, `KnowledgeGraph` for entities), all in one `june.db`, live behind `MemoryManager`. Direct access is available for scripts and migrations — see `packages/brain/src/june_brain/memory/`.

Promises live in `june_brain.tasks`. They are durable, observable units of
standing work: a goal, status, trace steps, blocked reason, next action, final
deliverable, and recurrence metadata. A tool blocked by local-only mode leaves
the promise `awaiting_user` instead of completing optimistically.

## See Also

- [Architecture overview](../../docs/architecture/overview.md)
- [ADR 0004: memory architecture](../../docs/decisions/0004-memory-architecture.md)
- [ADR 0005: skills as MCP](../../docs/decisions/0005-skills-as-mcp.md)
