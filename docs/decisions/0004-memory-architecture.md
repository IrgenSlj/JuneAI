# ADR 0004: SQLite for Structured Memory, ChromaDB for Semantic Recall

**Status:** Accepted
**Date:** 2026-04-17

## Context

Memory is the product. Every June response must be informed by what the system already knows about the user. Every June conversation must contribute new facts back into that knowledge base.

The v1 memory layer is a SQLite database with domain tables (chat, goals, calendar, body metrics, water, habits, preferences, telemetry). It is solid for structured facts that have a clear schema, but it cannot answer semantic queries such as "what did the user say last month about their relationship with their sister" or "what tone does the user prefer for morning messages."

A memory layer that cannot recall thematically is not a memory layer. It is a database.

## Decision

June's memory consists of three complementary stores, each with a specific role:

1. **Structured memory — SQLite.** Domain tables for facts with known schemas: calendar events, body metrics, workouts, journal entries, preferences, goals, habits, water logs. Fast, queryable, small footprint. The existing schema under `MEMORY_DIR/june.db` is preserved and migrated into `packages/brain/src/june_brain/memory/sqlite.py`.

2. **Semantic memory — ChromaDB.** Vector embeddings of conversation turns, journal entries, and free-form user notes. Enables "find me memories like this" retrieval. ChromaDB runs embedded (no server process), uses a local file for persistence, and supports metadata filters that cooperate with the SQLite layer.

3. **Knowledge graph — SQLite (same database, dedicated tables).** Named entities (people, projects, places) and their relationships, extracted from conversations. Stored as nodes and edges in SQLite for portability. This is the substrate that lets June say "you mentioned Ana in three conversations this week."

All three stores are accessed through a single `MemoryManager` facade at `packages/brain/src/june_brain/memory/manager.py`. Consumers (agent, tools, API) never touch the backing stores directly.

The memory loop runs on every turn:

1. **Recall:** before generation, the manager searches all three stores for context relevant to the incoming message, ranks the results, and injects the top-K into the system prompt.
2. **Generate:** the LLM produces a response with that context.
3. **Extract:** after generation, a small extractor pass pulls new facts, entities, and relationships from the exchange and writes them to the appropriate store.

Embeddings are generated locally using a small sentence-transformer model (e.g., `all-MiniLM-L6-v2` or a Gemma-family embedding model). No embedding data leaves the device.

## Consequences

**Positive:**

- Three memory shapes (structured, semantic, relational) are all supported by one facade. The agent code does not branch on memory type.
- All memory is local. The user's knowledge graph is portable as two files: `june.db` and a ChromaDB directory.
- Export is straightforward: dump SQLite, zip ChromaDB directory, done.
- The recall-extract loop is the system's actual moat. It compounds over time.

**Negative:**

- Three stores are more complex than one. Mitigated by the manager facade.
- ChromaDB adds a dependency and a few hundred megabytes of Python packages (torch, sentence-transformers). Mitigated by making the embedder swappable and keeping the default model small.
- Extraction is a second LLM call per turn. Mitigated by batching, running extraction on a background task, and keeping the extractor prompt small.

## Alternatives Considered

**Mem0.** Open-source memory framework built for exactly this use case. Strong candidate. Rejected for now because June's structured memory has domain-specific schemas (body metrics, calendar, water) that don't fit Mem0's generic key-value model. We revisit if Mem0 grows a richer structured layer, or we extract our manager into its own library.

**sqlite-vec extension.** Vector search inside SQLite. Tempting for single-store simplicity. Deferred because ChromaDB's metadata filtering, multi-collection support, and batching are more mature today. The manager facade means we can swap in sqlite-vec later without touching callers.

**LanceDB.** New, fast, columnar. Rejected because ChromaDB is more widely deployed and the performance gap does not matter at user-scale data volumes.

**Cloud vector DB (Pinecone, Weaviate hosted).** Rejected on local-first grounds. Memory cannot leave the device.

**No semantic layer, SQLite only.** Rejected because the product requires thematic recall.
