# Architecture Decision Records

This directory records the architectural decisions that shape June. Each ADR captures a single decision, the context that produced it, the alternatives considered, and the consequences accepted.

## Why ADRs

Code tells you what is true today. ADRs tell you why it is true. When a future contributor (or you, six months from now) asks "why did we pick Svelte instead of React?" or "why Tauri instead of Electron?", the ADR is the answer. Without these, every old decision becomes a re-debate.

## How to Write One

Follow the template of the existing files. Keep each ADR to one decision. Keep it under two pages. Write in complete sentences.

Status progression:

- **Proposed** — drafted but not yet acted on
- **Accepted** — in effect
- **Deprecated** — no longer applied but kept for historical context
- **Superseded by ADR-XXXX** — replaced by a newer decision

## Index

| ID | Title | Status |
|---|---|---|
| [0001](0001-monorepo-structure.md) | Monorepo structure with apps/packages/skills separation | Accepted |
| [0002](0002-gemma-gemini-only.md) | Gemma 4 and Gemini as the only supported models | Accepted |
| [0003](0003-streamlit-to-sveltekit.md) | Retire Streamlit, adopt SvelteKit frontend over FastAPI | Accepted |
| [0004](0004-memory-architecture.md) | SQLite for structured memory, ChromaDB for semantic recall | Vector backend superseded by ADR 0019 |
| [0005](0005-skills-as-mcp.md) | Skills as Model Context Protocol servers | Accepted |
| [0006](0006-desktop-and-mobile-shells.md) | Tauri for desktop, Capacitor for mobile | Accepted |
| [0007](0007-sse-over-websockets.md) | SSE over WebSockets for chat streaming | Accepted |
| [0008](0008-ollama-supervision.md) | In-app Ollama supervision (use, do not bundle) | Accepted |
| [0009](0009-private-by-default-and-model-routing.md) | Private-by-default with three-tier model routing | Proposed |
| [0010](0010-agentic-core-tasks-oauth-computer-use.md) | Agentic core: tasks, OAuth skills, browser/computer use, MCP universal compatibility | Proposed; near-term sequencing superseded by ADR 0014 |
| [0011](0011-python-version-upgrade.md) | Python 3.13 baseline | Accepted |
| [0012](0012-api-key-auth.md) | Local API key auth | Accepted |
| [0013](0013-personal-assistant-framework.md) | Personal assistant framework: scheduler, notifications, daemon skills, daily orchestration | Accepted; daily orchestration superseded by ADR 0016 |
| [0014](0014-personal-operating-layer.md) | Personal operating layer: capture, events, approvals, memory provenance, scheduled work | Superseded by ADR 0015 |
| [0015](0015-center-of-gravity-four-inversions.md) | Center of gravity is the user; the four inversions | Accepted |
| [0016](0016-event-driven-no-heartbeat.md) | Event-driven proactivity; no heartbeat-as-cron | Accepted |
| [0017](0017-model-specific-provider-layer.md) | Model-specific provider layer (Gemma 4 + Gemini), roles from config | Accepted |
| [0018](0018-one-loop-engine.md) | One loop engine (hand-written); LangGraph engine removed | Accepted |
| [0019](0019-single-engine-storage-sqlite-vec.md) | Single-engine storage (sqlite-vec) + Ollama-served embeddings | Accepted |
