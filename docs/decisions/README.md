# Architecture Decision Records

This directory records the architectural decisions that shape June. Each ADR captures a single decision, the context that produced it, the alternatives considered, and the consequences accepted.

> For the current state of the whole project (per-subsystem summary, active plan, release status), see [`../CURRENT.md`](../CURRENT.md).

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
| [0020](0020-provider-native-tool-calling.md) | Provider-native structured tool calling, prose-JSON fallback | Accepted |
| [0021](0021-guard-layer.md) | Guard layer: untrusted-content framing, action gates, skill permissions | Accepted; partially implemented |
| [0022](0022-trust-ledger.md) | Trust Ledger: tamper-evident, hash-chained local provenance | Accepted; implementation in progress |
| [0023](0023-silence-model.md) | Silence Model: local rules-first surface-vs-defer policy | Accepted; v1 in progress |
| [0024](0024-retrieval-v2-fusion-bitemporal.md) | Retrieval v2: multi-signal fusion and bi-temporal facts | Accepted; shipped and measured (+29% recall@8) |
| [0030](0030-june-as-mcp-memory-server.md) | June as an MCP memory server (read-only, consent-gated, ledgered) | Accepted |
| [0031](0031-update-check-egress.md) | Update check: the one automatic network call, ledgered and blockable | Accepted |
| [0032](0032-model-callable-memory-surface.md) | June's model-callable memory surface is four deliberate tools | Accepted |

## Planned for v0.2

These ADRs are called for by [`JUNE_V02_BRIEF.md`](../../JUNE_V02_BRIEF.md) §9 and must be drafted *before* their workstream's implementation. Numbers are reserved in sequence; none is written yet.

| ID | Title | Workstream | Status |
|---|---|---|---|
| 0025 | Provenance-gated memory writes | W3 | Not yet drafted |
| 0026 | Night Shift: ledgered offline consolidation (must reconcile with ADR 0016) | W4 | Not yet drafted |
| 0027 | Apple FM instant tier & the "no third provider" interpretation | W5 | Not yet drafted (FOUNDER sign-off) |
| 0028 | Opt-in telemetry design | W6 | Not yet drafted |
| 0029 | Update-check network call | W1.4 | Not yet drafted |

Numbers 0025-0029 stay reserved for the workstreams above even though 0030 was
written first: the MCP server was promoted ahead of them by the 2026-07-26
re-ordering, and renumbering a reserved slot would break the references that
already point at it.
