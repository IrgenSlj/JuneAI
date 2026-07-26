# ADR 0030 — June as an MCP memory server

## Status

Accepted. Implements Phase 4 of [`v0.3-execution-plan.md`](../product/v0.3-execution-plan.md).
Inverts the direction of ADR 0005 (June *consumes* MCP skills) — here June is the
MCP *server*. Constrained by the guard layer (ADR 0021), the Trust Ledger
(ADR 0022), and the single-facade storage rule (ADR 0019).

## Context

June's memory is her best-built subsystem: one SQLite file behind one facade,
four-signal retrieval measured at +29% recall@8 over vector search alone, and
first-class reversible forgetting. It is also, today, reachable by exactly one
population: people who own an Apple Silicon Mac, install Ollama, and download
roughly 8GB before they see a single answer.

Meanwhile the assistants people already run — Claude Desktop, Cursor, and the
large local-first agent ecosystem — all speak MCP, and all of them forget
everything between sessions. The gap between "June has good memory" and "anyone
can benefit from June's memory" is a protocol adapter, not a product rewrite.

There is a second reason, less obvious and more important. June's claim is not
"good memory" — plenty of projects claim that, and a local-first memory server
already ships from at least one vendor. June's claim is that **every access is on
the record**. An MCP server is the only surface where that claim becomes
immediately legible to a stranger: they connect a third-party agent, ask it
something, and then watch June's Receipts screen show exactly which memories that
agent read, when, and under what grant. The audit trail is the product demo.

That framing decides the hard questions below. If this were only a memory
adapter, exposing writes and skipping consent would be reasonable. Because it is
an *auditability* demonstration, the opposite is true.

## Decision

June exposes a **read-only, consent-gated, fully-ledgered** MCP server over the
existing `MemoryManager` facade.

### What is exposed

Three tools, all read paths:

| Tool | Behaviour |
|---|---|
| `search_memory` | Fused four-signal recall over `semantic_facts`, returning ranked hits with their salience and source |
| `get_memory` | Fetch a single fact by reference |
| `list_recent` | The most recently written or accessed facts, bounded |

### What is never exposed

- **No writes.** Not `remember`, not `forget`, not update. A memory server that
  any connected agent can write is a poisoning vector, and June's entire
  differentiator is that she can say where a memory came from. Writes wait for
  the provenance and quarantine columns (ADR 0025), and will arrive gated by
  them.
- **No raw store access.** The MCP layer calls the facade, never SQLite, never
  the vec0 index, never the graph directly. ADR 0019's single-facade rule holds
  here exactly as it holds everywhere else.
- **No quarantined or forgotten content.** Tombstoned facts stay tombstoned; a
  connected agent sees strictly less than the user does at `/memory`, never more.
- **No secrets, no config, no ledger mutation.** The ledger is append-only from
  every direction, including this one.

### Consent

Access defaults to **denied**. A client must be granted access explicitly, per
tool, and every grant is revocable from the Trust screen. Revocation takes effect
on the next call rather than at some later sync — there is no cached grant that
outlives the user's decision.

A grant is not a blanket key to memory. It is scoped, listed in the UI alongside
what it has actually read, and expires if unused.

### The record

Every MCP call — allowed, denied, or revoked mid-flight — writes a ledger entry.
This introduces one new ledger kind, `mcp_access`, carrying the client identity,
the tool called, the query, and how many facts were returned. It does not carry
fact contents; the ledger records *that* an access happened and its shape, not a
second copy of the memory.

This is deliberately stricter than the reply path. When June answers her own
user, the provenance frame is enough. When a third-party agent reads the user's
memory, the user gets a durable, verifiable record — because they were not in the
room when it happened.

## Alternatives considered

**Read-write from day one.** Rejected. It is the obvious way to be more useful
and the fastest way to destroy the claim that distinguishes June. An agent that
can write memories the user never said, with no provenance column to mark them,
turns June's memory into everyone else's memory: a pile of unattributable text.

**A REST API instead of MCP.** Rejected. REST would work technically and reach
nobody: the clients that matter already speak MCP, and asking them to adopt a
bespoke protocol is asking for work in exchange for nothing.

**Expose the whole `MemoryManager` surface reflectively.** Rejected. A protocol
boundary is a security boundary; generating it from the internal API means every
future facade method becomes externally callable by accident. The three tools are
enumerated by hand, and adding a fourth is a decision, not a side effect.

**Consent per session rather than per tool.** Rejected as too coarse. "This agent
may search my memory" and "this agent may list everything I recently wrote" are
different risks and deserve different answers.

**Skip the ledger for reads, on the grounds that reads are harmless.** Rejected,
and this is the crux. A read *is* the sensitive operation for a memory system —
exfiltration is the threat, not corruption. Logging only writes would mean the
one thing worth auditing is the one thing unaudited.

## Consequences

**Accepted:**

- June is useful to people who will never install the DMG, which is the point.
- A new external surface exists, and external surfaces are attack surface. It is
  loopback-only, denied by default, read-only, and ledgered — but it is new, and
  the threat model (Phase 5) must cover it explicitly.
- Read-only means the first version is genuinely less useful than competitors
  that accept writes. That is a deliberate trade of capability for
  attributability, and it is temporary — ADR 0025 unblocks writes.
- The ledger will grow faster, since a chatty agent can generate many reads per
  minute. Rate limiting and ledger compaction become real concerns sooner than
  they otherwise would.

**Rejected consequences we are explicitly not accepting:**

- No silent access. If a grant exists and a call happens, it is visible.
- No degradation of the local experience to serve the remote one. If the MCP
  server fails, June's own loop is unaffected; it is an adapter, not a dependency.
