# June AI — Roadmap

## Current Direction

June is a **personal assistant whose center of gravity is the user, not the task**:

> She remembers what matters, forgets what doesn't, tells the truth, knows when to
> stay quiet, and never does anything the user can't see.

This direction is defined in full by the canonical
[build specification](docs/product/build-spec.md). The interface stays calm and
simple; the system underneath is technically rigorous, local-first, and visibly
private.

## What Makes June June — Four Inversions

June borrows a coding agent's skeleton (loop, tools, tiered memory, compaction)
and inverts its four core operations:

| Coding agent | June |
|---|---|
| Verifies against ground truth | **Defers** — verifies *with* the user (human-in-the-loop is core) |
| Completes tasks (loop exits) | **Continues** — standing intentions modeled as promises |
| Accumulates context (repo is truth) | **Forgets** gracefully (the user is truth) |
| Acts fast | Knows when to **stay quiet** (surface vs defer is real code) |

## Governing Principles

1. **Efficiency and privacy are one axis.** Every cloud token is both a privacy
   and an efficiency cost. Prefer local; never spend cycles on unrequested work.
2. **The user never leaves June.** If a feature's value lives in another app,
   embed it natively or don't ship it.
3. **No dependency we can avoid.** Stdlib or a small custom implementation beats a
   new package. The one exception is cryptography — always use a vetted library.
4. **Visible, not promised.** Privacy and "what is June doing" are proven in the
   UI and in code, not asserted in docs.
5. **Respond to the user; don't perform.** June acts when the user speaks or the
   world genuinely changes — never merely because time passed.
6. **Model-specific, not model-agnostic.** June is tuned for a known roster
   (Gemma 4 + Gemini) so the harness can be tuned the way real harnesses are.

## Shipped (foundation we build on)

- Web PWA with chat, memory, settings, skills, tasks, and system activity.
- Tauri desktop shell with Ollama supervision, tray, hotkey, autostart, and
  native notification capability; v0.1.0 Apple Silicon macOS DMG.
- SQLite + Chroma + graph memory behind one `MemoryManager`.
- MCP skill supervisor and bundled skills.
- LangGraph agent with SSE streaming and per-message model provenance.

## Tier 1 — The Spine (build now, in order)

This is the active track. It delivers the one-sentence vision and nothing more.
Do not start Tier 2 until Tier 1 ships and has been *used*.

- **C.0 Portable data directory + versioned manifest.** One documented folder that
  is June; copy it to move machines.
- **C.1 Model-specific provider layer.** Gemma 4 + Gemini behind roles
  (`local-fast`, `local-deep`, `cloud-capable`); a clean seam for a third.
- **C.2 Loop behind an interface, chosen by measurement.** Hand-written loop vs
  LangGraph, scored on CLEAR (Cost, Latency, Efficacy, Assurance, Reliability).
- **C.3 Layered context + pinned state.** Fixed assembly order; compaction that
  merges into an anchor, never regenerates, so commitments survive trimming.
- **C.4 Salience-scored recall.** `recency × frequency × relevance` — recall what
  matters, not just what is textually similar.
- **C.5 Honest character as a self-authored block.** One recognizable June,
  deepening per user, with honesty as a fixed, non-editable core.
- **C.6 Visible cloud boundary + decision trace.** A provenance line every turn;
  local-only mode provably blocks egress; difficulty classifier feeds the router.

**Tier 1 is done when:** June runs on local Gemma 4, recalls a relevant older fact
over a merely-similar recent one, compacts a long conversation without losing the
stated goal, answers in a consistent voice that will gently disagree, and never
reaches the cloud without a visible provenance line.

## Tier 2 — Differentiators (after Tier 1 ships and is used)

Build simple, observe, refine — these need a working June to tune against.

- **D.1 Temporal context layer** — passive time-awareness, no timer.
- **D.2 Event-driven, deferred proactivity** — June never cold-starts a session;
  hard deadlines become OS notifications; the clock alone never wakes her.
- **D.3 Self-monitor + idle housekeeping + reference-context diffing** — hygiene
  yes, idle inference no.
- **D.4 Conservative, reversible forgetting** — biased hard toward retention.
- **D.5 Durable task ledger built around continuity** — tasks as promises.
- **D.6 Native memory graph** — custom canvas, ~40-line force sim, no library.
- **D.7 System page** — responsiveness + capability profile, calm not pulsing.
- **D.8 Privacy Mode 2** — client-side-encrypted backup (keychain + passphrase).
- **D.9 Privacy Mode 3** — Google as per-service skills, grant-once, revocable.

## Tier 3 — North Star (design intent; not built yet)

Full live brain map; self-improvement Rungs 2–3 (capability-blocked, not just
safety-gated); daily/weekly life loops (run when the user shows up, never on a
timer); page IA rename (Memory / Tasks / Trust); CLEAR as standing practice.
**Rung 4 (core self-modification) is permanently excluded** — the fixed engine is
what makes June auditable.

## Explicit Non-Goals

- No heartbeat-as-cron (waking on a timer to scan and maybe act).
- No external app (Obsidian) as the place to view June's memory — native graph.
- No graph-visualization library.
- No hand-rolled cryptography.
- No cloud account requirement, cloud memory service, or third model provider.
- No paid hosting dependency. No always-on audio.
- No core self-modification (Rung 4).
