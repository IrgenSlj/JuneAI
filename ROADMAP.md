# June AI — Roadmap

June is a **personal assistant whose center of gravity is the user, not the task**:
she remembers what matters, forgets what doesn't, tells the truth, knows when to
stay quiet, and never does anything the user can't see.

The full direction, invariants, and rationale live in the canonical
[build specification](docs/product/build-spec.md) and
[ADRs 0015–0017](docs/decisions/). June is built in tiers: **Tier 1** is the spine
(the one-sentence vision, nothing more); **Tier 2** adds differentiators only once
Tier 1 is shipped and *used*; **Tier 3** is the north star.

## Four inversions (what makes June June)

| A coding agent | June |
|---|---|
| Verifies against ground truth | **Defers** — verifies *with* the user |
| Completes tasks, then exits | **Continues** — standing intentions, modeled as promises |
| Accumulates context | **Forgets** gracefully — the user is the source of truth |
| Acts fast | Knows when to **stay quiet** |

## Status — Tier 1 spine is built

The seven spine modules are implemented, tested, and on `main`:

- **C.0** Portable data directory + versioned manifest.
- **C.1** Model-specific provider layer — Gemma 4 + Gemini behind roles.
- **C.2** Harness loop behind an interface + the CLEAR experiment harness.
- **C.3** Layered context + pinned state + anchored compaction.
- **C.4** Salience-scored recall (`recency × frequency × relevance`) — *live in the recall path.*
- **C.5** Honest character block — immutable candor + behavioral safety floor.
- **C.6** Difficulty classifier, capability probe, and the visible cloud boundary
  (per-turn provenance + plain-English rationale) — *provenance live in chat.*

## Now — Tier 1 is shipped; use it, then tune

The spine (C.0–C.6) is built and the **handwritten loop is the live chat path** —
provider layer, layered context + anchored compaction, character block, salience
recall, difficulty router, and capability probe all flow through it; LangGraph
stays as a flagged fallback (`JUNE_CHAT_USE_HARNESS=0`). The CLEAR baseline is
measured for both engines (`docs/experiments/loop-clear.md`): the handwritten loop
is 3-17x faster at equal efficacy, so `handwritten` is the default.

What "done" looks like — and now holds: June runs on local Gemma 4, recalls a
relevant older fact over a merely-similar recent one, compacts a long conversation
without losing the stated goal, answers in a consistent voice that will gently
disagree, and never reaches the cloud without a visible provenance line.

The discipline for the next session: **use June against real conversations before
opening Tier 2.** Tune salience weights, compaction triggers, and the difficulty
classifier from what you observe — then pick the first Tier 2 differentiator below.

## Next — Tier 2 differentiators

Build simple, observe, refine — these need a working June to tune against.

- **D.1** Temporal context layer — passive time-awareness, no timer.
- **D.2** Event-driven, deferred proactivity — never cold-starts a session; hard
  deadlines become OS notifications; the clock alone never wakes her.
- **D.3** Self-monitor + idle housekeeping + reference-context diffing.
- **D.4** Conservative, reversible forgetting — biased hard toward retention.
- **D.5** Durable task ledger built around continuity — tasks as promises.
- **D.6** Native memory graph — custom canvas, ~40-line force sim, no library.
- **D.7** System page — responsiveness + capability profile, calm not pulsing.
- **D.8** Privacy Mode 2 — client-side-encrypted backup (keychain + passphrase).
- **D.9** Privacy Mode 3 — Google as per-service skills, grant-once, revocable.

## Later — Tier 3 north star (design intent)

Full live brain map; self-improvement Rungs 2–3 (capability-blocked, not just
safety-gated); daily/weekly life loops (run when the user shows up, never on a
timer); page IA rename (Memory / Tasks / Trust). **Rung 4 (core self-modification)
is permanently excluded** — the fixed engine is what makes June auditable.

## Non-goals

- No heartbeat-as-cron (waking on a timer to scan and maybe act).
- No external app (e.g. Obsidian) as the place to view June's memory.
- No graph-visualization library. No hand-rolled cryptography.
- No account requirement, cloud memory service, or third model provider.
- No paid hosting dependency. No always-on audio. No core self-modification.
