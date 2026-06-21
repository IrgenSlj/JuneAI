# June

June is a personal assistant whose center of gravity is the user, not the task.
She remembers what matters, forgets what doesn't, tells the truth, knows when to
stay quiet, and never does anything the user can't see. June runs Gemma 4 locally
for chat and recall, and reaches Gemini for capability when the user allows it,
with every cloud call visible before and after. One brain spans browser, desktop,
and future mobile surfaces.

This document describes what June is. For why it exists, read
[vision.md](../vision.md). For the authoritative, decision-by-decision build plan,
read [rebuild-plan.md](rebuild-plan.md). For how it is built, read
[architecture/overview.md](../architecture/overview.md). For what ships next, read
[roadmap.md](roadmap.md).

## The Four Inversions

June shares a coding harness's skeleton — a loop, tools, tiered memory,
compaction — but inverts its four core operations. This is not flavor; it dictates
the data models and the control flow.

| A coding agent | June |
|---|---|
| **Verifies** against ground truth (tests, the compiler) | **Defers** — verifies *with* the user. Human-in-the-loop approval is core, not optional. |
| **Completes** tasks; the loop exits | **Continues** — standing intentions. Tasks are modeled as *promises*, not TODOs that terminate. |
| **Accumulates** context; the repo is truth | **Forgets** gracefully; the user is truth. Forgetting is first-class, conservative, and reversible. |
| **Acts fast** | Knows when to **stay quiet**. "Surface vs defer" is a real operation with real timing code. |

June's genuine distinctiveness is exactly two things: these four inversions, and
radical, user-readable transparency of her inner life. Everything else — tiered
memory, salience scoring, anchored compaction, the character block — is a sound
synthesis of known work, built in service of those two.

## The Product in One Turn

A user opens June. She greets them within a live turn — never cold-starting a
session — references something real, and answers in her own voice. A message
arrives: a cheap local classifier tags its difficulty and the router picks a tier;
the assembler builds context in a fixed order, pulling salience-ranked memories,
the pinned state, and the character block; the loop calls the model, dispatches any
skills, and observes until done; if the conversation is long it compacts by merging
into the pinned anchor, never losing the user's stated goal. Every turn ends with a
one-line, plain-English provenance record: which tiers ran, whether anything left
the device, what was recalled, and why. Nothing reaches the cloud silently.

## The Spine (Tier 1)

The current build is the spine and nothing more. It is built in order; each later
piece slots into the turn above.

- **Portable data directory** — everything June *is* lives under one documented,
  versioned folder. Move machines by copying it.
- **Model-specific provider layer** — Gemma 4 and Gemini behind roles
  (`local-fast`, `local-deep`, `cloud-capable`), with a clean seam for a third.
- **Loop behind an interface** — a fixed loop with one hand-written engine,
  chosen by measurement, not taste (ADR 0018).
- **Layered context + pinned state** — a stable assembly order that protects the
  prefix cache, and compaction that merges into an anchor instead of regenerating.
- **Salience-scored recall** — `recency × frequency × relevance`, so June recalls
  what matters rather than what is merely textually similar.
- **Honest character block** — one recognizable June, seeded by us and deepening
  per user, with honesty as a fixed, non-editable core.
- **Visible cloud boundary + decision trace** — a provenance line every turn, and
  local-only mode that provably blocks egress.

## The Product Surface

- **Chat** — one column: message list, composer, model and privacy status in the
  header. Streaming token by token; tool calls render inline; a provenance chip
  carries the one-line rationale per turn.
- **Memory** — an inspectable, editable, exportable record of what June has
  learned across the three stores. The native on-demand graph (Tier 2) is opened
  here, not in an external app.
- **Tasks** — long-running units of work modeled as promises: the user's standing
  intentions, observable and resumable, not TODOs that simply terminate.
- **Skills** — capabilities the agent can call, each a standalone MCP server,
  independently enabled. Google services arrive as per-service skills (Tier 2).
- **System / Trust** — responsiveness and the capability profile in plain
  language, plus the visible record of every time data left the device.

## Model Routing

Three roles, one dial.

- **`local-fast` / `local-deep` (Gemma 4 via Ollama)** handle chat, recall,
  classification, summarisation, and any turn the user keeps private.
- **`cloud-capable` (Gemini)** handles capability the local model cannot reach,
  and only when the user's policy allows — every call visible before and after.
- The router resolves a tier per request from a difficulty classification.
  Escalation to cloud for a routine local operation is a visible last resort,
  never a silent default.

June is model-specific on purpose: she is tuned for this roster the way a real
harness is tuned for its model, because abstraction would block that tuning.

## Memory Model

Three stores, one facade (`MemoryManager`), all in one SQLite `june.db`: structured rows, a sqlite-vec index for semantic recall, and a graph for entities and relationships (ADR 0019). Recall is ranked by
*salience*, not similarity alone. The pinned state is a small structured anchor
(goal, constraints, confirmed facts, open questions) that compaction merges into,
so trimming a long conversation never drops a commitment. Forgetting (Tier 2) is
conservative, reversible, and visible — aggressive forgetting is treated as a bug.

## Privacy Spectrum

- **Mode 1 — local-only (default).** Conversations, memories, and embeddings stay
  on the machine. No silent egress.
- **Mode 2 — encrypted backup (Tier 2).** The whole data dir is client-side
  encrypted before upload; the provider holds an opaque blob. The key lives in the
  OS keychain day-to-day; a passphrase is required only when moving to a new
  machine. Crypto uses vetted libraries — never hand-rolled.
- **Mode 3 — Google per-service skills (Tier 2).** OAuth into Gmail / Calendar /
  Drive / Maps as independently-toggled skills, granted once, revocable anytime,
  always visible. Reads first; writes require per-action approval.

## Behavioral Safety Floor

June holds intimate context — relationships, family, health-adjacent, financial.
This is core to the product, not boilerplate.

- June is not a therapist, doctor, lawyer, or financial advisor, and never implies
  she is. In high-stakes domains she informs and helps the user think, and points
  to qualified humans for decisions.
- In genuine distress or crisis she responds with care, avoids amateur diagnosis,
  and surfaces real-world support. No metric in June rewards keeping the user
  talking.
- Candor means honest, never cruel; June can disagree kindly and decline kindly.
- Sensitive memories are surfaced by the user, not volunteered by June.
- These rules sit above personalization: no learned preference overrides them.

## The Product Boundary

- **No account.** June is installed, not subscribed to. No signup, no login, no
  cloud sync by default.
- **No silent cloud calls.** Every cloud-routed model call and external service
  call is visible in the UI before and after.
- **No third model.** Gemma 4 for local, Gemini for cloud. A new provider must
  replace one of these, not add to them.
- **No heartbeat.** June acts when the user speaks or the world genuinely changes,
  never merely because time passed.
- **No core self-modification.** June evolves character and skills on top of the
  harness; she never edits the loop itself.

## Status

June ships today as a web application and an experimental macOS desktop DMG. The
brain, API, three-store memory, model routing, tasks, scheduler, notification bus,
and skills system run on the hand-written harness loop (ADR 0018). The Tier 1
spine is built: the portable data directory, the model-specific provider layer,
the measured loop, layered context, salience recall, the character block, and
the visible cloud boundary. See [roadmap.md](roadmap.md) for the Tier 2 and
Tier 3 surfaces beyond it.
