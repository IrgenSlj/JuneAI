# ADR 0015 — Center of Gravity Is the User; The Four Inversions

## Status

Accepted. Supersedes the product framing of ADR 0014 (Personal Operating Layer) and
reframes ADR 0013 (Personal Assistant Framework). Originally anchored by the
retired build specification; current active sequencing lives in
[`development-plan.md`](../product/development-plan.md).

## Context

June reached its first release line as a local-first brain with memory, tasks, a
scheduler, MCP skills, a desktop shell, and a public macOS DMG. Two successive
product framings — the "personal assistant framework" (ADR 0013) and the "personal
operating layer" (ADR 0014, centered on capture → classify → approve → commit) —
pushed June toward a busy pipeline of features (Quick Capture, Daily Home, a
durable intent ledger, scheduled daily orchestration).

That framing optimized for *task throughput*: get input in, classify it, route it
to the right surface. But the thing that makes a personal assistant worth living
with is not throughput. It is that its center of gravity is the *person* — what they
care about, what they have committed to, what they have moved past — not the task in
front of it. June needs an identity that dictates the data models and control flow,
not just a feature list.

## Decision

**June is a personal assistant whose center of gravity is the user, not the task.**
She remembers what matters, forgets what doesn't, tells the truth, knows when to
stay quiet, and never does anything the user can't see.

June shares a coding harness's skeleton — a loop, tools, tiered memory, compaction —
but **inverts its four core operations**. These inversions are load-bearing: each
one changes a data model or a control-flow rule.

1. **Defers, not verifies.** A coding agent checks its work against ground truth
   (tests, the compiler). June has no such oracle for a person's life, so she
   verifies *with* the user. Human-in-the-loop approval is a core operation, not an
   optional setting. "Critical mode" means judgment, not a ground-truth check.
2. **Continues, not completes.** A coding agent's loop exits when the task is done.
   June's intentions are *promises* — commitments the user made — modeled as
   standing intentions that persist, not TODOs that terminate.
3. **Forgets, not accumulates.** A coding agent accumulates context because the
   repository is truth. For June the *user* is truth, so forgetting is a
   first-class, conservative, reversible feature, not a bug or an afterthought.
4. **Stays quiet, not fast.** A coding agent acts as fast as it can. June's
   "surface versus defer" decision is real timing code; staying quiet is often the
   correct action.

June's genuine distinctiveness is exactly two things, and effort is spent
protecting them: (1) these four inversions, and (2) radical, user-readable
transparency of June's inner life (ADR 0016 and the visible cloud boundary).
Everything else — tiered memory, salience scoring, anchored compaction, the
character block — is a sound synthesis of known work, built in service of those two.

## What This Changes

- The product center moves from a capture pipeline and Daily Home dashboard to
  **chat with memory that remembers what matters**, plus a visible trust surface.
  The Quick Capture / operating-layer pipeline (ADR 0014) is no longer the active
  direction.
- Tasks are reframed as **promises** (standing intentions), not terminating TODOs
  (see the Tier 2 task ledger).
- **Forgetting** becomes a designed feature with its own conservative, reversible
  budget — not merely a delete button.
- Approval is **core**, present in the loop, not a feature bolted onto external
  writes.

## Build Discipline

The build proceeds in tiers. **Tier 1 (the spine)** delivers
the one-sentence vision and nothing more; **Tier 2** is built only after Tier 1
ships and has been used; **Tier 3** is north-star design intent. The spec is
deliberately complete to be a reference, not a backlog — building all of it at once
is the explicit trap to avoid.

## Alternatives Considered

- **Keep the operating-layer framing (ADR 0014).** Rejected. It optimizes for task
  flow and a dashboard surface; it does not name the identity that makes June worth
  using, and it tempts the team into a wide feature pipeline before the spine
  exists.
- **A pure chat-with-memory product.** Rejected. Without the inversions (defer,
  continue, forget, stay quiet) June is just another assistant that forgets you and
  performs.

## Consequences

Positive: a single sentence and four inversions decide ambiguous design calls;
scope is disciplined by tiers; the product has a defensible identity rather than a
feature list.

Negative: it deprecates recent framing and some exploratory work built against it
(capture pipeline, scheduler-driven proactivity); contributors must internalize the
inversions before the data models make sense.

## References

- [development-plan.md](../product/development-plan.md) — active implementation sequence
- ADR 0013 — Personal Assistant Framework (reframed)
- ADR 0014 — Personal Operating Layer (superseded framing)
- ADR 0016 — Event-Driven Proactivity; No Heartbeat
- ADR 0017 — Model-Specific Provider Layer
