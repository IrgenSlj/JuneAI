# ADR 0023 — Silence Model: Surface-vs-Defer Policy

## Status

Accepted; v1 (rules) implementation in progress. Anchored by the Silence/Trust
handoff (`docs/product/CLAUDE_HANDOFF_silence_and_trust.md`). Implements the
fourth inversion ("knows when to stay quiet", ADR 0015) as actual code, and is
constrained by the no-heartbeat decision (ADR 0016) and the one-loop-engine
decision (ADR 0018). Relates to the roadmap's D.2 deferred-proactivity track.

## Context

The fourth inversion says June "stays quiet, not fast": it acts when the user
speaks or the world changes, never on a timer, and it does not manufacture
reasons to interrupt. Until now that has been a principle in prose, not a
mechanism. As June gains things worth surfacing on its own initiative — a hard
deadline approaching, a contradiction found between two stored facts, a promise
that needs a nudge — the question "should I say this now, hold it, or drop it?"
becomes a real decision that needs a real, inspectable answer.

Every engagement-funded assistant answers this question with "surface now, as
often as plausible," because attention is the business model. June is funded by
trust, not attention, so June must answer it with *restraint*, and must be able to
show the user the honest reason for each choice. A policy that optimizes for
appropriate silence — and exposes why it stayed quiet — is the structural
inversion of the notification-maximizing competitor.

## Decision

A `june_brain/silence/` package provides a **local, rules-first decision function**
that governs **June-initiated surfacing only**. It is a classifier/policy function,
not a second agent and not a control loop.

```
decide(candidate: SurfacingCandidate, ctx: SurfacingContext) -> SurfacingDecision
```

The decision is `now | batch | suppress`, plus a **truthful** plain-English reason
and the **features** that produced it (for inspection and future v2 training).

1. **Scope guard — initiated surfacing only.** The Silence Model must never
   intercept, delay, or suppress June's response to a direct user message. June
   always answers when spoken to. The policy is called at the *surfacing* seam (a
   deadline fired, a contradiction was found, a promise needs a nudge), never in
   the reply path. This is asserted in code and protected by an invariant test:
   the reply path has no dependency on the silence policy.

2. **Local, zero egress.** All features are derived on-device: candidate salience
   and kind, time-to-deadline, recent dismissals of similar items, presence state
   (present-active / present-idle / absent — derived from existing
   session/activity signals, **not** a timer), whether the user is mid-task, and a
   coarse local-time bucket. No network call; works fully in local-only mode.

3. **Transparent rules (v1).** The v1 policy is human-readable rules, good on day
   one (solving cold-start) and the permanent fallback for later versions:
   - Hard deadline within the urgent window → `now`.
   - High salience + user present-idle + no recent dismissal of similar → `now`.
   - Similar item dismissed ≥2× recently → `suppress`.
   - User present-active / mid-task and non-urgent → `batch`.
   - Default for non-urgent, low-salience → `batch`.
   The reason string always names the actual deciding feature(s) — honesty core:
   never a reassuring fiction.

4. **Batch drains at event boundaries only (ADR 0016).** Batched items accumulate
   into a digest that is drained by a function the session-open / "user shows up"
   path calls — **never** a scheduled job, never a timer. There is no background
   agent and no clock. Hard deadlines remain OS notifications (roadmap D.2); the
   Silence Model decides *in-app* surfacing vs. batching, it does not replace the
   OS-notification path for true deadlines.

5. **Every decision is auditable.** Each decision is persisted to
   `surfacing_decisions` (with its features and reason) and is also written to the
   Trust Ledger (ADR 0022) as an `action` by `june` — June's restraint is itself a
   recorded, inspectable act. An `outcome` field (engaged | dismissed | expired),
   filled later, plus an explicit user `feedback` verdict, are the training signal
   for a future v2.

**Defer, not act (inversion 1).** The Silence Model produces *surfacings /
proposals*, never silent consequential actions. `now` means "show the user this
candidate"; it never means "do the thing." The user remains the resolver.

**v2 is out of scope here.** A local trained classifier over the
`surfacing_decisions` feedback log is a future, separate build; it must fall back
to these v1 rules when unavailable (graceful degradation invariant).

## Alternatives Considered

- **A background agent that scans and decides on a schedule.** Rejected outright:
  it is a heartbeat (ADR 0016) and a second control loop (ADR 0018). The policy is
  a synchronous function called at existing event seams.
- **An LLM call per surfacing decision.** Rejected for v1: adds latency, can add
  egress, and is non-inspectable. Rules are transparent and free; a model may
  inform v2 but must remain explainable and locally fall back to rules.
- **Surface everything and let the user mute.** Rejected: that is the
  engagement-maximizing default the product exists to invert. Restraint is the
  feature.
- **Silently drop low-value items.** Rejected: suppression must be *visible* in
  `/system` with its reason, or "stayed quiet" becomes indistinguishable from "had
  a bug." Honest silence is inspectable silence.

## Consequences

Positive: the fourth inversion becomes testable code with a truthful,
user-visible reason for every choice; an explicit anti-engagement engine that a
notification-driven competitor cannot adopt without contradicting its own
business model. The feedback log seeds a future local model without committing to
it now.

Negative / accepted: rules are coarse and will sometimes hold or surface
imperfectly — acceptable because every decision is inspectable and correctable via
feedback, and the rules are the floor a better v2 builds on. The policy adds a
decision seam that all initiated-surfacing producers must route through; the scope
guard and its invariant test keep that seam off the direct-reply path.
