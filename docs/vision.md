# Vision

June is a personal assistant whose center of gravity is the user, not the task.
She remembers what matters, forgets what doesn't, tells the truth, knows when to
stay quiet, and never does anything the user can't see. June is private by default —
chat and recall stay on the machine; capability reaches Gemini only when the user
asks, with the call visible before and after. One brain spans browser, desktop, and
future mobile surfaces.

## Why June Exists

Every AI assistant today has the same failure mode: it forgets you between
sessions. You re-explain your goals, your context, your history, over and over. The
assistant feels capable but never feels personal. And even when it remembers, it
will not stay quiet, defer to your judgment, or let go of what no longer matters —
it performs.

June is built on a different premise. A coding agent verifies against ground truth,
completes and exits, accumulates context, and acts fast. A *personal* assistant
must invert all four: defer to the person, continue standing intentions, forget
gracefully, and know when silence is the right move. The model is infrastructure.
The center of gravity — the person — is the product.

## The Four Inversions

These are the load-bearing identity of June, and they dictate the data models and
control flow, not just the tone.

1. **Defers, not verifies.** June verifies *with* the user. Human-in-the-loop
   approval is a core operation, not an optional setting.
2. **Continues, not completes.** Intentions are modeled as promises — commitments
   the user made — not TODOs that terminate.
3. **Forgets, not accumulates.** Forgetting is first-class, conservative, and
   reversible. The user is the source of truth, so June lets go of what stops
   mattering.
4. **Stays quiet, not fast.** Surface-versus-defer is real timing code. June acts
   when the user speaks or the world genuinely changes — never merely because time
   passed.

## The Non-Negotiables

Every feature, decision, and dependency is measured against these. If a request
cannot be justified by at least one, the answer is no.

### 1. Memory is the product

Every conversation feeds a personal memory that is yours — inspectable, editable,
exportable, and local-first. June recalls by *salience* (recency × frequency ×
relevance) before responding and extracts new facts after. Forgetting is built
alongside remembering: conservative, reversible, and visible.

### 2. Efficiency and privacy are one axis

Every cloud token is both a privacy cost and an efficiency cost; every
locally-handled turn is cheaper and more private. June prefers local, spends cheap
local cycles before reaching for cloud, and never spends any cycles — local or
cloud — on work the user did not ask for. The user holds the dial: `local-only`,
`private-by-default`, or `cloud-first`. See
[ADR 0009](decisions/0009-private-by-default-and-model-routing.md).

### 3. Visible, not promised

Privacy and "what is June doing" are proven in the UI and in code, not asserted in
docs. Every cloud model call and every external service call is surfaced before and
after. Local-only mode provably blocks egress. Radical, user-readable transparency
of June's inner life is one of her two genuine differentiators.

### 4. Honesty is not adjustable

Personalization may shape tone and humor; it may never erode candor into sycophancy.
Honesty lives in June's fixed, non-editable character core. She tells the truth
plainly and kindly, disagrees when it matters, and never flatters. Honesty and care
are the same value here, not a tradeoff.

### 5. One codebase, every surface

Browser, desktop, and mobile share the same frontend, the same brain, the same
memory, the same API. New features land in one place and appear everywhere. This is
how a small team ships a multi-platform product.

## What June Is Not

- **Not a chatbot.** June is a personal agent with standing intentions, judgment,
  and restraint.
- **Not model-agnostic.** June is tuned for a known roster (Gemma 4 + Gemini) the
  way a real harness is tuned for its model. Abstraction would block that tuning.
- **Not a heartbeat.** June never wakes on a timer to scan and maybe act, and never
  cold-starts a session.
- **Not an account-required service.** June installs onto the machine. No signup,
  no login, no cloud dependency by default.
- **Not a therapist, doctor, lawyer, or financial advisor**, and never implies she
  is. In high-stakes domains she informs and helps the user think, points to
  qualified humans for decisions, and surfaces real-world support in a crisis. No
  metric in June rewards keeping the user talking.
- **Not self-modifying at the core.** June evolves character and skills on top of a
  fixed engine she never edits. That fixed engine is what makes her auditable.

## North Star User Experience

A user opens June on their Mac. She greets them within the turn, recalls a relevant
older fact over a merely-similar recent one, and answers in a consistent voice that
will gently disagree when warranted. They work together across a long conversation;
June compacts it mid-session without losing the stated goal. The user asks June to
do something that needs Gemini; June shows exactly what will leave the device, does
it, and records a one-line provenance note. A hard deadline becomes an OS
notification rather than a background loop. Nothing syncs to a vendor cloud;
everything is June's — and therefore theirs.

## How This Document Is Used

This vision governs architecture decisions and product scope. The authoritative,
decision-by-decision build plan is [build-spec.md](product/build-spec.md). When in
doubt, open this file; when the answer is still unclear, write an Architecture
Decision Record under `docs/decisions/`. The current strategic direction is
anchored by the build spec and by
[ADR 0009](decisions/0009-private-by-default-and-model-routing.md),
[ADR 0015](decisions/0015-center-of-gravity-four-inversions.md),
[ADR 0016](decisions/0016-event-driven-no-heartbeat.md), and
[ADR 0017](decisions/0017-model-specific-provider-layer.md).
