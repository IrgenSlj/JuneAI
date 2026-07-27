# ADR 0016 — Event-Driven Proactivity; No Heartbeat

## Status

Accepted. Reverses the timer-driven proactivity introduced in ADR 0013 (Personal
Assistant Framework, the daily orchestration engine) and the scheduled background
jobs of ADR 0014. Originally anchored by the retired build specification; current
active sequencing lives in [`development-plan.md`](../product/development-plan.md).

## Context

ADR 0013 introduced a daily orchestration engine: a scheduler that wakes June on a
clock — morning briefing, mid-day check-in, evening review, weekly review — and ADR
0014 added scheduled background jobs (promise follow-up, stale open-loop review).
Exploratory work then extended this into a proactive-engagement engine that ran on a
periodic tick (a default 30-minute schedule) and chose, from clock-based triggers
(evening wind-down, Sunday-morning reflection), whether to message the user.

This is **heartbeat-as-cron**: waking every N minutes to scan and maybe act. It has
two fatal problems for June:

1. **It spends cycles on unrequested work.** Most ticks discover that nothing
   happened and burn local (and potentially cloud) compute to learn it. This
   violates the principle that efficiency and privacy are one axis — every wasted
   cycle is both a cost and an erosion of "June only acts when asked."
2. **It performs instead of responding.** A timer firing is not the user speaking
   and not the world changing. Acting on it makes June feel like a notification
   machine rather than an assistant with judgment.

## Decision

**June acts when the user speaks or the world genuinely changes — never merely
because time passed.** Concretely:

- **No heartbeat-as-cron.** June does not wake on a periodic timer to scan state and
  decide whether to act. Time-*awareness* is allowed as passive temporal context;
  time-*triggered action* is not.
- **June never cold-starts a session.** It does not initiate a conversation out of
  nowhere. Within a *live* turn June may open richly and surface a salient thread,
  but only when its salience (the recency × frequency × relevance score) crosses a
  high threshold.
- **Real-world events may wake June; the clock alone never does.** A calendar
  change, an incoming message, or a file change delivered through an event source
  (Tier 2 Mode-3 skills) is a legitimate trigger. A bare timer is not.
- **Hard deadlines become OS notifications, not loops.** When June learns a hard
  deadline June schedules an OS-level notification with a pre-written string (zero
  inference). The model wakes only if the user engages with it.
- **Sensitive context is surfaced by the user, not volunteered.** Heavy or painful
  memories are never resurfaced proactively (behavioral safety floor).

Proactivity is a later differentiator. It is built only after the continuity spine
is useful, and it is built simple-then-tuned against real use — not specified as a
perfect abstract rule.

## What This Changes

- The scheduler remains for **user-requested, deterministic** jobs (e.g. a
  reminder the user asked for, an OS notification for a known deadline). It is not
  used to drive proactive inference on a clock.
- Daily/weekly "life loops" (morning briefing, evening review) are **Tier 3** and,
  when built, run *when the user shows up*, not on a timer.
- The exploratory `proactive_tick` schedule and clock-based engagement triggers are
  not part of the direction and are not reintroduced.

## Alternatives Considered

- **Keep the daily orchestration timer (ADR 0013).** Rejected. It is the canonical
  heartbeat pattern; it spends cycles to discover nothing and trains the user to
  treat June as a notifier.
- **A low-frequency heartbeat (e.g. hourly).** Rejected. Lowering the frequency
  reduces the waste but keeps the wrong trigger model: time, not the user or the
  world.

## Consequences

Positive: June is cheaper, more private, and feels like it has judgment; "respond,
don't perform" is enforced in the architecture, not just the prompt.

Negative: genuinely useful proactive moments require an event source or a live turn,
which is more work than a timer; some "ambient" behaviors (a spontaneous good-morning
message) are deliberately out of scope.

## References

- [development-plan.md](../product/development-plan.md) — active continuity and Time sequence
- ADR 0013 — Personal Assistant Framework (daily orchestration reversed)
- ADR 0014 — Personal Operating Layer (scheduled background jobs reframed)
- ADR 0015 — Center of Gravity Is the User; The Four Inversions
