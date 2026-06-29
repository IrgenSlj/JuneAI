"""Silence Model v1 — local, rules-first surface-vs-defer policy (ADR 0023).

June "stays quiet, not fast" (the fourth inversion). When something worth saying
arises on *June's* initiative — a deadline approaching, a contradiction found, a
promise that needs a nudge — this function decides whether to surface it `now`,
hold it for the next natural opening (`batch`), or stay silent (`suppress`), and
returns a truthful, plain-English reason naming the deciding feature.

Two hard boundaries, both load-bearing:

1. **Initiated surfacing only.** ``decide`` takes a *candidate to surface*, never
   a user message. It must never sit in the reply path — June always answers when
   spoken to. The reply path has no dependency on this module (enforced by a test
   that scans the loop package for any silence import).
2. **No model, no egress, no clock.** v1 is transparent rules over locally derived
   features. ``now`` is injected via the context (the policy never reads the wall
   clock) so decisions are deterministic and testable, and so nothing here can be
   confused for a timer (ADR 0016). A future v2 may train a local classifier over
   the feedback log, but must fall back to these rules.

The policy optimizes for *appropriate restraint*, explicitly not engagement: the
default for ordinary, non-urgent candidates is to hold them, not to push them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

SurfacingAction = Literal["now", "batch", "suppress"]

# Tunables. Kept as module constants so the rules read declaratively and a future
# v2 / config can override them without touching the branch logic.
HIGH_SALIENCE_THRESHOLD = 0.7
URGENT_WINDOW_HOURS = 3.0
DISMISSAL_SUPPRESS_THRESHOLD = 2

PRESENCE_ACTIVE = "present-active"
PRESENCE_IDLE = "present-idle"
PRESENCE_ABSENT = "absent"


@dataclass(frozen=True)
class SurfacingCandidate:
    """Something June might surface on her own initiative."""

    candidate_id: str
    kind: str  # deadline | contradiction | promise_nudge | ...
    summary: str = ""
    salience: float = 0.5  # 0..1; how much this matters
    deadline_at: str | None = None  # ISO-8601 UTC, when this has a hard deadline


@dataclass(frozen=True)
class SurfacingContext:
    """Locally derived signals at the surfacing seam. No egress, no clock read."""

    now: str  # ISO-8601 UTC, injected by the caller (the policy never reads the clock)
    presence_state: str = PRESENCE_ABSENT  # present-active | present-idle | absent
    active_thread_open: bool = False  # is the user mid-task right now
    dismissals_for_similar: int = 0  # recent dismissals of similar candidates
    local_time_bucket: str = ""  # morning|afternoon|evening|night (recorded; v2 signal)
    urgent_window_hours: float = URGENT_WINDOW_HOURS


@dataclass(frozen=True)
class SurfacingDecision:
    action: SurfacingAction
    reason: str  # plain-English, truthful — names the deciding feature(s)
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "reason": self.reason, "features": self.features}


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _deadline_delta_hours(deadline_at: str | None, now_iso: str) -> float | None:
    """Hours from ``now`` until ``deadline_at`` (negative if overdue), or None."""
    if not deadline_at:
        return None
    deadline = _parse_iso(deadline_at)
    now = _parse_iso(now_iso)
    if deadline is None or now is None:
        return None
    try:
        return (deadline - now).total_seconds() / 3600.0
    except (TypeError, ValueError, OverflowError):
        return None


def _fmt_delta(hours: float) -> str:
    if hours < 0:
        return "now overdue"
    if hours < 1:
        return f"{int(round(hours * 60))}m"
    return f"{int(round(hours))}h"


def decide(candidate: SurfacingCandidate, ctx: SurfacingContext) -> SurfacingDecision:
    """Decide surface `now` / `batch` / `suppress` for a June-initiated candidate.

    Pure rules in priority order; the returned ``reason`` always names the feature
    that decided. ``features`` carries every input so the decision is inspectable
    in /system and usable as a v2 training row.
    """
    delta = _deadline_delta_hours(candidate.deadline_at, ctx.now)
    features: dict[str, Any] = {
        "salience": candidate.salience,
        "kind": candidate.kind,
        "deadline_delta_hours": delta,
        "dismissals_for_similar": ctx.dismissals_for_similar,
        "presence_state": ctx.presence_state,
        "active_thread_open": ctx.active_thread_open,
        "local_time_bucket": ctx.local_time_bucket,
    }

    def decision(action: SurfacingAction, reason: str) -> SurfacingDecision:
        return SurfacingDecision(action=action, reason=reason, features=features)

    # 1. A hard deadline inside the urgent window surfaces now — time-critical,
    #    so it trumps even recent dismissals (the OS-notification path, D.2, also
    #    fires for true deadlines; this is the in-app decision).
    if delta is not None and delta <= ctx.urgent_window_hours:
        if delta < 0:
            return decision("now", "a deadline has passed")
        return decision("now", f"deadline in {_fmt_delta(delta)}")

    # 2. Respect the user's dismissal signal: repeatedly dismissed → stay silent.
    if ctx.dismissals_for_similar >= DISMISSAL_SUPPRESS_THRESHOLD:
        return decision(
            "suppress",
            f"you dismissed similar items {ctx.dismissals_for_similar} times recently",
        )

    # 3. High-salience and the user is around but free (and not mid-task), with a
    #    clean dismissal history → a good moment to surface. An open thread is an
    #    explicit "don't interrupt" signal that gates this even when idle.
    if (
        candidate.salience >= HIGH_SALIENCE_THRESHOLD
        and ctx.presence_state == PRESENCE_IDLE
        and not ctx.active_thread_open
        and ctx.dismissals_for_similar == 0
    ):
        return decision("now", "high salience and you're free right now")

    # 4. Mid-task: don't interrupt. Hold for the next opening.
    if ctx.presence_state == PRESENCE_ACTIVE or ctx.active_thread_open:
        return decision("batch", "you're mid-task; held for later")

    # 5. Away: hold for when the user is back.
    if ctx.presence_state == PRESENCE_ABSENT:
        return decision("batch", "you're away; held for when you're back")

    # 6. Default for ordinary, non-urgent candidates: hold, don't push.
    return decision("batch", "low urgency; held for the next opening")
