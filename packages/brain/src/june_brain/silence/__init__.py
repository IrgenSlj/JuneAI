"""Silence Model — local, rules-first surface-vs-defer policy (ADR 0023).

Governs June-initiated surfacing only; never the direct-reply path.
"""

from __future__ import annotations

from .policy import (
    DISMISSAL_SUPPRESS_THRESHOLD,
    HIGH_SALIENCE_THRESHOLD,
    PRESENCE_ABSENT,
    PRESENCE_ACTIVE,
    PRESENCE_IDLE,
    URGENT_WINDOW_HOURS,
    SurfacingAction,
    SurfacingCandidate,
    SurfacingContext,
    SurfacingDecision,
    decide,
)
from .presence import derive_presence
from .producers import build_deadline_candidates, run_silence_producers
from .store import (
    VALID_OUTCOMES,
    SurfacingRecord,
    SurfacingStore,
    get_store,
    reset_for_tests,
)

__all__ = [
    "DISMISSAL_SUPPRESS_THRESHOLD",
    "HIGH_SALIENCE_THRESHOLD",
    "PRESENCE_ABSENT",
    "PRESENCE_ACTIVE",
    "PRESENCE_IDLE",
    "URGENT_WINDOW_HOURS",
    "SurfacingAction",
    "SurfacingCandidate",
    "SurfacingContext",
    "SurfacingDecision",
    "decide",
    "VALID_OUTCOMES",
    "SurfacingRecord",
    "SurfacingStore",
    "get_store",
    "reset_for_tests",
    "derive_presence",
    "build_deadline_candidates",
    "run_silence_producers",
]
