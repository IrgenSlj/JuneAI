"""Silence Model schemas — the 'why June stayed quiet' surface (ADR 0023)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SurfacingDecisionView(BaseModel):
    """One recorded surface-vs-defer decision."""

    id: str
    candidate_id: str
    kind: str = Field(..., description="Candidate kind: deadline | contradiction | promise_nudge | ...")
    action: str = Field(..., description="'now', 'batch', or 'suppress'.")
    reason: str = Field(..., description="Truthful plain-English reason naming the deciding feature.")
    features: dict[str, Any] = Field(default_factory=dict, description="Inputs that produced the decision.")
    ts: str = Field(..., description="ISO-8601 UTC timestamp of the decision.")
    outcome: str | None = Field(
        default=None, description="engaged | dismissed | expired, once known; null otherwise."
    )
    surfaced_at: str | None = Field(
        default=None, description="When a batched item was drained into a digest; null if still held."
    )
    verdict: str | None = Field(
        default=None, description="User judgment of the decision: 'good' | 'bad'; null if none."
    )


class SurfacingPageResponse(BaseModel):
    """GET /system/surfacing payload — recent decisions, newest-first."""

    entries: list[SurfacingDecisionView] = Field(default_factory=list)
    count: int = 0
    next_cursor: str | None = Field(
        default=None, description="Pass as ?cursor= to fetch the next (older) page; null when no more."
    )


class SurfacingDrainResponse(BaseModel):
    """POST /system/surfacing/drain result — the batch digest drained at this user action.

    ``drained`` is the list of batch decisions that were held since the last drain
    and are now marked surfaced. A second call returns an empty list. This is the
    explicit user-pulled 'catch me up' boundary (ADR 0023: drain is an event-boundary
    action triggered by the user, never by a timer).
    """

    drained: list[SurfacingDecisionView] = Field(default_factory=list)
    count: int = 0


class SurfacingFeedbackRequest(BaseModel):
    """POST /system/surfacing/{id}/feedback body."""

    verdict: Literal["good", "bad"] = Field(
        ..., description="Was June's decision good (right call) or bad (wrong call)?"
    )


class SurfacingFeedbackResponse(BaseModel):
    """POST /system/surfacing/{id}/feedback result — the updated decision."""

    id: str
    verdict: str
    outcome: str | None = Field(
        default=None, description="The outcome the verdict mapped to (engaged | dismissed)."
    )
