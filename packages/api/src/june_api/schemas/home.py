"""Home 'what June is holding' schema — the control-room surface (handoff S5)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .silence import SurfacingDecisionView


class NextDeadlineView(BaseModel):
    """The soonest deadline across open promises, if any."""

    task_id: str
    goal: str
    due_at: str = Field(..., description="ISO-8601 deadline.")


class HomeHoldings(BaseModel):
    """Read-only aggregation of what June is holding for the user.

    Pure aggregation over existing promise state, the Silence digest, and the
    Trust Ledger summary — no new state of its own.
    """

    open_promises: int = Field(default=0, description="Active promises (planning/running/paused/awaiting).")
    waiting_on_user: int = Field(default=0, description="Promises awaiting the user's input.")
    blocked_by_local_only: int = Field(
        default=0, description="Promises blocked by local-only mode (a networked tool the dial forbids)."
    )
    next_deadline: NextDeadlineView | None = Field(
        default=None, description="Soonest deadline across open promises, or null."
    )
    held_digest: list[SurfacingDecisionView] = Field(
        default_factory=list, description="Batched surfacings June held for the next opening."
    )
    held_count: int = Field(default=0, description="Total held (batched, not yet surfaced) items.")
    needs_now: list[SurfacingDecisionView] = Field(
        default_factory=list,
        description="Unresolved 'now' surfacings June judged need attention immediately.",
    )
    egress_today: int = Field(default=0, description="Cloud-egress ledger entries since midnight UTC.")
    chain_verified: bool | None = Field(
        default=None, description="Result of the last ledger verification this session; null if none."
    )
