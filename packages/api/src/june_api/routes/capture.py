"""Quick Capture routes (P3) — classify a thought into typed candidates."""

from __future__ import annotations

from fastapi import APIRouter
from june_brain.capture import process_capture
from june_brain.events import EventLedger

from ..schemas.capture import (
    CaptureCandidate,
    CaptureRecentResponse,
    CaptureRequest,
    CaptureResponse,
)

router = APIRouter(tags=["capture"])


@router.post("/capture/{user_id}", response_model=CaptureResponse, status_code=201)
def create_capture(user_id: str, body: CaptureRequest) -> CaptureResponse:
    """Capture a thought, classify it locally, and return candidate actions."""
    result = process_capture(body.text, user_id=user_id, source=body.source)
    cap = result["capture"]
    return CaptureResponse(
        id=cap["id"],
        text=cap["text"],
        kinds=cap["kinds"],
        candidates=[CaptureCandidate(**c) for c in _candidate_views(result["candidates"])],
        message=result["message"],
        created_at=cap["created_at"],
    )


@router.get("/capture/{user_id}/recent", response_model=CaptureRecentResponse)
def recent_captures(user_id: str, limit: int = 20) -> CaptureRecentResponse:
    """Return this user's recent captures, newest first."""
    items = [c.to_dict() for c in EventLedger().recent_captures(user_id, limit=limit)]
    return CaptureRecentResponse(items=items)


def _candidate_views(candidates: list[dict]) -> list[dict]:
    """Project the ledger intent dicts onto the CaptureCandidate fields."""
    return [
        {
            "id": c["id"],
            "kind": c["kind"],
            "title": c["title"],
            "summary": c["summary"],
            "risk": c["risk"],
            "requires_approval": c["requires_approval"],
            "approval_status": c["approval_status"],
            "can_commit": c["can_commit"],
        }
        for c in candidates
    ]
