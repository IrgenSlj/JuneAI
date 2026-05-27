"""Schemas for the Quick Capture endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CaptureRequest(BaseModel):
    """A raw thought to capture and classify."""

    text: str
    source: str = "chat"


class CaptureCandidate(BaseModel):
    """A proposed action intent derived from a capture."""

    id: str
    kind: str
    title: str
    summary: str
    risk: str
    requires_approval: bool
    approval_status: str
    can_commit: bool


class CaptureResponse(BaseModel):
    """Result of classifying a capture: the item, candidates, and any message."""

    id: str
    text: str
    kinds: list[str]
    candidates: list[CaptureCandidate]
    message: str = ""
    created_at: str


class CaptureRecentResponse(BaseModel):
    """Recent captures for this user, newest first."""

    items: list[dict[str, Any]]
