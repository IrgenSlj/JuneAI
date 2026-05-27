"""Schema for the startup greeting endpoint."""

from __future__ import annotations

from pydantic import BaseModel


class GreetingResponse(BaseModel):
    """A short, local greeting for the empty-chat state."""

    greeting: str
    has_context: bool
