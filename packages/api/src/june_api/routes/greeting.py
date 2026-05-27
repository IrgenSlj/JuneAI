"""Startup greeting route — local, cheap, never cloud."""

from __future__ import annotations

from fastapi import APIRouter
from june_brain.greeting import build_greeting

from ..schemas.greeting import GreetingResponse

router = APIRouter(tags=["greeting"])


@router.get("/greeting/{user_id}", response_model=GreetingResponse)
def get_greeting(user_id: str, name: str = "") -> GreetingResponse:
    """Return a short greeting for the empty chat, built from local memory."""
    return GreetingResponse(**build_greeting(user_id, name))
