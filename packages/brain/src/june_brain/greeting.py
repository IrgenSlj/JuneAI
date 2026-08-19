"""Startup greeting — local, deterministic, no model call, never cloud.

June greets the user when they open an empty chat. This is intentionally a
template over recalled memory (not an LLM call): it is instant, works before
the model has warmed, and honours local-first by construction. It is a small
seed of the Daily Home.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .failure import degrade_quietly


def _time_of_day(hour: int) -> str:
    if hour < 12:
        return "Good morning"
    if hour < 18:
        return "Good afternoon"
    return "Good evening"


def build_greeting(user_id: str, name: str = "") -> dict[str, Any]:
    """Return ``{"greeting": str, "has_context": bool}`` for the given user.

    Uses the display name, the local time of day, and — if available — the most
    recent thing June learned, so the greeting references something real. All
    recall is best-effort: any failure degrades to a plain welcome.
    """
    name = (name or "").strip()
    opener = f"{_time_of_day(datetime.now().hour)}, {name}." if name else f"{_time_of_day(datetime.now().hour)}."

    reference = ""
    returning = False
    try:
        from .memory import Memory, VectorStore

        returning = len(Memory(user_id).load_chat()) > 0
        facts = VectorStore(user_id).list_facts(limit=50)
        if facts:
            newest = max(facts, key=lambda f: str(f.get("created_at", "")))
            reference = str(newest.get("text", "")).strip()[:120]
    except Exception:  # noqa: BLE001 — greeting is best-effort, never fatal
        degrade_quietly("greeting personalisation")

    if reference:
        body = f'{opener} Last time, you mentioned: “{reference}” What’s on your mind?'
    elif returning:
        body = f"{opener} Welcome back. What would you like to do?"
    else:
        body = f"{opener} I’m June, your private assistant. What’s on your mind?"

    return {"greeting": body, "has_context": bool(reference)}
