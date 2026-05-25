"""Notification endpoints — bridge between the notification bus and shells."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from june_brain.notification import Notification, bus

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.post("/dispatch")
def dispatch_notification(body: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a notification through the notification bus.

    The desktop shell (Tauri) calls this when the user triggers a local
    notification. The bus delivers it to all registered channels (log,
    desktop, Telegram, etc.).
    """
    notification = Notification(
        title=body.get("title", ""),
        body=body.get("body", ""),
        priority=body.get("priority", "medium"),
        channel_hint=body.get("channel_hint"),
        source=body.get("source", "api"),
        metadata=body.get("metadata", {}),
    )
    results = bus.dispatch(notification)
    return {"dispatched": len(results), "channels": [r[0] for r in results if r[1]]}
