"""Layout helpers for the JuneAI app shell."""

from __future__ import annotations

from typing import Any


def normalize_layout(layout: str) -> str:
    """Normalize a requested layout value."""
    chosen = layout.strip().lower()
    return chosen if chosen in {"split", "focus", "chat"} else "split"


def normalize_rail_view(rail_view: str) -> str:
    """Normalize a requested rail surface."""
    chosen = rail_view.strip().lower()
    return chosen if chosen in {"today", "onboarding", "memory", "workspace", "debug"} else "today"


def sync_ui_layout(session_state: Any, layout: str) -> str:
    """Persist the active single-page layout mode."""
    chosen = normalize_layout(layout)
    if "ui_state" in session_state:
        session_state["ui_state"]["layout"] = chosen
    else:
        session_state["layout"] = chosen
    return chosen


def sync_right_panel_visibility(session_state: Any, is_visible: object) -> bool:
    """Persist whether the right rail is visible."""
    visible = bool(is_visible)
    if "ui_state" in session_state:
        session_state["ui_state"]["show_right_panel"] = visible
    else:
        session_state["show_right_panel"] = visible
    return visible


def sync_rail_view(session_state: Any, rail_view: str) -> str:
    """Persist the currently selected right-rail surface."""
    chosen = normalize_rail_view(rail_view)
    session_state["rail_view"] = chosen
    return chosen


__all__ = [
    "layout_column_widths",
    "normalize_layout",
    "normalize_rail_view",
    "sync_rail_view",
    "sync_right_panel_visibility",
    "sync_ui_layout",
]


def layout_column_widths(layout: str, show_right_panel: bool) -> list[float]:
    """Return conversation/right-panel widths for the current page mode."""
    if not show_right_panel:
        return [1.0]
    return {
        "chat": [2.45, 0.8],
        "focus": [1.3, 1.55],
        "split": [1.9, 1.0],
    }.get(layout, [1.9, 1.0])
