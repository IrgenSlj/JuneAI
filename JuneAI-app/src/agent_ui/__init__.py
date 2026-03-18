"""Reusable UI helpers for the JuneAI Streamlit app."""

from .chapters import CHAPTERS, chapter_items, chapter_subtitle
from .rendering import (
    default_ui_state,
    energy_dots_html,
    extract_text,
    habit_ring_svg,
    render_activity,
    render_capture_health,
    render_list,
    render_memory_focus,
    render_notifications,
    render_workspace,
    transcript_html,
    water_dots_html,
)

__all__ = [
    "CHAPTERS",
    "chapter_items",
    "chapter_subtitle",
    "default_ui_state",
    "energy_dots_html",
    "extract_text",
    "habit_ring_svg",
    "render_activity",
    "render_capture_health",
    "render_list",
    "render_memory_focus",
    "render_notifications",
    "render_workspace",
    "transcript_html",
    "water_dots_html",
]
