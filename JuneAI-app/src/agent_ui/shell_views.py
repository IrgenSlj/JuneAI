"""Streamlit shell view helpers extracted from the main app."""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any

import streamlit as st

from .onboarding import FirstRunSummary


def render_layout_controls(
    current_layout: str,
    *,
    on_change: Callable[[str], None],
) -> None:
    """Render the layout mode controls for the right rail."""
    st.markdown('<div class="june-label">Window</div>', unsafe_allow_html=True)
    for label, value in (("Chat", "chat"), ("Split", "split"), ("Focus", "focus")):
        pressed = st.button(
            f"{label}{' · active' if current_layout == value else ''}",
            key=f"layout_{value}",
            use_container_width=True,
            disabled=current_layout == value,
        )
        if pressed:
            on_change(value)


def render_command_bar(
    *,
    active_view: str,
    layout: str,
    show_right_panel: bool,
    show_onboarding: bool,
    on_select_view: Callable[[str], None],
    on_toggle_rail: Callable[[bool], None],
) -> None:
    """Render the primary command bar for shell navigation."""
    st.markdown('<div class="june-command-label">Control</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="june-meta-row" style="margin-bottom:0.45rem;">'
        f'<div class="june-chip">rail: {html.escape(active_view.title())}</div>'
        f'<div class="june-chip">layout: {html.escape(layout)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    command_specs: list[tuple[str, str, bool]] = [
        ("Today", "today", active_view == "today"),
        ("Memory", "memory", active_view == "memory"),
        ("Workspace", "workspace", active_view == "workspace"),
        ("Trust", "debug", active_view == "debug"),
    ]
    if show_onboarding:
        command_specs.insert(1, ("Setup", "onboarding", active_view == "onboarding"))
    widths = [1.0] * len(command_specs) + [1.1]
    cols = st.columns(widths, gap="small")
    for col, (label, value, active) in zip(cols[:-1], command_specs):
        with col:
            if st.button(
                label,
                key=f"rail_view_{value}",
                use_container_width=True,
                disabled=active,
            ):
                on_select_view(value)
    with cols[-1]:
        label = "Hide rail" if show_right_panel else "Show rail"
        if st.button(label, key="toggle_right_rail", use_container_width=True):
            on_toggle_rail(not show_right_panel)


def render_onboarding_surface(
    summary: FirstRunSummary,
    *,
    recommended_label: str,
    on_recommended: Callable[[], None],
) -> None:
    """Render the dedicated onboarding rail surface."""
    stage_rows = "".join(
        '<div class="june-item">'
        f'<div class="june-label">{html.escape(stage.title)}</div>'
        f'<div class="june-item-meta">{html.escape(stage.summary)}</div>'
        f'<div class="june-item-meta">{html.escape(stage.step)}</div>'
        '</div>'
        for stage in summary.stages
    )
    st.markdown(
        '<div class="june-rail-card june-rail-card-primary">'
        '<div class="june-label">Onboarding</div>'
        '<div class="june-title">Set June up once, then work from Today</div>'
        f'<div class="june-panel-caption">{html.escape(summary.headline)}</div>'
        f'<div class="june-item-meta">{html.escape(summary.recommended_next_reason)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns([1.15, 0.85], gap="small")
    with cols[0]:
        st.markdown(stage_rows, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(
            '<div class="june-rail-card june-rail-card-quiet">'
            '<div class="june-label">Recommended next step</div>'
            f'<div class="june-title">{html.escape(summary.recommended_next_action)}</div>'
            f'<div class="june-panel-caption">{html.escape(recommended_label)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button(recommended_label, key="onboarding_recommended_action", use_container_width=True):
            on_recommended()


def render_turn_save_feedback(
    save_summary: dict[str, Any] | None,
    *,
    on_open_chapter: Callable[[str], None],
) -> None:
    """Render compact post-turn save feedback with chapter shortcuts."""
    if not save_summary:
        return
    items = [
        str(item)
        for item in save_summary.get("items", [])
        if isinstance(item, str) and str(item).strip()
    ]
    if not items:
        return
    st.markdown(
        '<div class="june-rail-card june-rail-card-quiet" style="margin-top:0.5rem;">'
        '<div class="june-label">Saved this turn</div>'
        f'<div class="june-item-meta">{html.escape(str(save_summary.get("label", "What June saved")))}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="june-item-meta" style="margin-top:-0.2rem;margin-bottom:0.35rem;">'
        + "<br>".join(html.escape(item) for item in items[:4])
        + "</div>",
        unsafe_allow_html=True,
    )
    actions = [
        action
        for action in save_summary.get("actions", [])
        if isinstance(action, dict) and str(action.get("chapter_key", "")).strip()
    ]
    if not actions:
        return
    cols = st.columns(min(3, len(actions)), gap="small")
    for col, action in zip(cols, actions[:3]):
        with col:
            label = str(action.get("chapter_label", "Open"))
            chapter_key = str(action.get("chapter_key", ""))
            if st.button(f"Open {label}", key=f"save_feedback_{chapter_key}", use_container_width=True):
                on_open_chapter(chapter_key)
