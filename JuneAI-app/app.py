"""JuneAI Streamlit frontend.

Run with: streamlit run app.py
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from agent.config import RuntimeConfig, resolve_runtime_config, runtime_preset_options
from agent.graph import create_june_agent, startup_error
from agent.memory import Memory
from agent.runtime_privacy import build_runtime_privacy_status
from agent.skills import DEFAULT_SKILL, SKILLS, infer_skill_from_text
from agent_ui.chapters import (
    chapter_items,
    chapter_subtitle,
)
from agent_ui.focus_views import (
    body_metric_stats as detail_body_metric_stats,
    body_snapshot_line as detail_body_snapshot_line,
    recent_body_series as detail_recent_body_series,
    render_detail_focus as render_chapter_detail_focus,
)
from agent_ui.layout import (
    layout_column_widths,
    sync_rail_view,
    sync_right_panel_visibility,
    sync_ui_layout,
)
from agent_ui.panels import (
    SetupProgressModel,
    build_debug_panel_model,
    build_memory_panel_model,
    build_today_panel_model,
    build_workspace_panel_model,
)
from agent_ui.onboarding import FirstRunSummary, first_run_setup_summary
from agent_ui.rendering import (
    energy_dots_html,
    extract_text,
    habit_ring_svg,
    render_activity,
    render_list,
    render_memory_focus,
    render_workspace,
    transcript_html,
    water_dots_html,
)
from agent_ui.shell_views import (
    render_command_bar as render_shell_command_bar,
    render_layout_controls as render_shell_layout_controls,
    render_onboarding_surface,
    render_turn_save_feedback,
)
from agent_ui.state import (
    initialize_session_state,
    reset_session_state,
    sync_selected_chapter,
)

st.set_page_config(page_title="June", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --june-text:        #161410;
        --june-muted:       #6c655d;
        --june-line:        rgba(22, 20, 16, 0.08);
        --june-accent:      #0f5f4a;
        --june-accent-soft: rgba(15, 95, 74, 0.08);
        --june-accent-mist: rgba(15, 95, 74, 0.04);
        --june-user:        rgba(15, 95, 74, 0.07);
        --june-panel:       #ffffff;
        --june-radius:      18px;
        --june-shadow:      0 8px 32px rgba(40, 28, 18, 0.06);
        --june-shadow-lg:   0 16px 48px rgba(40, 28, 18, 0.10);
    }

    html, body, [class*="css"],
    [data-testid="stAppViewContainer"],
    [data-testid="stMarkdownContainer"] {
        font-family: "IBM Plex Mono", monospace;
        color: var(--june-text);
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stApp"],
    .main, .block-container {
        background: #f5f4f2;
    }

    .block-container {
        max-width: 1440px;
        padding-top: 0.75rem;
        padding-bottom: 1rem;
    }

    /* ── Sidebar ───────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #fbfaf8 100%);
        border-right: 1px solid var(--june-line);
        box-shadow: 4px 0 18px rgba(40, 28, 18, 0.035);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    /* ── Form inputs ───────────────────────────────────────── */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: #ffffff;
        border: 1px solid var(--june-line);
        border-radius: 14px;
        color: var(--june-text);
        font-family: "IBM Plex Mono", monospace;
    }

    /* ── Buttons ───────────────────────────────────────────── */
    .stButton > button {
        border-radius: 12px;
        border: 1px solid var(--june-line);
        background: #ffffff;
        color: var(--june-text);
        min-height: 2.2rem;
        font-family: "IBM Plex Mono", monospace;
        font-size: 10px;
        transition: border-color 0.18s, color 0.18s, background 0.18s, box-shadow 0.18s;
    }

    .stButton > button:hover {
        border-color: rgba(15, 95, 74, 0.30);
        color: var(--june-accent);
        background: var(--june-accent-soft);
        box-shadow: 0 2px 8px rgba(15, 95, 74, 0.08);
    }

    .june-chapter-grid .stButton > button {
        min-height: 4.4rem;
        border-radius: 14px;
        text-align: left;
        padding: 0.65rem 0.75rem;
        font-size: 11px;
    }

    /* ── Animations ────────────────────────────────────────── */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0);   }
    }
    @keyframes waterPop {
        0%   { transform: scale(0.6); opacity: 0.3; }
        60%  { transform: scale(1.3); }
        100% { transform: scale(1);   opacity: 1;   }
    }
    @keyframes writingPulse {
        0%, 100% { opacity: 1;   }
        50%       { opacity: 0.2; }
    }
    @keyframes breathe {
        0%, 100% { opacity: 1;    }
        50%       { opacity: 0.55; }
    }
    @keyframes badgePop {
        0%   { transform: scale(0.8); opacity: 0; }
        60%  { transform: scale(1.1); }
        100% { transform: scale(1);   opacity: 1; }
    }
    @keyframes panelIn {
        from { opacity: 0; transform: translateX(12px); }
        to   { opacity: 1; transform: translateX(0);    }
    }

    /* ── Surface cards ─────────────────────────────────────── */
    .june-surface {
        background: var(--june-panel);
        border: 1px solid var(--june-line);
        border-radius: var(--june-radius);
        box-shadow: var(--june-shadow);
        padding: 0.9rem;
        margin-bottom: 0.75rem;
        animation: fadeUp 0.22s ease both;
    }

    /* ── Brand ─────────────────────────────────────────────── */
    .june-brand {
        font-family: "Syne", sans-serif;
        letter-spacing: -0.04em;
        font-size: 2.2rem;
        line-height: 0.92;
        margin: 0 0 0.35rem 0;
        animation: breathe 3.2s ease-in-out infinite;
    }

    .june-copy {
        color: var(--june-muted);
        font-size: 11px;
        line-height: 1.55;
        margin-bottom: 0.5rem;
    }

    /* ── Labels ────────────────────────────────────────────── */
    .june-label {
        color: var(--june-accent);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 9px;
        margin-bottom: 0.35rem;
    }

    .june-title {
        font-family: "Syne", sans-serif;
        font-size: 1.1rem;
        letter-spacing: -0.025em;
        margin: 0 0 0.5rem 0;
    }

    .june-subtitle {
        color: var(--june-muted);
        font-size: 11px;
        margin-bottom: 0.5rem;
    }

    .june-meta-row {
        display: flex;
        gap: 0.35rem;
        flex-wrap: wrap;
        margin-bottom: 0.6rem;
    }

    .june-chip {
        border: 1px solid var(--june-line);
        border-radius: 999px;
        padding: 0.22rem 0.5rem;
        font-size: 9px;
        color: var(--june-muted);
        background: #ffffff;
    }

    .june-command-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-bottom: 0.75rem;
    }

    .june-command-label {
        color: var(--june-accent);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 9px;
        margin-bottom: 0.35rem;
    }

    /* ── Conversation ──────────────────────────────────────── */
    .june-transcript {
        max-height: 62vh;
        overflow-y: auto;
        padding-right: 0.1rem;
    }

    .june-message {
        border: 1px solid var(--june-line);
        border-radius: 14px;
        padding: 0.7rem 0.85rem;
        margin-bottom: 0.5rem;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        font-size: 12px;
        line-height: 1.6;
        animation: fadeUp 0.18s ease both;
    }

    .june-message-user      { background: var(--june-user); }
    .june-message-assistant { background: #ffffff; }

    .june-message-label {
        color: var(--june-accent);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 9px;
        margin-bottom: 0.25rem;
    }

    .june-writing {
        color: var(--june-accent);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 9px;
        margin-top: 0.4rem;
        animation: writingPulse 1.3s ease-in-out infinite;
    }

    /* ── Lists ─────────────────────────────────────────────── */
    .june-list { display: grid; gap: 0.45rem; }

    .june-item {
        border: 1px solid var(--june-line);
        border-radius: 12px;
        padding: 0.55rem 0.7rem;
        background: linear-gradient(180deg, #ffffff 0%, rgba(255,255,255,0.86) 100%);
        transition: border-color 0.18s, transform 0.18s, box-shadow 0.18s;
    }
    .june-item:hover {
        border-color: rgba(15, 95, 74, 0.22);
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(40, 28, 18, 0.05);
    }

    .june-item-title {
        font-family: "Syne", sans-serif;
        font-size: 0.85rem;
        margin-bottom: 0.1rem;
    }

    .june-item-meta {
        color: var(--june-muted);
        font-size: 10px;
        line-height: 1.5;
    }

    /* ── Today panel (in sidebar) ──────────────────────────── */
    .june-today-divider {
        border: none;
        border-top: 1px solid var(--june-line);
        margin: 0.8rem 0 0.75rem 0;
    }

    .june-section-label {
        color: var(--june-accent);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 9px;
        margin: 0.5rem 0 0.25rem 0;
        padding-top: 0.45rem;
        border-top: 1px solid var(--june-line);
    }
    .june-section-label.first { border-top: none; margin-top: 0; padding-top: 0; }

    .june-water-track {
        display: flex;
        gap: 4px;
        align-items: center;
        flex-wrap: wrap;
        padding: 0.2rem 0;
    }

    .june-water-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        display: inline-block;
        transition: background 0.25s ease, transform 0.2s ease;
    }
    .june-water-dot.filled {
        background: #0f5f4a;
        animation: waterPop 0.35s ease both;
    }
    .june-water-dot.empty {
        background: rgba(22, 20, 16, 0.08);
        border: 1px solid rgba(22, 20, 16, 0.14);
    }

    .june-metric-dots { display: inline-flex; gap: 3px; align-items: center; }
    .june-metric-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        display: inline-block;
        transition: background 0.2s;
    }
    .june-metric-dot.active   { background: #0f5f4a; }
    .june-metric-dot.inactive { background: rgba(22, 20, 16, 0.12); }

    .june-progress-track {
        height: 2px;
        background: rgba(22, 20, 16, 0.08);
        border-radius: 999px;
        overflow: hidden;
        margin: 0.25rem 0 0.5rem 0;
    }
    .june-progress-inner {
        height: 100%;
        background: #0f5f4a;
        border-radius: 999px;
        transition: width 0.7s cubic-bezier(0.34, 1.56, 0.64, 1);
    }

    .june-badge {
        display: inline-block;
        padding: 0.15rem 0.45rem;
        border-radius: 999px;
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        animation: badgePop 0.3s ease both;
    }
    .june-badge-done {
        background: rgba(15, 95, 74, 0.10);
        color: #0f5f4a;
        border: 1px solid rgba(15, 95, 74, 0.22);
    }
    .june-badge-rest {
        background: rgba(22, 20, 16, 0.04);
        color: var(--june-muted);
        border: 1px solid var(--june-line);
    }

    .june-body-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.2rem 0;
        font-size: 11px;
    }
    .june-body-key { color: var(--june-muted); font-size: 10px; }

    /* ── Right panel ───────────────────────────────────────── */
    .june-right-panel {
        display: grid;
        gap: 0.65rem;
        animation: panelIn 0.28s ease both;
        max-height: calc(100vh - 2rem);
        overflow-y: auto;
        position: sticky;
        top: 0.75rem;
    }

    .june-panel-card {
        background: #ffffff;
        border: 1px solid var(--june-line);
        border-radius: 22px;
        box-shadow: var(--june-shadow-lg);
        padding: 1rem;
    }

    .june-panel-card-quiet {
        box-shadow: var(--june-shadow);
    }

    .june-panel-divider {
        height: 1px;
        background: var(--june-line);
        margin: 0.8rem 0;
    }

    .june-panel-kicker {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
        margin-bottom: 0.55rem;
    }

    .june-panel-kicker-left {
        min-width: 0;
    }

    .june-panel-kicker-right {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        min-width: 0;
        flex: 1;
    }

    .june-active-tag {
        border: 1px solid rgba(15, 95, 74, 0.18);
        background: rgba(15, 95, 74, 0.07);
        color: var(--june-accent);
        border-radius: 999px;
        padding: 0.18rem 0.5rem;
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .june-compact-copy {
        color: var(--june-muted);
        font-size: 10px;
        line-height: 1.45;
        text-align: right;
    }

    .june-panel-caption {
        color: var(--june-muted);
        font-size: 10px;
        line-height: 1.45;
        margin-top: -0.15rem;
        margin-bottom: 0.7rem;
    }

    .june-primary-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.65rem;
    }

    .june-primary-copy {
        min-width: 0;
        flex: 1;
    }

    .june-attention-strip {
        display: grid;
        gap: 0.4rem;
        margin-bottom: 0.7rem;
    }

    .june-attention-card {
        border: 1px solid rgba(15, 95, 74, 0.16);
        background: linear-gradient(135deg, rgba(15, 95, 74, 0.08), rgba(15, 95, 74, 0.02));
        border-radius: 14px;
        padding: 0.7rem 0.8rem;
        box-shadow: 0 10px 22px rgba(15, 95, 74, 0.05);
    }

    .june-attention-title {
        font-family: "Syne", sans-serif;
        font-size: 0.92rem;
        margin-bottom: 0.15rem;
    }

    .june-attention-copy {
        color: var(--june-muted);
        font-size: 10px;
        line-height: 1.45;
    }

    .june-rail-grid {
        display: grid;
        gap: 0.8rem;
    }

    .june-rail-card {
        background: linear-gradient(180deg, #ffffff 0%, rgba(255,255,255,0.94) 100%);
        border: 1px solid var(--june-line);
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(40, 28, 18, 0.045);
        padding: 0.85rem;
    }

    .june-rail-card-primary {
        border-color: rgba(15, 95, 74, 0.14);
        box-shadow: 0 14px 34px rgba(15, 95, 74, 0.08);
        background: linear-gradient(180deg, #ffffff 0%, rgba(15, 95, 74, 0.025) 100%);
    }

    .june-rail-card-quiet {
        box-shadow: none;
        background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(255,255,255,0.78) 100%);
    }

    .june-kpi-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.45rem;
    }

    .june-kpi {
        border: 1px solid var(--june-line);
        border-radius: 14px;
        padding: 0.6rem;
        background: rgba(255, 255, 255, 0.8);
    }

    .june-kpi-value {
        font-family: "Syne", sans-serif;
        font-size: 1rem;
        line-height: 1;
        margin-bottom: 0.15rem;
    }

    .june-kpi-label {
        color: var(--june-muted);
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    .june-runtime-pill {
        border: 1px solid rgba(15, 95, 74, 0.16);
        background: rgba(15, 95, 74, 0.06);
        color: var(--june-accent);
        border-radius: 999px;
        padding: 0.18rem 0.5rem;
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        display: inline-block;
    }

    .june-starter-grid {
        display: grid;
        gap: 0.55rem;
        margin: 0.55rem 0 0.35rem 0;
    }

    .june-starter-copy {
        color: var(--june-muted);
        font-size: 10px;
        line-height: 1.5;
    }

    /* ── Stats ─────────────────────────────────────────────── */
    .june-stat-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.45rem;
    }

    .june-stat-card {
        border: 1px solid var(--june-line);
        border-radius: 12px;
        padding: 0.55rem;
        background: linear-gradient(180deg, #ffffff 0%, rgba(255,255,255,0.84) 100%);
        transition: border-color 0.18s, box-shadow 0.18s;
    }
    .june-stat-card:hover {
        border-color: rgba(15, 95, 74, 0.22);
        box-shadow: 0 3px 10px rgba(15, 95, 74, 0.06);
    }

    .june-stat-label {
        color: var(--june-muted);
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.2rem;
    }

    .june-stat-value {
        font-family: "Syne", sans-serif;
        font-size: 1.1rem;
        line-height: 1;
    }

    .june-focus-hero {
        border: 1px solid rgba(15, 95, 74, 0.12);
        background: linear-gradient(135deg, rgba(15,95,74,0.08), rgba(15,95,74,0.015));
        border-radius: 16px;
        padding: 0.75rem 0.85rem;
        margin-bottom: 0.7rem;
    }

    .june-focus-copy {
        color: var(--june-muted);
        font-size: 10px;
        line-height: 1.5;
        margin-top: 0.2rem;
    }

    .june-mini-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.45rem;
        margin: 0.55rem 0 0.7rem 0;
    }

    .june-mini-card {
        border: 1px solid var(--june-line);
        border-radius: 12px;
        padding: 0.55rem 0.6rem;
        background: var(--june-accent-mist);
    }

    .june-mini-label {
        color: var(--june-muted);
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.2rem;
    }

    .june-mini-value {
        font-family: "Syne", sans-serif;
        font-size: 1rem;
        line-height: 1;
    }

    ul { margin: 0.35rem 0 0 1rem; padding: 0; }

    /* Hide default Streamlit expander arrow styling a bit */
    details summary { font-size: 11px; color: var(--june-muted); }
    </style>
    """,
    unsafe_allow_html=True,
)

if startup_error:
    st.error("June could not start.")
    st.markdown("**Reason:** " + startup_error)
    st.markdown("**Fix:** Make sure Ollama is running and the model is available.")
    st.code("ollama serve\nollama pull mistral")
    st.stop()

WATER_GOAL = 8
RUNTIME_CONFIG = resolve_runtime_config()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def chapter_saved_count(memory: Memory, chapter_key: str) -> int:
    """Return saved item count for a chapter card."""
    return len(chapter_items(memory, chapter_key))


def append_activity(message: str) -> None:
    st.session_state.activity_log.append(message)


@st.cache_resource(show_spinner=False)
def get_compiled_agent(
    preset_key: str,
    provider: str,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    tool_strategy: str,
) -> Any:
    """Compile and cache a LangGraph agent for one resolved runtime profile."""
    runtime = resolve_runtime_config(preset_key)
    runtime = runtime.__class__(
        preset_key=preset_key,
        label=runtime.label,
        provider=provider,
        model=model,
        api_key=runtime.api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        tool_strategy=tool_strategy,
    )
    return create_june_agent(runtime=runtime)


def runtime_for_preset(preset_key: str) -> RuntimeConfig:
    """Resolve one runtime preset for the current app session."""
    return resolve_runtime_config(preset_key)


def build_turn_save_summary(last_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Build a compact in-chat summary of what June saved this turn."""
    label_map = {
        "save_goal": "Goal saved",
        "save_open_loop": "Open loop saved",
        "save_calendar_item": "Calendar item saved",
        "save_relationship_profile": "Relationship note saved",
        "save_user_preference": "Preference saved",
        "save_favorite_recommendation": "Favorite saved",
        "save_journal_entry": "Journal saved",
        "log_body_metrics": "Body check-in saved",
        "log_workout_session": "Workout saved",
        "log_nutrition": "Meal log saved",
        "log_water": "Water updated",
        "log_habit_completion": "Habit completion saved",
        "track_goal": "Goal saved",
        "create_habit": "Habit saved",
    }
    chapter_map = {
        "save_goal": ("plans", "Plans"),
        "track_goal": ("plans", "Plans"),
        "save_open_loop": ("plans", "Plans"),
        "save_calendar_item": ("calendar", "Calendar"),
        "save_relationship_profile": ("family", "Family"),
        "log_body_metrics": ("body", "Body"),
        "log_workout_session": ("gym", "Gym"),
        "log_nutrition": ("food", "Food"),
        "log_water": ("water", "Water"),
        "log_habit_completion": ("habits", "Habits"),
        "create_habit": ("habits", "Habits"),
    }
    items: list[str] = []
    actions: list[dict[str, str]] = []
    for call in last_calls:
        if call.get("status") != "success":
            continue
        tool_name = str(call.get("name", "")).strip()
        if not tool_name.startswith(("save_", "log_", "track_", "create_")):
            continue
        preview = str(call.get("preview", "")).strip() or "Saved to memory."
        label = label_map.get(tool_name, tool_name.replace("_", " ").title())
        items.append(f"{label}: {preview}")
        chapter = chapter_map.get(tool_name)
        if chapter and not any(item.get("chapter_key") == chapter[0] for item in actions):
            actions.append({"chapter_key": chapter[0], "chapter_label": chapter[1]})
    if not items:
        return None
    return {
        "type": "save_summary",
        "label": "What June saved",
        "items": items[:5],
        "actions": actions[:3],
    }


def onboarding_prompt(summary: FirstRunSummary) -> tuple[str, str, str]:
    """Return the highest-value starter prompt for the current setup state."""
    if not summary.missing_surfaces:
        return (
            "Keep one active priority visible",
            "My current priority is to keep one active goal visible and I want you to help me track the next step.",
            "onboarding | maintain priority",
        )
    target = summary.missing_surfaces[0].lower()
    prompt_map = {
        "calendar": (
            "Add an upcoming event",
            "Save this upcoming event to my calendar: Launch review on 2026-04-02. Remind me to review the onboarding copy.",
            "onboarding | calendar",
        ),
        "plans": (
            "Capture a current goal",
            "Track this goal for me: ship the next June onboarding sprint, next step is to polish the Today view.",
            "onboarding | plans",
        ),
        "habits": (
            "Add one habit",
            "Create a habit for me: 20 minute walk every day.",
            "onboarding | habits",
        ),
        "body metrics": (
            "Log a body check-in",
            "Log my body check-in: 7.5 hours sleep, energy 4/5, stress 2/5, soreness 1/5, resting heart rate 55, 9200 steps.",
            "onboarding | body",
        ),
        "family": (
            "Add one family profile",
            "Save this family context: Ava is my sister and we talk every Sunday.",
            "onboarding | family",
        ),
        "birthdays": (
            "Save one birthday",
            "Remember this birthday: Anna on 2026-04-14.",
            "onboarding | birthdays",
        ),
    }
    return prompt_map.get(
        target,
        (
            "Teach June one useful thing",
            "Save one useful detail about my life so you can remember it later.",
            "onboarding | starter",
        ),
    )


def render_first_run_onboarding(memory: Memory) -> None:
    """Render a staged onboarding card with one recommended next action."""
    summary = first_run_setup_summary(memory)
    if summary.has_data and len(summary.missing_surfaces) <= 1:
        return
    label, prompt, reason = onboarding_prompt(summary)
    render_onboarding_surface(
        summary,
        recommended_label=label,
        on_recommended=lambda: queue_prompt(prompt, reason),
    )


def render_layout_controls() -> None:
    """Let the user shrink or expand the single-page layout."""
    def on_layout_change(value: str) -> None:
        sync_ui_layout(st.session_state, value)
        append_activity(f"layout | {value}")
        st.rerun()

    render_shell_layout_controls(
        st.session_state.ui_state.get("layout", "split"),
        on_change=on_layout_change,
    )


def render_panel_lines(items: list[tuple[str, str]]) -> None:
    """Render compact title/copy rows inside a rail card."""
    if not items:
        return
    st.markdown(render_list(items), unsafe_allow_html=True)


def render_setup_progress_card(setup_model: SetupProgressModel) -> None:
    """Render setup progress from the structured panel model."""
    if setup_model.is_complete:
        return
    st.markdown(
        '<div class="june-rail-card">'
        '<div class="june-label">Setup</div>'
        f'<div class="june-title">{html.escape(setup_model.title)}</div>'
        f'<div class="june-panel-caption">{html.escape(setup_model.caption)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    render_panel_lines([(item.title, item.copy) for item in setup_model.missing_rows])


def queue_prompt(prompt: str, reason: str) -> None:
    """Queue a starter prompt into the normal generation flow."""
    st.session_state.pending_prompt = prompt.strip()
    st.session_state.active_skill_key = infer_skill_from_text(prompt)
    st.session_state.is_generating = True
    append_activity(reason)
    st.rerun()


def render_command_bar(show_right_panel: bool) -> None:
    """Render the primary command bar for shell navigation."""
    summary = first_run_setup_summary(Memory(st.session_state.get("profile_input", "admin")))
    def on_select_view(value: str) -> None:
        sync_rail_view(st.session_state, value)
        if not show_right_panel:
            sync_right_panel_visibility(st.session_state, True)
        append_activity(f"rail view | {value}")
        st.rerun()

    def on_toggle_rail(visible: bool) -> None:
        sync_right_panel_visibility(st.session_state, visible)
        append_activity(f"right rail | {'shown' if visible else 'hidden'}")
        st.rerun()

    render_shell_command_bar(
        active_view=st.session_state.rail_view,
        layout=str(st.session_state.ui_state.get("layout", "split")),
        show_right_panel=show_right_panel,
        show_onboarding=not (summary.has_data and len(summary.missing_surfaces) <= 1),
        on_select_view=on_select_view,
        on_toggle_rail=on_toggle_rail,
    )


def _daily_focus_items(memory: Memory) -> list[tuple[str, str]]:
    """Return high-priority daily attention items for the Today surface."""
    items: list[tuple[str, str]] = []
    notifications = memory.get_upcoming_notifications(limit=3)
    for item in notifications:
        timing = "today" if item["days_until"] == 0 else f"in {item['days_until']}d"
        items.append((item["title"], f"{item['kind']} · {timing}"))
    loops = memory.get_open_loops(status="open", limit=2)
    for loop in loops:
        suffix = f"due {loop['due_date']}" if loop.get("due_date") else "needs resolution"
        items.append((loop["topic"], f"open loop · {suffix}"))
    return items[:4]


def render_attention_strip(memory: Memory) -> None:
    """Render a concise 'what matters now' strip for the conversation column."""
    items = _daily_focus_items(memory)
    if not items:
        return
    cards = "".join(
        '<div class="june-attention-card">'
        f'<div class="june-attention-title">{html.escape(title)}</div>'
        f'<div class="june-attention-copy">{html.escape(copy)}</div>'
        '</div>'
        for title, copy in items[:2]
    )
    st.markdown('<div class="june-attention-strip">' + cards + '</div>', unsafe_allow_html=True)


def render_starter_prompts(memory: Memory) -> None:
    """Render guided starter prompts when the profile is still sparse."""
    if memory.get_progress_snapshot()["calendar_count"] or st.session_state.messages:
        return
    st.markdown(
        '<div class="june-rail-card"><div class="june-label">Start Here</div>'
        '<div class="june-title">Build June into your daily console</div>'
        '<div class="june-starter-copy">Seed the key surfaces once, then June can carry continuity forward with more useful suggestions and reminders.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    prompt_specs = [
        ("Set my weekly gym split", "Track my weekly gym split: push on Monday, pull on Wednesday, legs on Friday.", "starter | gym"),
        ("Log today’s body check-in", "Log my body check-in: 7.5 hours sleep, energy 4/5, stress 2/5, soreness 1/5, 10200 steps.", "starter | body"),
        ("Save key birthdays", "Save these birthdays: Anna on May 14 and Lucas on August 24.", "starter | birthdays"),
        ("Capture current priorities", "My current priorities are to improve sleep consistency, plan the mountain trip, and finish the website relaunch.", "starter | plans"),
    ]
    cols = st.columns(2, gap="small")
    for index, (label, prompt, reason) in enumerate(prompt_specs):
        with cols[index % 2]:
            if st.button(label, key=f"starter_prompt_{index}", use_container_width=True):
                queue_prompt(prompt, reason)


def _calendar_focus_items(memory: Memory, chapter_key: str) -> list[dict[str, Any]]:
    items = memory.get_calendar_items(status="", limit=20)
    if chapter_key == "trips":
        return [
            item for item in items
            if any(term in " ".join(str(item.get(field, "")).lower() for field in ("title", "details", "source"))
                   for term in ("trip", "travel", "flight"))
        ]
    if chapter_key == "birthdays":
        return [
            item for item in items
            if "birthday" in " ".join(str(item.get(field, "")).lower() for field in ("title", "details", "source"))
        ]
    return items


def render_calendar_focus(memory: Memory, chapter_key: str) -> None:
    """Render calendar-like entries with inline status actions."""
    items = _calendar_focus_items(memory, chapter_key)
    if not items:
        st.markdown(render_memory_focus(memory, chapter_key), unsafe_allow_html=True)
        return
    st.markdown(f'<div class="june-subtitle">{html.escape(chapter_subtitle(chapter_key))}</div>', unsafe_allow_html=True)
    for index, item in enumerate(items):
        label = (
            f"{item.get('date', 'date?')}"
            f"{' ' + item.get('time', '') if item.get('time') else ''}"
            f" · {item.get('title', 'Item')}"
            f" · {item.get('status', 'planned')}"
        )
        with st.expander(label, expanded=index == 0):
            if item.get("details"):
                st.write(item["details"])
            st.caption(f"source: {item.get('source', 'conversation')}")
            cols = st.columns(3, gap="small")
            options = [("planned", "Plan"), ("completed", "Done"), ("cancelled", "Cancel")]
            for col, (status, button_label) in zip(cols, options):
                with col:
                    if st.button(
                        button_label,
                        key=f"{chapter_key}_{index}_{status}",
                        use_container_width=True,
                        disabled=item.get("status", "").lower() == status,
                    ):
                        memory.update_calendar_item_status(
                            title=item["title"],
                            status=status,
                            date=item.get("date", ""),
                            time=item.get("time", ""),
                        )
                        append_activity(f"calendar status | {item['title']} -> {status}")
                        st.rerun()


def render_plan_focus(memory: Memory) -> None:
    """Render goals and open loops with status controls."""
    goals = memory.get_goals(status="", limit=20)
    loops = memory.get_open_loops(status="", limit=20)
    st.markdown(f'<div class="june-subtitle">{html.escape(chapter_subtitle("plans"))}</div>', unsafe_allow_html=True)
    if not goals and not loops:
        return
    if goals:
        st.markdown('<div class="june-panel-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="june-label">Goals</div>', unsafe_allow_html=True)
        for index, goal in enumerate(goals):
            with st.expander(
                f"{goal['title']} · {goal.get('status', 'active')} · {goal.get('category', 'personal')}",
                expanded=index == 0,
            ):
                if goal.get("next_step"):
                    st.write(f"Next: {goal['next_step']}")
                if goal.get("target_date"):
                    st.caption(f"Target: {goal['target_date']}")
                cols = st.columns(3, gap="small")
                for col, (status, label) in zip(cols, [("active", "Active"), ("paused", "Pause"), ("completed", "Done")]):
                    with col:
                        if st.button(
                            label,
                            key=f"goal_{index}_{status}",
                            use_container_width=True,
                            disabled=goal.get("status", "").lower() == status,
                        ):
                            memory.update_goal_status(goal["title"], status)
                            append_activity(f"goal status | {goal['title']} -> {status}")
                            st.rerun()
    if loops:
        st.markdown('<div class="june-panel-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="june-label">Open Loops</div>', unsafe_allow_html=True)
        for index, loop in enumerate(loops):
            with st.expander(
                f"{loop['topic']} · {loop.get('status', 'open')}",
                expanded=not goals and index == 0,
            ):
                if loop.get("next_step"):
                    st.write(f"Next: {loop['next_step']}")
                if loop.get("due_date"):
                    st.caption(f"Due: {loop['due_date']}")
                cols = st.columns(3, gap="small")
                for col, (status, label) in zip(cols, [("open", "Open"), ("resolved", "Resolve"), ("closed", "Close")]):
                    with col:
                        if st.button(
                            label,
                            key=f"loop_{index}_{status}",
                            use_container_width=True,
                            disabled=loop.get("status", "").lower() == status,
                        ):
                            memory.update_open_loop_status(loop["topic"], status)
                            append_activity(f"loop status | {loop['topic']} -> {status}")
                            st.rerun()


def render_habits_focus(memory: Memory) -> None:
    """Render tracked habits with inline completion controls."""
    habits = memory.get_habits()
    st.markdown(f'<div class="june-subtitle">{html.escape(chapter_subtitle("habits"))}</div>', unsafe_allow_html=True)
    if not habits:
        return
    st.markdown('<div class="june-panel-divider"></div>', unsafe_allow_html=True)
    for index, habit in enumerate(habits):
        cols = st.columns([1.25, 0.7, 0.55], gap="small")
        with cols[0]:
            st.markdown(f"**{habit['name']}**")
            st.caption(
                f"{habit.get('category', 'health')} · {habit.get('target_days', 'daily')} · "
                f"streak {habit.get('streak', 0)}d"
            )
        with cols[1]:
            status = "done today" if habit.get("done_today") else "pending"
            st.caption(status)
        with cols[2]:
            if st.button(
                "Done",
                key=f"habit_done_{index}",
                use_container_width=True,
                disabled=habit.get("done_today", False),
            ):
                item = memory.log_habit_completion(habit["name"])
                append_activity(f"habit done | {item['name']} | streak {item.get('streak', 0)}")
                st.rerun()


def render_water_focus(memory: Memory) -> None:
    """Render today's hydration controls inline."""
    count = memory.get_water_today()
    st.markdown(f'<div class="june-subtitle">{html.escape(chapter_subtitle("water"))}</div>', unsafe_allow_html=True)
    st.markdown(water_dots_html(count, WATER_GOAL), unsafe_allow_html=True)
    cols = st.columns([1, 0.5, 0.5], gap="small")
    with cols[0]:
        st.caption(f"{count}/{WATER_GOAL} glasses")
    with cols[1]:
        if st.button("−", key="focus_water_minus", use_container_width=True, disabled=count <= 0):
            memory.set_water(count - 1)
            append_activity("water | decrement")
            st.rerun()
    with cols[2]:
        if st.button("+", key="focus_water_plus", use_container_width=True):
            memory.log_water(1)
            append_activity("water | increment")
            st.rerun()


def _recent_body_series(memory: Memory, days: int = 7) -> list[dict[str, Any]]:
    """Return recent body entries in ascending date order."""
    items = memory.get_body_metrics(days=days)
    return sorted(items, key=lambda item: item.get("date", ""))


def _body_metric_stats(items: list[dict[str, Any]], key: str) -> tuple[float | None, float | None, float | None]:
    """Return current, delta-vs-previous, and simple average for one body metric."""
    values = [float(item.get(key, 0)) for item in items if item.get(key)]
    if not values:
        return None, None, None
    current = values[-1]
    previous = values[-2] if len(values) > 1 else None
    delta = current - previous if previous is not None else None
    average = sum(values) / len(values)
    return current, delta, average


def _metric_card(label: str, current: float | None, delta: float | None, average: float | None, suffix: str = "") -> str:
    """Render a compact metric card for the body trend section."""
    if current is None:
        return (
            '<div class="june-stat-card">'
            f'<div class="june-stat-label">{html.escape(label)}</div>'
            '<div class="june-stat-value">-</div>'
            '<div class="june-item-meta">No data</div>'
            '</div>'
        )
    current_text = f"{current:.1f}{suffix}" if isinstance(current, float) and not current.is_integer() else f"{int(current) if float(current).is_integer() else current}{suffix}"
    if delta is None:
        delta_text = "first entry"
    else:
        sign = "+" if delta > 0 else ""
        delta_value = f"{delta:.1f}" if isinstance(delta, float) and not float(delta).is_integer() else f"{int(delta)}"
        delta_text = f"vs prev {sign}{delta_value}{suffix}"
    avg_text = (
        f"7d avg {average:.1f}{suffix}"
        if average is not None and not float(average).is_integer()
        else (f"7d avg {int(average)}{suffix}" if average is not None else "")
    )
    return (
        '<div class="june-stat-card">'
        f'<div class="june-stat-label">{html.escape(label)}</div>'
        f'<div class="june-stat-value">{html.escape(current_text)}</div>'
        f'<div class="june-item-meta">{html.escape(delta_text)}</div>'
        f'<div class="june-item-meta">{html.escape(avg_text)}</div>'
        '</div>'
    )


def render_body_trend_card(memory: Memory, days: int = 7) -> None:
    """Render compact 7-day trend cards for key body metrics."""
    items = _recent_body_series(memory, days=days)
    if not items:
        st.caption("No body trend data yet.")
        return

    cards = []
    for label, key, suffix in [
        ("Sleep", "sleep_hours", "h"),
        ("Energy", "energy", "/5"),
        ("Stress", "stress", "/5"),
        ("Soreness", "soreness", "/5"),
        ("Weight", "weight_kg", "kg"),
    ]:
        current, delta, average = _body_metric_stats(items, key)
        cards.append(_metric_card(label, current, delta, average, suffix))
    st.markdown('<div class="june-stat-grid">' + "".join(cards) + '</div>', unsafe_allow_html=True)


def _body_snapshot_line(item: dict[str, Any]) -> str:
    """Build a compact one-line summary for a body check-in."""
    parts = []
    if item.get("sleep_hours"):
        parts.append(f"sleep {item['sleep_hours']:.1f}h")
    if item.get("energy"):
        parts.append(f"energy {item['energy']}/5")
    if item.get("stress"):
        parts.append(f"stress {item['stress']}/5")
    if item.get("soreness"):
        parts.append(f"soreness {item['soreness']}/5")
    if item.get("weight_kg"):
        parts.append(f"weight {item['weight_kg']:.1f}kg")
    return " · ".join(parts) if parts else "No body metrics recorded."


def render_body_focus(memory: Memory) -> None:
    """Render detailed body metrics with a richer daily log form."""
    today = memory.get_today_body_metrics()
    recent = memory.get_body_metrics(days=7)
    st.markdown(f'<div class="june-subtitle">{html.escape(chapter_subtitle("body"))}</div>', unsafe_allow_html=True)

    if today:
        details = []
        if today.get("weight_kg"):
            details.append(f"Weight {today['weight_kg']} kg")
        if today.get("sleep_hours"):
            details.append(f"Sleep {today['sleep_hours']} h")
        if today.get("sleep_quality"):
            details.append(f"Sleep quality {today['sleep_quality']}/5")
        if today.get("energy"):
            details.append(f"Energy {today['energy']}/5")
        if today.get("stress"):
            details.append(f"Stress {today['stress']}/5")
        if today.get("soreness"):
            details.append(f"Soreness {today['soreness']}/5")
        if today.get("resting_hr"):
            details.append(f"Resting HR {today['resting_hr']}")
        if today.get("steps"):
            details.append(f"Steps {today['steps']}")
        st.markdown("**Today**")
        st.caption(" | ".join(details) if details else "No body metrics logged today.")
        if today.get("notes"):
            st.write(today["notes"])
    else:
        st.caption("No body metrics logged today.")

    if recent:
        st.markdown("**7-day trend**")
        render_body_trend_card(memory, days=7)
        st.markdown("**Recent check-ins**")
        st.markdown(render_memory_focus(memory, "body"), unsafe_allow_html=True)

    with st.expander("Log body check-in", expanded=not bool(today)):
        with st.form("body_focus_form", clear_on_submit=False):
            top_left, top_right = st.columns(2, gap="small")
            with top_left:
                weight_kg = st.number_input("Weight kg", min_value=0.0, max_value=300.0, step=0.1, value=float(today.get("weight_kg", 0.0)) if today else 0.0)
                sleep_hours = st.number_input("Sleep hours", min_value=0.0, max_value=24.0, step=0.5, value=float(today.get("sleep_hours", 0.0)) if today else 0.0)
                resting_hr = st.number_input("Resting HR", min_value=0, max_value=240, step=1, value=int(today.get("resting_hr", 0)) if today else 0)
                steps = st.number_input("Steps", min_value=0, max_value=100000, step=500, value=int(today.get("steps", 0)) if today else 0)
            with top_right:
                sleep_quality = st.select_slider("Sleep quality", options=[0, 1, 2, 3, 4, 5], value=int(today.get("sleep_quality", 3)) if today else 3)
                energy = st.select_slider("Energy", options=[0, 1, 2, 3, 4, 5], value=int(today.get("energy", 3)) if today else 3)
                stress = st.select_slider("Stress", options=[0, 1, 2, 3, 4, 5], value=int(today.get("stress", 0)) if today else 0)
                soreness = st.select_slider("Soreness", options=[0, 1, 2, 3, 4, 5], value=int(today.get("soreness", 0)) if today else 0)
            notes = st.text_area("Notes", value=today.get("notes", "") if today else "", placeholder="Recovery notes, pain points, appetite, mood-body link, cycle, illness, etc.")
            if st.form_submit_button("Save body check-in", use_container_width=True):
                memory.log_body_metrics(
                    weight_kg=weight_kg,
                    sleep_hours=sleep_hours,
                    sleep_quality=sleep_quality,
                    energy=energy,
                    stress=stress,
                    soreness=soreness,
                    resting_hr=resting_hr,
                    steps=steps,
                    notes=notes,
                )
                append_activity("body | detailed check-in saved")
                st.rerun()


def render_detail_focus(memory: Memory, chapter_key: str) -> None:
    """Render the selected right-panel surface inline on the same page."""
    if chapter_key in {"calendar", "trips", "birthdays"}:
        render_calendar_focus(memory, chapter_key)
        return
    if chapter_key == "plans":
        render_plan_focus(memory)
        return
    if chapter_key == "habits":
        render_habits_focus(memory)
        return
    if chapter_key == "water":
        render_water_focus(memory)
        return
    if chapter_key == "body":
        render_body_focus(memory)
        return
    st.markdown(render_memory_focus(memory, chapter_key), unsafe_allow_html=True)


def current_local_time() -> datetime:
    return datetime.now().astimezone()


def current_part_of_day(now: datetime) -> str:
    if 5 <= now.hour < 12:
        return "morning"
    if 12 <= now.hour < 17:
        return "afternoon"
    if 17 <= now.hour < 22:
        return "evening"
    return "night"


def phrase_bucket(now: datetime) -> str:
    return f"{now.strftime('%Y-%m-%d')}-{now.hour:02d}-{now.minute // 15}"


def generate_sidebar_phrase(now: datetime) -> str:
    openings = [
        "Health begins with gentle attention.",
        "A good life grows from small kindnesses to the body.",
        "Energy is built in quiet disciplines.",
        "Strength lasts longer when it is renewed by care.",
        "Well-being lives between motion, rest, and meaning.",
        "A clear day starts with one honest choice.",
    ]
    middles = [
        "Let today favour steady movement.",
        "Protect your peace like a serious habit.",
        "Feed the body with rhythm.",
        "Choose what leaves you lighter by tonight.",
        "Walk toward what strengthens you quietly.",
        "Keep your routines human and durable.",
    ]
    seed = now.timetuple().tm_yday + (now.hour * 4) + (now.minute // 15)
    return f"{openings[seed % len(openings)]} {middles[(seed * 3 + now.weekday()) % len(middles)]}"


def get_rotating_sidebar_phrase(memory: Memory, now: datetime) -> str:
    bucket = phrase_bucket(now)
    state = memory.get_app_state()
    if state.get("sidebar_phrase_bucket") != bucket:
        phrase = generate_sidebar_phrase(now)
        memory.set_app_state_value("sidebar_phrase_text", phrase)
        memory.set_app_state_value("sidebar_phrase_bucket", bucket)
        return phrase
    return str(state.get("sidebar_phrase_text", generate_sidebar_phrase(now)))


def build_daily_checkin(memory: Memory) -> str:
    """Build June's proactive daily opening with a chapter-specific intake question."""
    now = current_local_time()
    part = current_part_of_day(now)
    opening = {"morning": "Good morning.", "afternoon": "Good afternoon.",
               "evening": "Good evening.", "night": "Good evening."}[part]

    notifications = memory.get_upcoming_notifications(limit=3)
    empty_chapters = memory.get_chapters_needing_attention()
    today_metrics = memory.get_today_body_metrics()
    today_workout = memory.get_today_workout()
    today_nutrition = memory.get_nutrition_today()
    water_today = memory.get_water_today()

    lines = [
        f"{opening} It's {now.strftime('%A')}, day {now.timetuple().tm_yday} of the year.",
        "How is your day going, what are your plans, and how are you feeling?",
    ]

    if notifications:
        parts = []
        for item in notifications:
            prefix = "today" if item["days_until"] == 0 else f"in {item['days_until']} days"
            parts.append(f"{item['title']} ({prefix})")
        lines.append("Reminders: " + ", ".join(parts) + ".")

    # Priority: ask about the most critical empty chapter
    chapter_questions = {
        "Habits":        "To start building your profile — what daily habits are you working on? I can track your streaks.",
        "Gym Schedule":  "I don't have your training structure yet. What does your weekly split look like?",
        "Food Schedule": "Tell me about your current nutrition approach and I'll keep it on file.",
        "Calendar":      "Any upcoming events or appointments I should know about?",
        "Birthdays":     "Whose birthdays should I remember? Give me names and dates.",
        "Family":        "Tell me about the key people in your family — names, relationships, and any context.",
        "Goals & Plans": "What are you working toward right now? I'll track your goals and next steps.",
        "Trips":         "Any travel planned in the coming months?",
        "Dating/Love":   "Are you in a relationship or dating? Context helps me support you better.",
        "Body Metrics":  "Want to log a body check-in for today? Weight, sleep, energy, stress, soreness, resting heart rate, and steps all help me reason better.",
        "Workout Sessions": "Did you train today, or should I mark this as a rest day?",
        "Nutrition":     "What have you eaten so far today?",
        "Water":         f"You're at {water_today}/{WATER_GOAL} glasses today. Want to log more water?",
    }

    if empty_chapters:
        question = chapter_questions.get(empty_chapters[0])
        if question:
            lines.append(question)
    else:
        if not today_workout:
            lines.append("Did you train today, or is today a rest day?")
        elif not today_metrics:
            lines.append("Want to log your weight, sleep, or energy for today?")
        elif not today_nutrition:
            lines.append("What have you eaten so far today?")
        elif water_today < WATER_GOAL:
            lines.append(f"You're at {water_today}/{WATER_GOAL} glasses today. Want to log more water?")
        else:
            # Maintenance rotation by day of week
            maintenance = {
                0: "How did training go this week?",
                1: "Any new plans or appointments coming up?",
                2: "How are your habits holding up?",
                3: "Anything on your mind that needs a plan?",
                4: "How are you feeling heading into the weekend?",
                5: "Any goals you want to review or update?",
                6: "How was the week? Anything worth capturing before Monday?",
            }
            lines.append(maintenance.get(now.weekday(), "What is on your mind?"))

    return "\n\n".join(lines)


def render_today_panel(memory: Memory, snapshot: dict[str, int]) -> None:
    """Render the primary daily operating surface in the right rail."""
    model = build_today_panel_model(memory, snapshot)
    kpi_html = "".join(
        '<div class="june-kpi">'
        f'<div class="june-kpi-value">{html.escape(metric.value)}</div>'
        f'<div class="june-kpi-label">{html.escape(metric.label)} · {html.escape(metric.detail)}</div>'
        '</div>'
        for metric in model.kpis
    )
    st.markdown(
        '<div class="june-rail-card june-rail-card-primary">'
        '<div class="june-label">Today</div>'
        f'<div class="june-title">{html.escape(model.title)}</div>'
        f'<div class="june-panel-caption">{html.escape(model.caption)}</div>'
        f'<div class="june-item-meta">{html.escape(model.subheadline)}</div>'
        f'<div class="june-kpi-grid">{kpi_html}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    render_setup_progress_card(model.setup)
    for section in model.sections:
        if not section.items and not section.note:
            continue
        st.markdown(
            '<div class="june-rail-card june-rail-card-quiet">'
            f'<div class="june-label">{html.escape(section.title)}</div>'
            f'<div class="june-item-meta">{html.escape(section.note)}</div>',
            unsafe_allow_html=True,
        )
        render_panel_lines([(item.title, item.copy) for item in section.items])
        st.markdown('</div>', unsafe_allow_html=True)


def render_memory_panel(memory: Memory) -> None:
    """Render memory browsing and chapter access in the right rail."""
    model = build_memory_panel_model(memory, st.session_state.selected_chapter)
    selected_chapter = model.selected_key
    selected_label = model.selected_label
    kicker_right = (
        f'<div class="june-active-tag">Open: {html.escape(selected_label)}</div>'
        if selected_chapter
        else f'<div class="june-compact-copy">{html.escape(model.kicker_copy)}</div>'
    )
    st.markdown(
        '<div class="june-rail-card">'
        '<div class="june-panel-kicker">'
        f'<div class="june-panel-kicker-left"><div class="june-label">Memory</div><div class="june-title">{html.escape(model.title)}</div></div>'
        f'<div class="june-panel-kicker-right">{kicker_right}</div>'
        '</div>'
        f'<div class="june-panel-caption">{html.escape(model.caption)}</div>',
        unsafe_allow_html=True,
    )
    chapter_cols = st.columns(2, gap="small")
    for idx, status in enumerate(model.chapter_cards):
        with chapter_cols[idx % 2]:
            title = status["title"]
            if status["attention"] == "needs_attention":
                subtitle = "needs setup"
            elif status["attention"] == "watch":
                subtitle = status["preview"][:44]
            else:
                subtitle = f'{status["freshness"]} · {status["last_updated"]}'
            button_label = f"{title}\n{subtitle}"
            if st.button(button_label, key=f'ch_{status["key"]}', use_container_width=True):
                sync_selected_chapter(st.session_state, "" if st.session_state.selected_chapter == status["key"] else status["key"])
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if selected_chapter:
        st.markdown('<div class="june-rail-card">', unsafe_allow_html=True)
        render_chapter_detail_focus(
            memory,
            selected_chapter,
            water_goal=WATER_GOAL,
            on_activity=append_activity,
        )
        st.markdown('</div>', unsafe_allow_html=True)


def render_workspace_panel() -> None:
    """Render the current workspace surface."""
    model = build_workspace_panel_model(st.session_state.ui_state)
    st.markdown(
        '<div class="june-rail-card">'
        '<div class="june-label">Workspace</div>'
        f'<div class="june-title">{html.escape(model.focus_title)}</div>'
        f'<div class="june-panel-caption">{html.escape(model.caption)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(render_workspace(st.session_state.ui_state, include_header=False), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_debug_panel(memory: Memory) -> None:
    """Render a quieter debug/trust surface separate from the main workflow."""
    model = build_debug_panel_model(memory, st.session_state.activity_log)
    st.markdown(
        '<div class="june-rail-card">'
        '<div class="june-label">Trust</div>'
        f'<div class="june-title">{html.escape(model.title)}</div>'
        f'<div class="june-panel-caption">{html.escape(model.caption)}</div>',
        unsafe_allow_html=True,
    )
    if model.what_june_saved:
        st.markdown('<div class="june-label">What June saved</div>', unsafe_allow_html=True)
        render_panel_lines([(item.title, item.copy) for item in model.what_june_saved])
        st.markdown('<div class="june-panel-divider"></div>', unsafe_allow_html=True)
    if model.recent_assistant_actions:
        st.markdown('<div class="june-label">Recent assistant actions</div>', unsafe_allow_html=True)
        render_panel_lines([(item.title, item.copy) for item in model.recent_assistant_actions])
        st.markdown('<div class="june-panel-divider"></div>', unsafe_allow_html=True)
    health_rows = [(label, str(count)) for label, count in model.capture_health_counts.items() if count > 0][:6]
    if health_rows:
        st.markdown('<div class="june-label">Light health check</div>', unsafe_allow_html=True)
        render_panel_lines(health_rows)
        st.markdown('<div class="june-panel-divider"></div>', unsafe_allow_html=True)
    with st.expander("Raw diagnostics", expanded=False):
        success = st.session_state.get("tool_stats", {}).get("succeeded", 0)
        total = st.session_state.get("tool_stats", {}).get("requested", 0)
        if total > 0:
            colour = "🟢" if success == total else "🟡"
            st.caption(f"{colour} Tools: {success}/{total} saved this turn")
        if model.recent_events:
            render_panel_lines([
                (
                    f'{event.get("event_type", "event")} · {event.get("name", "")}'.strip(" ·"),
                    f'{event.get("status", "ok")} · {event.get("timestamp", "")[:16]}',
                )
                for event in model.recent_events
            ])
        st.markdown(render_activity(st.session_state.activity_log), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def open_memory_chapter(chapter_key: str) -> None:
    """Open a chapter from post-turn save feedback."""
    sync_selected_chapter(st.session_state, chapter_key)
    sync_rail_view(st.session_state, "memory")
    sync_right_panel_visibility(st.session_state, True)
    append_activity(f"memory open | {chapter_key}")
    st.rerun()


def render_scroll_to_latest() -> None:
    """Force the transcript to open on the latest visible message."""
    components.html(
        """
        <script>
        const tryScroll = () => {
            const parentDoc = window.parent.document;
            const transcript = parentDoc.getElementById("june-transcript");
            const end = parentDoc.getElementById("june-transcript-end");
            if (transcript) {
                transcript.scrollTop = transcript.scrollHeight;
            }
            if (end) {
                end.scrollIntoView({block: "end", behavior: "auto"});
            }
            window.parent.scrollTo({top: parentDoc.body.scrollHeight, behavior: "auto"});
        };
        const bindObserver = () => {
            const parentDoc = window.parent.document;
            const transcript = parentDoc.getElementById("june-transcript");
            if (!transcript || transcript.dataset.juneObserverBound === "1") {
                return;
            }
            transcript.dataset.juneObserverBound = "1";
            const observer = new MutationObserver(() => {
                requestAnimationFrame(tryScroll);
            });
            observer.observe(transcript, {childList: true, subtree: true});
        };
        setTimeout(tryScroll, 0);
        setTimeout(bindObserver, 0);
        setTimeout(tryScroll, 120);
        setTimeout(tryScroll, 260);
        setTimeout(tryScroll, 520);
        </script>
        """,
        height=0,
        width=0,
    )


def handle_stream_chunk(
    mode: str,
    data: Any,
    transcript_placeholder: Any,
    workspace_placeholder: Any,
    activity_placeholder: Any,
) -> None:
    if mode == "custom":
        event = data or {}
        if event.get("event") == "chat_started":
            append_activity(f"route | {event.get('skill')}")
            append_activity(
                "runtime | "
                + f"{event.get('runtime_label', st.session_state.get('current_runtime_label', RUNTIME_CONFIG.label))}"
                + f" | {event.get('runtime_model', st.session_state.get('current_runtime_model', RUNTIME_CONFIG.model))}"
            )
        elif event.get("event") == "tool_calls_requested":
            append_activity("tool request | " + ", ".join(event.get("tools", [])))
        elif event.get("event") == "tool_results":
            summary = event.get("summary", {})
            append_activity(
                "tool results | "
                + f"{summary.get('succeeded', 0)} ok / {summary.get('failed', 0)} failed total"
            )
            for call in event.get("calls", []):
                append_activity(
                    f"tool {call.get('status', 'unknown')} | "
                    f"{call.get('name', '?')} | {call.get('preview', '')}"
                )
        elif event.get("event") == "response_completed":
            append_activity("response | direct answer")
        activity_placeholder.markdown(render_activity(st.session_state.activity_log), unsafe_allow_html=True)
        return

    if mode == "messages":
        message, _metadata = data
        if isinstance(message, AIMessageChunk):
            token_text = extract_text(message.content)
            if token_text:
                st.session_state.live_response += token_text
                transcript_placeholder.markdown(
                    transcript_html(st.session_state.messages, st.session_state.live_response),
                    unsafe_allow_html=True,
                )
                render_scroll_to_latest()
            for chunk in getattr(message, "tool_call_chunks", []) or []:
                name = chunk.get("name")
                if name:
                    append_activity(f"planning | {name}")
        activity_placeholder.markdown(render_activity(st.session_state.activity_log), unsafe_allow_html=True)
        return

    if mode == "updates":
        for node_name, payload in (data or {}).items():
            append_activity(f"node | {node_name}")
            if isinstance(payload, dict):
                if "ui_state" in payload:
                    st.session_state.ui_state = payload["ui_state"]
                    sync_selected_chapter(st.session_state, st.session_state.ui_state.get("selected_chapter", ""))
                    if not st.session_state.selected_chapter:
                        workspace_placeholder.markdown(
                            render_workspace(st.session_state.ui_state, include_header=False),
                            unsafe_allow_html=True,
                        )
                for message in payload.get("messages", []):
                    if isinstance(message, ToolMessage):
                        append_activity(f"tool | {extract_text(message.content)}")
                    elif isinstance(message, AIMessage):
                        for tc in getattr(message, "tool_calls", []) or []:
                            append_activity(f"tool args | {tc.get('name')} {tc.get('args')}")
        activity_placeholder.markdown(render_activity(st.session_state.activity_log), unsafe_allow_html=True)
        return

    if mode == "values" and isinstance(data, dict):
        st.session_state.final_state = data
        if "tool_stats" in data:
            st.session_state.tool_stats = data["tool_stats"]
        if "ui_state" in data:
            st.session_state.ui_state = data["ui_state"]
            sync_selected_chapter(st.session_state, st.session_state.ui_state.get("selected_chapter", ""))
            if not st.session_state.selected_chapter:
                workspace_placeholder.markdown(
                    render_workspace(st.session_state.ui_state, include_header=False),
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Left sidebar — brand + quote + Today panel
# ---------------------------------------------------------------------------

with st.sidebar:
    now = current_local_time()
    st.markdown(
        '<script>setTimeout(function(){ window.parent.location.reload(); }, 900000);</script>',
        unsafe_allow_html=True,
    )

    # Brand
    user_id = st.session_state.get("profile_input", "admin")
    memory_for_sidebar = Memory(user_id)
    stored_runtime_preset = str(
        st.session_state.get(
            "selected_runtime_preset",
            memory_for_sidebar.get_app_state().get("runtime_preset", RUNTIME_CONFIG.preset_key),
        )
    )
    if "selected_runtime_preset" not in st.session_state:
        st.session_state.selected_runtime_preset = stored_runtime_preset
    current_runtime = runtime_for_preset(stored_runtime_preset)
    current_privacy = build_runtime_privacy_status(current_runtime)
    sidebar_phrase = get_rotating_sidebar_phrase(memory_for_sidebar, now)

    st.markdown('<div class="june-brand">June</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="june-copy">{html.escape(sidebar_phrase)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"{now.strftime('%A %d %B')} · {current_part_of_day(now)} · day {now.timetuple().tm_yday}")
    st.caption(f"{current_runtime.label} · {current_runtime.model}")
    st.caption(f"{current_privacy['mode_label']} · {current_privacy['privacy_label']}")

    preset_options = list(runtime_preset_options())
    preset_keys = [preset.key for preset in preset_options]
    selected_index = preset_keys.index(stored_runtime_preset) if stored_runtime_preset in preset_keys else 0
    chosen_preset = st.selectbox(
        "Runtime",
        options=preset_keys,
        index=selected_index,
        format_func=lambda key: runtime_for_preset(key).label,
        key="runtime_preset_picker",
        disabled=st.session_state.get("is_generating", False),
    )
    chosen_runtime = runtime_for_preset(chosen_preset)
    chosen_privacy = build_runtime_privacy_status(chosen_runtime)
    if chosen_preset != stored_runtime_preset:
        if current_runtime.is_local and chosen_runtime.is_api:
            st.markdown(
                '<div class="june-item-meta">'
                + html.escape(chosen_privacy["summary"])
                + "</div>",
                unsafe_allow_html=True,
            )
            confirm_col, cancel_col = st.columns(2, gap="small")
            with confirm_col:
                if st.button("Use API runtime", key="confirm_runtime_switch", use_container_width=True):
                    st.session_state.selected_runtime_preset = chosen_preset
                    memory_for_sidebar.set_app_state_value("runtime_preset", chosen_preset)
                    st.rerun()
            with cancel_col:
                if st.button("Keep local runtime", key="cancel_runtime_switch", use_container_width=True):
                    st.session_state.runtime_preset_picker = stored_runtime_preset
                    st.rerun()
        else:
            st.session_state.selected_runtime_preset = chosen_preset
            memory_for_sidebar.set_app_state_value("runtime_preset", chosen_preset)
            st.rerun()

    # Today divider
    st.markdown('<hr class="june-today-divider">', unsafe_allow_html=True)

    # Habits
    habits_sidebar = memory_for_sidebar.get_habits()
    st.markdown('<div class="june-section-label first">Habits</div>', unsafe_allow_html=True)

    if habits_sidebar:
        done_c = sum(1 for h in habits_sidebar if h.get("done_today"))
        total_c = len(habits_sidebar)
        pct = int((done_c / total_c) * 100) if total_c else 0
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;font-size:9px;'
            f'color:var(--june-muted);margin-bottom:0.2rem;">'
            f'<span>{done_c}/{total_c}</span>'
            f'<span style="color:var(--june-accent);font-family:Syne">{pct}%</span>'
            f'</div>'
            f'<div class="june-progress-track">'
            f'<div class="june-progress-inner" style="width:{pct}%"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        for habit in habits_sidebar:
            r_col, n_col, b_col = st.columns([0.32, 1.4, 0.75], gap="small")
            with r_col:
                st.markdown(habit_ring_svg(habit["done_today"]), unsafe_allow_html=True)
            with n_col:
                streak_html = (
                    f' <span class="june-habit-streak" style="font-size:9px;color:var(--june-accent);font-family:Syne;">'
                    f'{habit["streak"]}d</span>'
                    if habit["streak"] else ""
                )
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.25rem;padding:0.15rem 0;font-size:10px;">'
                    f'{html.escape(habit["name"])}{streak_html}</div>',
                    unsafe_allow_html=True,
                )
            with b_col:
                if not habit["done_today"]:
                    if st.button("done", key=f"sb_habit_{habit['name']}", use_container_width=True):
                        memory_for_sidebar.log_habit_completion(habit["name"])
                        st.rerun()
                else:
                    st.markdown('<span class="june-badge june-badge-done">done</span>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:10px;color:var(--june-muted);">Tell June to add habits.</div>', unsafe_allow_html=True)

    # Water
    water_count = memory_for_sidebar.get_water_today()
    st.markdown('<div class="june-section-label">Water</div>', unsafe_allow_html=True)
    st.markdown(water_dots_html(water_count, WATER_GOAL), unsafe_allow_html=True)
    wc_col, wm_col, wp_col = st.columns([1, 0.45, 0.45], gap="small")
    with wc_col:
        st.markdown(
            f'<div style="font-size:9px;color:var(--june-muted);padding-top:0.3rem;">'
            f'{water_count}/{WATER_GOAL} glasses</div>',
            unsafe_allow_html=True,
        )
    with wm_col:
        if st.button("−", key="sb_water_minus", use_container_width=True):
            if water_count > 0:
                memory_for_sidebar.set_water(water_count - 1)
            st.rerun()
    with wp_col:
        if st.button("+", key="sb_water_plus", use_container_width=True):
            memory_for_sidebar.log_water(1)
            st.rerun()

    # Body metrics
    today_metrics = memory_for_sidebar.get_today_body_metrics()
    sidebar_body_series = detail_recent_body_series(memory_for_sidebar, days=7)
    latest_body = sidebar_body_series[-1] if sidebar_body_series else None
    st.markdown('<div class="june-section-label">Body</div>', unsafe_allow_html=True)

    if today_metrics:
        st.markdown(
            f'<div class="june-body-row"><span class="june-body-key">energy</span>'
            f'{energy_dots_html(today_metrics.get("energy", 0))}</div>'
            f'<div class="june-body-row"><span class="june-body-key">sleep</span>'
            f'<span style="font-size:10px;">{today_metrics.get("sleep_hours", 0)}h</span></div>'
            + (
                f'<div class="june-body-row"><span class="june-body-key">stress</span>'
                f'<span style="font-size:10px;">{today_metrics.get("stress", 0)}/5</span></div>'
                if today_metrics.get("stress")
                else ""
            )
            + (
                f'<div class="june-body-row"><span class="june-body-key">soreness</span>'
                f'<span style="font-size:10px;">{today_metrics.get("soreness", 0)}/5</span></div>'
                if today_metrics.get("soreness")
                else ""
            )
            + (
                f'<div class="june-body-row"><span class="june-body-key">weight</span>'
                f'<span style="font-size:10px;">{today_metrics.get("weight_kg", 0)}kg</span></div>'
                if today_metrics.get("weight_kg") else ""
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:9px;color:var(--june-accent);margin-top:0.15rem;">Today\'s check-in</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div style="font-size:10px;color:var(--june-muted);">No check-in today.</div>', unsafe_allow_html=True)
        if latest_body:
            st.markdown(
                '<div style="font-size:9px;color:var(--june-accent);margin-top:0.15rem;">'
                + f"Last check-in · {html.escape(latest_body.get('date', ''))}"
                + '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="font-size:9px;color:var(--june-muted);margin-top:0.15rem;">'
                + html.escape(detail_body_snapshot_line(latest_body))
                + '</div>',
                unsafe_allow_html=True,
            )

    if sidebar_body_series:
        sleep_now, sleep_delta, _sleep_avg = detail_body_metric_stats(sidebar_body_series, "sleep_hours")
        energy_now, energy_delta, _energy_avg = detail_body_metric_stats(sidebar_body_series, "energy")
        stress_now, stress_delta, _stress_avg = detail_body_metric_stats(sidebar_body_series, "stress")
        trend_bits = []
        if sleep_now is not None:
            delta_txt = "" if sleep_delta is None else f" ({sleep_delta:+.1f}h)"
            trend_bits.append(f"sleep {sleep_now:.1f}h{delta_txt}")
        if energy_now is not None:
            delta_txt = "" if energy_delta is None else f" ({energy_delta:+.0f})"
            trend_bits.append(f"energy {energy_now:.0f}/5{delta_txt}")
        if stress_now is not None:
            delta_txt = "" if stress_delta is None else f" ({stress_delta:+.0f})"
            trend_bits.append(f"stress {stress_now:.0f}/5{delta_txt}")
        if trend_bits:
            st.markdown(
                '<div style="font-size:9px;color:var(--june-muted);margin-top:0.25rem;">7d trend</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="font-size:9px;color:var(--june-muted);margin-top:0.1rem;">'
                + " · ".join(html.escape(bit) for bit in trend_bits)
                + '</div>',
                unsafe_allow_html=True,
            )

    with st.expander("Log body", expanded=False):
        with st.form("sb_body_form", clear_on_submit=True):
            e_in = st.select_slider("Energy", options=[0, 1, 2, 3, 4, 5], value=3)
            sq_in = st.select_slider("Sleep quality", options=[0, 1, 2, 3, 4, 5], value=3)
            stress_in = st.select_slider("Stress", options=[0, 1, 2, 3, 4, 5], value=0)
            soreness_in = st.select_slider("Soreness", options=[0, 1, 2, 3, 4, 5], value=0)
            s_in = st.number_input("Sleep h", min_value=0.0, max_value=24.0, step=0.5, value=0.0)
            w_in = st.number_input("Weight kg", min_value=0.0, max_value=300.0, step=0.1, value=0.0)
            hr_in = st.number_input("Resting HR", min_value=0, max_value=240, step=1, value=0)
            steps_in = st.number_input("Steps", min_value=0, max_value=100000, step=500, value=0)
            notes_in = st.text_area("Notes", value="", placeholder="Anything affecting recovery or performance?")
            if st.form_submit_button("Save", use_container_width=True):
                memory_for_sidebar.log_body_metrics(
                    weight_kg=w_in,
                    sleep_hours=s_in,
                    sleep_quality=sq_in,
                    energy=e_in,
                    stress=stress_in,
                    soreness=soreness_in,
                    resting_hr=hr_in,
                    steps=steps_in,
                    notes=notes_in,
                )
                st.rerun()

    # Workout
    today_workout = memory_for_sidebar.get_today_workout()
    st.markdown('<div class="june-section-label">Workout</div>', unsafe_allow_html=True)
    if today_workout:
        st.markdown(
            f'<span class="june-badge june-badge-done">done</span>'
            f'<div style="font-size:10px;margin-top:0.2rem;">'
            f'{html.escape(today_workout["plan_name"])}'
            f'{" · " + str(today_workout["duration_min"]) + "min" if today_workout.get("duration_min") else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="june-badge june-badge-rest">rest day</span>', unsafe_allow_html=True)

    # Nutrition
    today_nutrition = memory_for_sidebar.get_nutrition_today()
    if today_nutrition:
        total_kcal = sum(e.get("calories_est", 0) for e in today_nutrition)
        total_prot = sum(e.get("protein_est", 0) for e in today_nutrition)
        st.markdown('<div class="june-section-label">Nutrition</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:10px;color:var(--june-muted);">'
            f'{len(today_nutrition)} meals'
            f'{" · ~" + str(total_kcal) + " kcal" if total_kcal else ""}'
            f'{" · ~" + str(total_prot) + "g protein" if total_prot else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Profile + clear at the bottom
    st.markdown('<hr class="june-today-divider">', unsafe_allow_html=True)
    user_id = st.text_input("Profile", value=user_id, key="profile_input")
    if st.button("Clear chat", use_container_width=True):
        reset_session_state(st.session_state, user_id)
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

initialize_session_state(st.session_state, user_id)

if st.session_state.ui_state.get("selected_chapter") and st.session_state.selected_chapter != st.session_state.ui_state.get("selected_chapter"):
    st.session_state.selected_chapter = st.session_state.ui_state.get("selected_chapter", "")
st.session_state.show_right_panel = st.session_state.ui_state.get("show_right_panel", True)

if st.session_state.last_user_id != user_id:
    reset_session_state(st.session_state, user_id)

# ---------------------------------------------------------------------------
# Memory, daily check-in, snapshot
# ---------------------------------------------------------------------------

memory = Memory(user_id)
if "selected_runtime_preset" not in st.session_state:
    st.session_state.selected_runtime_preset = str(memory.get_app_state().get("runtime_preset", RUNTIME_CONFIG.preset_key))
if "turn_summary_message" not in st.session_state:
    st.session_state.turn_summary_message = None
active_runtime = runtime_for_preset(str(st.session_state.selected_runtime_preset))
active_privacy = build_runtime_privacy_status(active_runtime)
st.session_state.current_runtime_label = active_runtime.label
st.session_state.current_runtime_model = active_runtime.model

if not st.session_state.is_generating and memory.should_send_daily_checkin():
    daily_message = build_daily_checkin(memory)
    st.session_state.messages.append(AIMessage(content=daily_message))
    memory.save_message("assistant", daily_message)
    memory.mark_daily_checkin_sent()
    append_activity("daily check-in | sent")

snapshot = memory.get_progress_snapshot()
active_skill = SKILLS.get(st.session_state.active_skill_key, SKILLS[DEFAULT_SKILL])
current_layout = st.session_state.ui_state.get("layout", "split")
show_right_panel = st.session_state.ui_state.get("show_right_panel", True)

# ---------------------------------------------------------------------------
# Main layout: Conversation | Right panel
# ---------------------------------------------------------------------------

layout_widths = layout_column_widths(current_layout, show_right_panel)
columns = st.columns(layout_widths, gap="medium")
chat_col = columns[0]
plan_col = columns[1] if show_right_panel else None

# ── Conversation ──────────────────────────────────────────────────────────

with chat_col:
    st.markdown('<div class="june-surface">', unsafe_allow_html=True)
    st.markdown('<div class="june-label">Conversation</div>', unsafe_allow_html=True)
    render_command_bar(show_right_panel)
    st.markdown(
        '<div class="june-meta-row">'
        f'<div class="june-chip">profile: {html.escape(user_id)}</div>'
        f'<div class="june-chip">route: {html.escape(active_skill.label)}</div>'
        f'<div class="june-chip">agenda: {snapshot["calendar_count"]}</div>'
        f'<div class="june-chip">plans: {snapshot["goal_count"] + snapshot["open_loop_count"]}</div>'
        f'<div class="june-chip">habits: {snapshot["habits_done_today"]}/{snapshot["habit_count"]}</div>'
        f'<div class="june-chip">water: {snapshot["water_today"]}/{WATER_GOAL}</div>'
        f'<div class="june-chip">tools: {st.session_state.tool_stats["succeeded"]}/{st.session_state.tool_stats["requested"]}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    render_attention_strip(memory)
    render_first_run_onboarding(memory)
    render_starter_prompts(memory)
    transcript_placeholder = st.empty()
    transcript_placeholder.markdown(
        transcript_html(
            st.session_state.messages,
            st.session_state.live_response,
            extra_messages=[{"role": "assistant", "content": st.session_state.turn_summary_message}]
            if st.session_state.turn_summary_message
            else None,
        ),
        unsafe_allow_html=True,
    )
    render_turn_save_feedback(st.session_state.turn_summary_message, on_open_chapter=open_memory_chapter)
    render_scroll_to_latest()
    if st.session_state.is_generating:
        st.markdown('<div class="june-writing">June is writing</div>', unsafe_allow_html=True)
    with st.form("june_input_form", clear_on_submit=True):
        prompt = st.text_input(
            "Message June",
            value="",
            placeholder="Tell June about your day, plans, feelings, routines, people, or reminders.",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted and prompt.strip() and not st.session_state.is_generating:
        st.session_state.pending_prompt = prompt.strip()
        st.session_state.active_skill_key = infer_skill_from_text(prompt)
        st.session_state.is_generating = True
        append_activity(f"auto route | {st.session_state.active_skill_key}")
        st.rerun()

# ── Right panel ───────────────────────────────────────────────────────────

if show_right_panel and plan_col is not None:
    with plan_col:
        st.markdown('<div class="june-right-panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="june-rail-card june-rail-card-quiet">'
            '<div class="june-label">Window</div>'
            '<div class="june-title">Single-page workspace</div>'
            '<div class="june-panel-caption">Switch the page posture without losing context. The rail surface below adapts to the task you are doing.</div>'
            f'<div class="june-item-meta">{html.escape(active_privacy["summary"])}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="margin-top:-0.25rem;margin-bottom:0.35rem;">'
            f'<span class="june-runtime-pill">{"local-first" if active_runtime.is_local else "api-assisted"}</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        render_layout_controls()

        workspace_placeholder = st.empty()
        activity_placeholder = st.empty()
        if st.session_state.rail_view == "today":
            render_today_panel(memory, snapshot)
        elif st.session_state.rail_view == "onboarding":
            render_first_run_onboarding(memory)
        elif st.session_state.rail_view == "memory":
            render_memory_panel(memory)
        elif st.session_state.rail_view == "workspace":
            render_workspace_panel()
        else:
            render_debug_panel(memory)

        st.markdown("</div>", unsafe_allow_html=True)  # close june-right-panel
else:
    workspace_placeholder = st.empty()
    activity_placeholder = st.empty()

# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

if st.session_state.is_generating and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    memory.save_message("user", prompt)
    st.session_state.pending_prompt = ""
    st.session_state.live_response = ""
    st.session_state.final_state = None
    st.session_state.turn_summary_message = None
    append_activity(f"user | {prompt}")
    transcript_placeholder.markdown(
        transcript_html(st.session_state.messages, st.session_state.live_response),
        unsafe_allow_html=True,
    )
    render_turn_save_feedback(None, on_open_chapter=open_memory_chapter)
    render_scroll_to_latest()

    # activity_placeholder may be inside a collapsed expander; guard against it
    try:
        activity_placeholder.markdown(render_activity(st.session_state.activity_log), unsafe_allow_html=True)
    except Exception:
        pass

    try:
        active_agent = get_compiled_agent(
            active_runtime.preset_key,
            active_runtime.provider,
            active_runtime.model,
            active_runtime.base_url,
            active_runtime.temperature,
            active_runtime.max_tokens,
            active_runtime.tool_strategy,
        )
        for mode, data in active_agent.stream(
            {
                "messages": st.session_state.messages,
                "user_id":  user_id,
                "skill":    st.session_state.active_skill_key,
                "ui_state": st.session_state.ui_state,
                "tool_stats": st.session_state.tool_stats,
            },
            stream_mode=["messages", "updates", "custom", "values"],
        ):
            handle_stream_chunk(
                mode, data,
                transcript_placeholder,
                workspace_placeholder,
                activity_placeholder,
            )
    except Exception as exc:
        st.session_state.is_generating = False
        st.error(f"June ran into an issue: {exc}")
        st.stop()

    result = st.session_state.final_state
    if result:
        response = next(
            (m for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content),
            None,
        )
        if response:
            final_text = extract_text(response.content)
            st.session_state.messages = result["messages"]
            st.session_state.live_response = ""
            st.session_state.turn_summary_message = build_turn_save_summary(
                list(st.session_state.tool_stats.get("last_calls", []))
            )
            transcript_placeholder.markdown(
                transcript_html(
                    st.session_state.messages,
                    extra_messages=[{"role": "assistant", "content": st.session_state.turn_summary_message}]
                    if st.session_state.turn_summary_message
                    else None,
                ),
                unsafe_allow_html=True,
            )
            render_turn_save_feedback(st.session_state.turn_summary_message, on_open_chapter=open_memory_chapter)
            render_scroll_to_latest()
            memory.save_message("assistant", final_text)

    st.session_state.is_generating = False
    st.rerun()
