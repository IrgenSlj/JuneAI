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
from agent.ollama_manager import (
    cleanup_progress_file,
    is_model_available,
    is_ollama_running,
    model_size_label,
    ollama_cli_available,
    read_pull_progress,
    start_pull,
    start_pull_with_progress,
)
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
    chapter_ring_svg,
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

st.set_page_config(
    page_title="June",
    page_icon="/app/static/favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=Inter:wght@400;500&display=swap');

    :root {
        --j-bg:          #F5F4F1;
        --j-surface:     #FFFFFF;
        --j-text:        #1A1815;
        --j-muted:       #6B6259;
        --j-line:        rgba(26, 24, 21, 0.08);
        --j-accent:      #0F5F4A;
        --j-accent-soft: rgba(15, 95, 74, 0.09);
        --j-accent-mist: rgba(15, 95, 74, 0.04);
        --j-user-bg:     rgba(15, 95, 74, 0.07);
        --j-radius:      16px;
        --j-shadow:      0 4px 24px rgba(26, 24, 21, 0.06);
        --j-shadow-lg:   0 12px 40px rgba(26, 24, 21, 0.09);
        --j-sidebar-w:   260px;

        /* Legacy aliases so existing helper HTML still works */
        --june-text:        var(--j-text);
        --june-muted:       var(--j-muted);
        --june-line:        var(--j-line);
        --june-accent:      var(--j-accent);
        --june-accent-soft: var(--j-accent-soft);
        --june-accent-mist: var(--j-accent-mist);
        --june-user:        var(--j-user-bg);
        --june-panel:       var(--j-surface);
        --june-radius:      var(--j-radius);
        --june-shadow:      var(--j-shadow);
        --june-shadow-lg:   var(--j-shadow-lg);
    }

    /* ── Dark mode overrides ──────────────────────────────── */
    body.june-dark {
        --j-bg:          #111110;
        --j-surface:     #1A1918;
        --j-text:        #F0EDE8;
        --j-muted:       #8A8178;
        --j-line:        rgba(240, 237, 232, 0.08);
        --j-accent:      #2ECC9A;
        --j-accent-soft: rgba(46, 204, 154, 0.12);
        --j-accent-mist: rgba(46, 204, 154, 0.05);
        --j-user-bg:     rgba(46, 204, 154, 0.09);
    }
    body.june-dark [data-testid="stAppViewContainer"],
    body.june-dark [data-testid="stApp"],
    body.june-dark .main, body.june-dark .block-container {
        background: var(--j-bg) !important;
    }

    /* ── Nav radio (horizontal) styled as nav links ──────── */
    [data-testid="stRadio"] {
        width: 100%;
    }
    [data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        gap: 0 !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        height: 44px !important;
    }
    [data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        padding: 0 0.9rem !important;
        height: 44px !important;
        cursor: pointer !important;
        color: var(--j-muted) !important;
        font-size: 13px !important;
        font-family: "Inter", monospace !important;
        font-weight: 400 !important;
        border-bottom: 2px solid transparent !important;
        white-space: nowrap !important;
        background: transparent !important;
        border-radius: 0 !important;
        margin: 0 !important;
        gap: 0 !important;
        transition: color 0.12s, border-color 0.12s;
    }
    [data-testid="stRadio"] label:hover {
        color: var(--j-text) !important;
        border-bottom-color: var(--j-line) !important;
    }
    [data-testid="stRadio"] label:has(input:checked) {
        color: var(--j-accent) !important;
        font-weight: 500 !important;
        border-bottom: 2px solid var(--j-accent) !important;
    }
    /* Hide radio circles and extra markup */
    [data-testid="stRadio"] input[type="radio"] {
        position: absolute; opacity: 0; width: 0; height: 0; pointer-events: none;
    }
    [data-testid="stRadio"] label > span:first-child { display: none !important; }
    [data-testid="stRadio"] > label { display: none !important; }

    /* Right nav action buttons (dark mode, settings) */
    .june-nav-action .stButton > button {
        background: transparent !important;
        border: 1px solid var(--j-line) !important;
        color: var(--j-muted) !important;
        font-size: 13px !important;
        padding: 0 0.6rem !important;
        height: 32px !important;
        min-height: 32px !important;
        box-shadow: none !important;
        border-radius: 8px !important;
        transition: color 0.12s, background 0.12s;
    }
    .june-nav-action .stButton > button:hover {
        color: var(--j-accent) !important;
        background: var(--j-accent-soft) !important;
        border-color: rgba(15,95,74,0.25) !important;
        box-shadow: none !important; transform: none !important;
    }

    /* ── Hide chat toggle button ─────────────────────────── */
    .june-chat-toggle .stButton > button {
        background: transparent !important;
        border: none !important;
        color: var(--j-muted) !important;
        font-size: 11px !important;
        padding: 0 0 0.25rem 0 !important;
        min-height: 1.5rem !important;
        box-shadow: none !important;
        border-radius: 0 !important;
    }
    .june-chat-toggle .stButton > button:hover {
        color: var(--j-accent) !important;
        background: transparent !important;
        box-shadow: none !important; transform: none !important;
    }

    /* ── Global reset ──────────────────────────────────── */
    html, body, [class*="css"],
    [data-testid="stAppViewContainer"],
    [data-testid="stMarkdownContainer"] {
        font-family: "Inter", "IBM Plex Mono", monospace;
        color: var(--j-text);
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stApp"],
    .main, .block-container {
        background: var(--j-bg) !important;
    }

    /* Prevent page-level scrolling — layout is self-contained */
    [data-testid="stAppViewContainer"] {
        overflow: hidden !important;
        height: 100vh !important;
    }
    [data-testid="stApp"] {
        overflow: hidden !important;
        height: 100vh !important;
    }

    .block-container {
        max-width: 1400px !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0 !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        overflow: hidden !important;
        height: 100vh !important;
    }

    /* Chat column (first in horizontal block): fixed height, internal scroll */
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
        height: calc(100vh - 90px);
        overflow: hidden;
        display: flex;
        flex-direction: column;
    }
    /* Input form pinned to bottom of chat column */
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child [data-testid="stForm"] {
        flex-shrink: 0;
        position: sticky;
        bottom: 0;
        background: var(--j-bg);
        padding-top: 0.5rem;
        border-top: 1px solid var(--j-line);
        z-index: 10;
    }
    /* Transcript area grows and scrolls */
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child .june-transcript {
        flex: 1;
        max-height: none !important;
        overflow-y: auto;
    }

    /* Right panel column: fixed height, internal scroll */
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
        height: calc(100vh - 90px);
        overflow-y: auto;
        scrollbar-width: thin;
        scrollbar-color: rgba(26,24,21,0.1) transparent;
    }

    /* ── Hide native Streamlit chrome (sidebar + toolbar) ── */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarNavButton"],
    [data-testid="collapsedControl"] { display: none !important; }

    /* Hide the fixed Streamlit header/toolbar so our header shows fully */
    header[data-testid="stHeader"],
    [data-testid="stHeader"] { display: none !important; }

    /* ── Top bar ───────────────────────────────────────────── */
    .june-topbar-wrap {
        border-bottom: 1px solid var(--j-line);
        margin-bottom: 0.85rem;
        padding-bottom: 0.55rem;
        animation: slideDown 0.18s ease both;
    }
    .june-topbar-logo {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.25rem 0;
    }
    .june-logo-icon {
        flex-shrink: 0;
    }
    .june-logo-text {
        font-family: "Syne", sans-serif;
        font-weight: 700;
        font-size: 1.15rem;
        letter-spacing: -0.04em;
        color: var(--j-text);
        line-height: 1;
        animation: breathe 4s ease-in-out infinite;
    }
    .june-topbar-quote {
        font-size: 10px;
        color: var(--j-muted);
        font-style: italic;
        line-height: 1.55;
        padding: 0.3rem 0;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .june-topbar-datetime {
        text-align: center;
        padding: 0.3rem 0;
        font-size: 12px;
        font-weight: 600;
        color: var(--j-text);
        letter-spacing: 0.01em;
    }
    .june-topbar-time {
        display: block;
        font-size: 9px;
        font-weight: 400;
        color: var(--j-muted);
        margin-top: 1px;
    }

    /* ── Panel card (shared by both left and right panels) ─── */
    .june-panel-card {
        background: var(--j-surface);
        border: 1px solid var(--j-line);
        border-radius: var(--j-radius);
        padding: 1rem 0.85rem;
        min-height: 60vh;
        animation: panelIn 0.2s ease both;
    }
    .june-panel-label {
        font-family: "Syne", sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--j-accent);
        margin-bottom: 0.6rem;
    }
    .june-panel-section {
        margin-top: 0.75rem;
        padding-top: 0.65rem;
        border-top: 1px solid var(--j-line);
    }

    /* ── Form inputs ───────────────────────────────────────── */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: var(--j-surface);
        border: 1px solid var(--j-line);
        border-radius: 12px;
        color: var(--j-text);
        font-family: "Inter", monospace;
        font-size: 14px !important;
        line-height: 1.55;
    }
    [data-testid="stTextArea"] textarea {
        min-height: 56px !important;
        resize: none;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: rgba(15,95,74,0.35) !important;
        box-shadow: 0 0 0 3px rgba(15,95,74,0.08) !important;
        outline: none;
    }

    /* ── Buttons ───────────────────────────────────────────── */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid var(--j-line);
        background: var(--j-surface);
        color: var(--j-text);
        min-height: 2.1rem;
        font-family: "Inter", monospace;
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.01em;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: rgba(15, 95, 74, 0.28);
        color: var(--j-accent);
        background: var(--j-accent-soft);
        box-shadow: 0 2px 8px rgba(15, 95, 74, 0.07);
        transform: translateY(-1px);
    }
    .stButton > button:active { transform: translateY(0); }

    /* Send button — primary CTA */
    .june-send-btn .stButton > button {
        background: var(--j-accent) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        min-height: 2.5rem !important;
        font-size: 13px !important;
        letter-spacing: 0.02em;
    }
    .june-send-btn .stButton > button:hover {
        background: #0d5242 !important;
        color: #ffffff !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(15, 95, 74, 0.22) !important;
    }

    .june-chapter-grid .stButton > button {
        min-height: 4.2rem;
        border-radius: 12px;
        text-align: left;
        padding: 0.65rem 0.75rem;
        font-size: 11px;
    }

    /* ── Animations ────────────────────────────────────────── */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0);   }
    }
    @keyframes waterPop {
        0%   { transform: scale(0.6); opacity: 0.3; }
        60%  { transform: scale(1.3); }
        100% { transform: scale(1);   opacity: 1;   }
    }
    @keyframes typingBounce {
        0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
        40%            { transform: translateY(-5px); opacity: 1; }
    }
    @keyframes breathe {
        0%, 100% { opacity: 1;    }
        50%       { opacity: 0.6; }
    }
    @keyframes badgePop {
        0%   { transform: scale(0.85); opacity: 0; }
        60%  { transform: scale(1.05); }
        100% { transform: scale(1);    opacity: 1; }
    }
    @keyframes panelIn {
        from { opacity: 0; transform: translateX(10px); }
        to   { opacity: 1; transform: translateX(0);    }
    }
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-6px); }
        to   { opacity: 1; transform: translateY(0);    }
    }

    /* ── Page header bar (legacy, kept for compatibility) ─── */
    .june-header-bar { display: none; }
    .june-header-pill {
        border: 1px solid var(--j-line);
        border-radius: 999px;
        padding: 0.2rem 0.6rem;
        font-size: 10px;
        color: var(--j-muted);
        background: var(--j-surface);
        letter-spacing: 0.03em;
    }
    .june-header-pill.accent {
        border-color: rgba(15,95,74,0.2);
        background: var(--j-accent-soft);
        color: var(--j-accent);
    }

    /* ── Surface cards ─────────────────────────────────────── */
    .june-surface {
        background: var(--j-surface);
        border: 1px solid var(--j-line);
        border-radius: var(--j-radius);
        box-shadow: var(--j-shadow);
        padding: 1rem;
        margin-bottom: 0.75rem;
        animation: fadeUp 0.2s ease both;
    }

    /* ── Brand (sidebar) ────────────────────────────────────── */
    .june-brand {
        font-family: "Syne", sans-serif;
        letter-spacing: -0.04em;
        font-size: 1.6rem;
        line-height: 1;
        margin: 0 0 0.25rem 0;
        animation: breathe 4s ease-in-out infinite;
    }

    .june-copy {
        color: var(--j-muted);
        font-size: 10px;
        line-height: 1.55;
        margin-bottom: 0.4rem;
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
        max-height: 64vh;
        overflow-y: auto;
        padding-right: 0.25rem;
        padding-bottom: 0.5rem;
        scrollbar-width: thin;
        scrollbar-color: rgba(26,24,21,0.12) transparent;
    }
    .june-transcript::-webkit-scrollbar { width: 4px; }
    .june-transcript::-webkit-scrollbar-thumb {
        background: rgba(26,24,21,0.12);
        border-radius: 999px;
    }

    .june-message {
        border-radius: 16px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        font-size: 14px;
        line-height: 1.65;
        animation: fadeUp 0.18s ease both;
        max-width: 86%;
    }

    .june-message-user {
        background: var(--j-user-bg);
        border: 1px solid rgba(15,95,74,0.12);
        margin-left: auto;
        margin-right: 0;
        border-bottom-right-radius: 4px;
        color: var(--j-text);
    }
    .june-message-assistant {
        background: var(--j-surface);
        border: 1px solid var(--j-line);
        margin-left: 0;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        box-shadow: 0 1px 4px rgba(26,24,21,0.03);
    }

    /* Three-dot typing indicator */
    .june-typing {
        display: flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.7rem 0.9rem;
        background: var(--j-surface);
        border: 1px solid var(--j-line);
        border-radius: 14px;
        border-bottom-left-radius: 4px;
        width: fit-content;
        margin-bottom: 0.6rem;
        box-shadow: 0 1px 6px rgba(26,24,21,0.04);
    }
    .june-typing-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: var(--j-accent);
        animation: typingBounce 1.3s ease infinite;
    }
    .june-typing-dot:nth-child(2) { animation-delay: 0.18s; }
    .june-typing-dot:nth-child(3) { animation-delay: 0.36s; }
    .june-typing-label {
        font-size: 10px;
        color: var(--j-muted);
        margin-left: 0.2rem;
    }

    /* ── Lists ─────────────────────────────────────────────── */
    .june-list { display: grid; gap: 0.45rem; }

    .june-item {
        border: 1px solid var(--j-line);
        border-radius: 10px;
        padding: 0.5rem 0.65rem;
        background: var(--j-surface);
        transition: all 0.15s ease;
    }
    .june-item:hover {
        border-color: rgba(15, 95, 74, 0.2);
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(26, 24, 21, 0.05);
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
        gap: 0.6rem;
        animation: panelIn 0.25s ease both;
        max-height: calc(100vh - 3rem);
        overflow-y: auto;
        position: sticky;
        top: 0.5rem;
        scrollbar-width: none;
    }
    .june-right-panel::-webkit-scrollbar { display: none; }

    .june-panel-card {
        background: var(--j-surface);
        border: 1px solid var(--j-line);
        border-radius: 16px;
        box-shadow: var(--j-shadow);
        padding: 0.9rem;
    }

    .june-panel-card-quiet {
        box-shadow: none;
        background: rgba(255,255,255,0.8);
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
        border: 1px solid rgba(15, 95, 74, 0.14);
        background: linear-gradient(135deg, rgba(15, 95, 74, 0.06), rgba(15, 95, 74, 0.01));
        border-radius: 12px;
        padding: 0.6rem 0.75rem;
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
        background: var(--j-surface);
        border: 1px solid var(--j-line);
        border-radius: 14px;
        box-shadow: var(--j-shadow);
        padding: 0.85rem;
        transition: box-shadow 0.2s ease;
    }
    .june-rail-card:hover { box-shadow: var(--j-shadow-lg); }

    .june-rail-card-primary {
        border-color: rgba(15, 95, 74, 0.15);
        box-shadow: 0 8px 28px rgba(15, 95, 74, 0.09);
        background: linear-gradient(160deg, #fff 60%, rgba(15,95,74,0.03) 100%);
    }

    .june-rail-card-quiet {
        box-shadow: none;
        background: rgba(255,255,255,0.75);
    }

    .june-kpi-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.4rem;
        margin-top: 0.5rem;
    }

    .june-kpi {
        border: 1px solid var(--j-line);
        border-radius: 12px;
        padding: 0.55rem 0.65rem;
        background: var(--j-bg);
    }

    .june-kpi-value {
        font-family: "Syne", sans-serif;
        font-size: 1.05rem;
        line-height: 1;
        margin-bottom: 0.15rem;
        letter-spacing: -0.02em;
    }

    .june-kpi-label {
        color: var(--j-muted);
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
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

    /* Expanders */
    details summary {
        font-size: 11px;
        color: var(--j-muted);
        cursor: pointer;
    }
    details summary:hover { color: var(--j-accent); }

    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {
        border-radius: 10px !important;
        font-size: 12px !important;
        border-color: var(--j-line) !important;
    }

    /* Captions */
    [data-testid="stCaptionContainer"] {
        font-size: 10px !important;
        color: var(--j-muted) !important;
    }

    /* Compact top padding now that Streamlit toolbar is hidden */
    .main .block-container { padding-top: 0.75rem !important; }

    /* Smooth scrolling page-wide */
    html { scroll-behavior: smooth; }

    /* Section dividers */
    .june-today-divider {
        border: none;
        border-top: 1px solid var(--j-line);
        margin: 0.65rem 0;
    }

    /* Progress bar */
    .june-progress-track {
        height: 3px;
        background: var(--j-line);
        border-radius: 999px;
        overflow: hidden;
        margin: 0.2rem 0 0.5rem 0;
    }
    .june-progress-inner {
        height: 100%;
        background: var(--j-accent);
        border-radius: 999px;
        transition: width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
    }

    /* Badges */
    .june-badge {
        display: inline-block;
        padding: 0.12rem 0.4rem;
        border-radius: 999px;
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        animation: badgePop 0.25s ease both;
    }
    .june-badge-done {
        background: var(--j-accent-soft);
        color: var(--j-accent);
        border: 1px solid rgba(15,95,74,0.2);
    }
    .june-badge-rest {
        background: rgba(26,24,21,0.04);
        color: var(--j-muted);
        border: 1px solid var(--j-line);
    }

    /* Runtime pill */
    .june-runtime-pill {
        border: 1px solid rgba(15,95,74,0.18);
        background: var(--j-accent-soft);
        color: var(--j-accent);
        border-radius: 999px;
        padding: 0.15rem 0.45rem;
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        display: inline-block;
    }

    /* Chips */
    .june-chip {
        border: 1px solid var(--j-line);
        border-radius: 999px;
        padding: 0.18rem 0.45rem;
        font-size: 9px;
        color: var(--j-muted);
        background: var(--j-surface);
    }

    /* Water dots */
    .june-water-track {
        display: flex;
        gap: 4px;
        align-items: center;
        flex-wrap: wrap;
        padding: 0.15rem 0;
    }
    .june-water-dot {
        width: 9px; height: 9px;
        border-radius: 50%;
        display: inline-block;
        transition: background 0.2s ease, transform 0.15s ease;
    }
    .june-water-dot.filled {
        background: var(--j-accent);
        animation: waterPop 0.3s ease both;
    }
    .june-water-dot.empty {
        background: transparent;
        border: 1px solid rgba(26,24,21,0.15);
    }

    /* Energy / metric dots */
    .june-metric-dots { display: inline-flex; gap: 2px; align-items: center; }
    .june-metric-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        display: inline-block;
    }
    .june-metric-dot.active   { background: var(--j-accent); }
    .june-metric-dot.inactive { background: rgba(26,24,21,0.1); }

    /* Body row */
    .june-body-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.15rem 0;
        font-size: 11px;
    }
    .june-body-key { color: var(--j-muted); font-size: 10px; }

    /* Focus hero */
    .june-focus-hero {
        border: 1px solid rgba(15,95,74,0.12);
        background: linear-gradient(135deg, rgba(15,95,74,0.06), rgba(15,95,74,0.01));
        border-radius: 14px;
        padding: 0.7rem 0.8rem;
        margin-bottom: 0.6rem;
    }
    .june-focus-copy { color: var(--j-muted); font-size: 10px; line-height: 1.5; margin-top: 0.15rem; }

    /* Stat grid */
    .june-stat-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.4rem;
    }
    .june-stat-card {
        border: 1px solid var(--j-line);
        border-radius: 10px;
        padding: 0.5rem;
        background: var(--j-bg);
        transition: all 0.15s ease;
    }
    .june-stat-card:hover {
        border-color: rgba(15,95,74,0.2);
        box-shadow: 0 2px 8px rgba(15,95,74,0.05);
    }
    .june-stat-label { color: var(--j-muted); font-size: 9px; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.18rem; }
    .june-stat-value { font-family: "Syne", sans-serif; font-size: 1rem; line-height: 1; letter-spacing: -0.02em; }

    /* Mini grid */
    .june-mini-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.4rem;
        margin: 0.5rem 0 0.6rem 0;
    }
    .june-mini-card {
        border: 1px solid var(--j-line);
        border-radius: 10px;
        padding: 0.5rem 0.55rem;
        background: var(--j-accent-mist);
    }
    .june-mini-label { color: var(--j-muted); font-size: 9px; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.18rem; }
    .june-mini-value { font-family: "Syne", sans-serif; font-size: 0.95rem; line-height: 1; }

    /* Starter section */
    .june-starter-copy { color: var(--j-muted); font-size: 10px; line-height: 1.5; }

    /* ── Tool activity badge ─────────────────────────────────── */
    @keyframes toolPop {
        0%   { transform: scale(0.85); opacity: 0; }
        55%  { transform: scale(1.05); }
        100% { transform: scale(1);    opacity: 1; }
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    .june-tool-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.28rem 0.6rem;
        background: var(--j-accent-soft);
        border: 1px solid rgba(15,95,74,0.14);
        border-radius: 999px;
        font-size: 10px;
        color: var(--j-accent);
        animation: toolPop 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) both;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
    }
    .june-tool-spinner {
        width: 9px; height: 9px;
        border: 1.5px solid rgba(15,95,74,0.25);
        border-top-color: var(--j-accent);
        border-radius: 50%;
        animation: spin 0.7s linear infinite;
        flex-shrink: 0;
    }
    .june-tool-activity {
        display: flex;
        flex-wrap: wrap;
        gap: 0.3rem;
        margin-bottom: 0.5rem;
        animation: fadeUp 0.18s ease both;
    }

    /* ── Streaming cursor ────────────────────────────────────── */
    @keyframes streamBlink {
        0%, 49%  { opacity: 1; }
        50%, 100% { opacity: 0; }
    }
    .june-stream-cursor {
        display: inline-block;
        width: 2px;
        height: 1em;
        background: var(--j-accent);
        margin-left: 2px;
        vertical-align: text-bottom;
        animation: streamBlink 0.65s ease-in-out infinite;
    }

    /* ── Keyboard hint ───────────────────────────────────────── */
    .june-input-hint {
        font-size: 9px;
        color: var(--j-muted);
        text-align: right;
        margin-top: 0.2rem;
        letter-spacing: 0.02em;
    }

    /* ── Sticky chat input — pinned to bottom on all screen sizes ── */
    .june-input-wrap {
        position: sticky;
        bottom: 0;
        background: var(--j-bg);
        padding-top: 0.5rem;
        padding-bottom: 0.25rem;
        z-index: 20;
        border-top: 1px solid var(--j-line);
        margin-top: 0.5rem;
    }

    /* ── Header action buttons — styled as plain text, matching date typography ── */
    .june-hdr-btn .stButton > button {
        background: transparent !important;
        border: none !important;
        border-radius: 6px !important;
        color: var(--j-muted) !important;
        font-family: "Inter", monospace !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em !important;
        min-height: 38px !important;
        padding: 0 0.4rem !important;
        width: 100%;
        transition: color 0.12s ease, background 0.12s ease;
        box-shadow: none !important;
    }
    .june-hdr-btn .stButton > button:hover {
        color: var(--j-accent) !important;
        background: var(--j-accent-soft) !important;
        border: none !important;
        box-shadow: none !important;
        transform: none !important;
    }
    .june-hdr-btn-active .stButton > button {
        color: var(--j-accent) !important;
        background: var(--j-accent-soft) !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* ── Streamlit tab active indicator ─────────────────────── */
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--j-accent) !important;
    }
    [data-testid="stTabs"] [role="tablist"] {
        border-bottom: 1px solid var(--j-line) !important;
    }
    [data-testid="stTabs"] button[role="tab"] {
        font-size: 11px !important;
        font-family: "Inter", monospace !important;
        letter-spacing: 0.01em !important;
        color: var(--j-muted) !important;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--j-accent) !important;
        border-bottom-color: var(--j-accent) !important;
    }

    /* ── Responsive ──────────────────────────────────────────── */
    @media (max-width: 960px) {
        .june-topbar-quote { display: none; }
    }
    @media (max-width: 768px) {
        .block-container { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
        .june-message { font-size: 13px !important; max-width: 98% !important; }
        .june-message-user { max-width: 86% !important; }
        .june-message-assistant { max-width: 98% !important; }
        .stForm { position: sticky; bottom: 0; background: var(--j-bg); z-index: 10; padding-top: 0.5rem; }
        .june-transcript { max-height: 55vh; }
        .june-panel-card { display: none; }
    }
    @media (max-width: 480px) {
        .june-message { font-size: 13px !important; }
        .june-kpi-grid { grid-template-columns: repeat(2, 1fr) !important; }
    }

    /* ── Accessibility ───────────────────────────────────────── */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
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

# ── Auto-pull: ensure the default model is present before the main UI loads ─
_startup_base_url = RUNTIME_CONFIG.base_url
_startup_model = RUNTIME_CONFIG.model
if _startup_base_url and not is_model_available(_startup_model, _startup_base_url):
    import time as _time

    _size_label = model_size_label(_startup_model)
    _size_str = f" · {_size_label}" if _size_label else ""
    _has_cli = ollama_cli_available()

    # Minimal header so the brand is visible during startup download
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0.5rem;padding:0.5rem 0 0.75rem 0;'
        'border-bottom:1px solid rgba(26,24,21,0.08);margin-bottom:1.5rem;">'
        '<span style="font-family:Syne,sans-serif;font-weight:700;font-size:1.1rem;'
        'letter-spacing:-0.04em;color:var(--j-text);">June</span>'
        '<span style="width:5px;height:5px;border-radius:50%;background:#0F5F4A;'
        'display:inline-block;margin-left:4px;vertical-align:middle;margin-bottom:2px;"></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Guard: if Ollama service itself isn't running, show a clear error.
    if not is_ollama_running(_startup_base_url):
        _, _err_col, _ = st.columns([1, 2, 1])
        with _err_col:
            st.error("Ollama is not running.")
            st.markdown(
                "Start the Ollama service, then refresh this page."
            )
            st.code("ollama serve")
        st.stop()

    # Kick off `ollama pull` as a fire-and-forget OS process (once per session).
    # Using the CLI subprocess keeps Streamlit's main thread completely free —
    # we just poll is_model_available() every 2 s via st.rerun().
    if not st.session_state.get("_su_pull_started"):
        start_pull(_startup_model)
        st.session_state["_su_pull_started"] = True
        st.session_state["_su_pull_start"] = _time.time()

    _su_elapsed = int(_time.time() - st.session_state.get("_su_pull_start", _time.time()))
    _su_elapsed_str = (
        f"{_su_elapsed}s" if _su_elapsed < 60
        else f"{_su_elapsed // 60}m {_su_elapsed % 60}s"
    )

    _, _su_col, _ = st.columns([1, 2, 1])
    with _su_col:
        st.markdown(
            f'<div style="text-align:center;padding:3rem 0 1.5rem;">'
            f'<div style="font-family:Syne,sans-serif;font-size:1.35rem;font-weight:600;'
            f'color:var(--j-text);margin-bottom:0.35rem;">Downloading model</div>'
            f'<div style="font-size:13px;color:var(--j-muted);margin-bottom:1.5rem;">'
            f'<code>{_startup_model}</code>{_size_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.progress(0, text=f"Downloading… · {_su_elapsed_str} elapsed")
        st.markdown(
            '<div style="font-size:11px;color:var(--j-muted);text-align:center;margin-top:0.4rem;">'
            f'Downloading {html.escape(_startup_model)}'
            f'{(" (" + html.escape(_size_label) + ")") if _size_label else ""} — '
            'do not close this window.'
            '</div>',
            unsafe_allow_html=True,
        )
        if not _has_cli:
            st.info("Ollama CLI not found on PATH. Run this in your terminal to download manually:")
            st.code(f"ollama pull {_startup_model}")

    # Poll is_model_available every 2 s — ticks from reruns, not from a blocking loop.
    if is_model_available(_startup_model, _startup_base_url):
        for _k in ("_su_pull_started", "_su_pull_start"):
            st.session_state.pop(_k, None)
        st.rerun()
    else:
        _time.sleep(2)
        st.rerun()


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
    prompt_style: str,
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
        prompt_style=prompt_style,
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


def is_first_run(memory: Memory) -> bool:
    """Return True only once per user — when no messages and no data exist yet."""
    if memory.get_app_state().get("_welcomed"):
        return False
    snapshot = memory.get_progress_snapshot()
    has_any_data = any(
        snapshot.get(k, 0)
        for k in ("goal_count", "calendar_count", "habit_count", "gym_plan_count",
                  "food_program_count", "relationship_count", "workout_session_count")
    )
    if has_any_data:
        # Returning user whose welcome flag was never set — backfill and skip
        memory.set_app_state_value("_welcomed", True)
        return False
    return True


def build_welcome_message(memory: Memory) -> str:
    """Build the one-time welcome message for a brand-new user."""
    memory.set_app_state_value("_welcomed", True)
    empty_chapters = memory.get_chapters_needing_attention()
    priority_order = [
        "Habits",
        "Goals & Plans",
        "Gym Schedule",
        "Calendar",
        "Food Schedule",
        "Body Metrics",
        "Family",
        "Birthdays",
        "Dating/Love",
        "Trips",
    ]
    first_question_map = {
        "Habits":        "What daily habits are you trying to build or protect? I'll track your streaks.",
        "Goals & Plans": "What are you working toward right now — one goal or priority is enough to start.",
        "Gym Schedule":  "What does your training week look like? Push, pull, legs, or something else?",
        "Calendar":      "Any upcoming events, deadlines, or trips I should know about?",
        "Food Schedule": "How do you approach food — any structure, goals, or dietary preferences worth noting?",
        "Body Metrics":  "Want to do a quick body check-in? Weight, sleep, energy, stress, soreness — whatever you have.",
        "Family":        "Tell me a bit about your family — names and relationships help me give better context.",
        "Birthdays":     "Whose birthdays or anniversaries should I remember?",
        "Dating/Love":   "Are you in a relationship or dating? Context helps me support you better.",
        "Trips":         "Any travel planned in the coming months?",
    }
    first_question = "What is one thing you are working on right now — a goal, a plan, or a priority?"
    for chapter in priority_order:
        if chapter in empty_chapters:
            first_question = first_question_map.get(chapter, first_question)
            break

    return (
        "Hello. I'm June.\n\n"
        "I work as a personal operating layer — tracking your plans, health, habits, relationships, "
        "and anything else that should not be forgotten.\n\n"
        "I get more useful the more context you give me, so let's start with one thing.\n\n"
        + first_question
    )


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

    # Chapter completeness ring
    completeness = memory.get_chapter_completeness()
    _total_chapters = len(completeness)
    _filled_chapters = sum(1 for v in completeness.values() if v > 0)
    _ring_svg = chapter_ring_svg(_filled_chapters, _total_chapters)
    _ring_label = (
        f"All {_total_chapters} chapters active"
        if _filled_chapters == _total_chapters
        else f"{_total_chapters - _filled_chapters} chapter(s) still empty"
    )

    st.markdown(
        '<div class="june-rail-card june-rail-card-primary">'
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:0.75rem;">'
        '<div style="flex:1;min-width:0;">'
        '<div class="june-label">Today</div>'
        f'<div class="june-title">{html.escape(model.title)}</div>'
        f'<div class="june-panel-caption">{html.escape(model.caption)}</div>'
        f'<div class="june-item-meta">{html.escape(model.subheadline)}</div>'
        '</div>'
        f'<div style="display:flex;flex-direction:column;align-items:center;gap:0.2rem;flex-shrink:0;">'
        f'{_ring_svg}'
        f'<div style="font-size:9px;color:var(--j-muted);text-align:center;white-space:nowrap;">{html.escape(_ring_label)}</div>'
        f'</div>'
        '</div>'
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
    """Force the transcript to scroll to the latest message."""
    components.html(
        """
        <script>
        const tryScroll = () => {
            try {
                const p = window.parent.document;
                const t = p.getElementById("june-transcript");
                const end = p.getElementById("june-transcript-end");
                if (t) t.scrollTop = t.scrollHeight;
                if (end) end.scrollIntoView({block: "nearest", behavior: "auto"});
                // Also scroll the Streamlit main app container
                const main = p.querySelector("[data-testid='stAppViewContainer']");
                if (main) main.scrollTop = main.scrollHeight;
            } catch(e) {}
        };
        const bindObserver = () => {
            try {
                const p = window.parent.document;
                const t = p.getElementById("june-transcript");
                if (!t || t.dataset.juneObserver === "1") return;
                t.dataset.juneObserver = "1";
                new MutationObserver(() => requestAnimationFrame(tryScroll))
                    .observe(t, {childList: true, subtree: true, characterData: true});
            } catch(e) {}
        };
        [0, 50, 150, 300, 600, 1000].forEach(ms => setTimeout(tryScroll, ms));
        setTimeout(bindObserver, 0);
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
            st.session_state.active_tool_names = []
        elif event.get("event") == "tool_calls_requested":
            tools = event.get("tools", [])
            st.session_state.active_tool_names = tools
            append_activity("tool request | " + ", ".join(tools))
        elif event.get("event") == "tool_results":
            st.session_state.active_tool_names = []  # tools done → back to typing state
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
# Health panel (left column)
# ---------------------------------------------------------------------------

def render_health_panel(memory: Memory) -> None:
    """Render the health/wellness left panel content."""
    st.markdown('<div class="june-panel-label">Health</div>', unsafe_allow_html=True)

    # Habits
    _habits = memory.get_habits()
    if _habits:
        st.markdown('<div class="june-panel-label" style="margin-top:0.4rem;">Habits</div>', unsafe_allow_html=True)
        for _h in _habits[:8]:
            _hname = _h.get("name", "")
            _done = memory.is_habit_completed_today(_hname)
            _hc1, _hc2 = st.columns([2.5, 1], gap="small")
            with _hc1:
                _ring = habit_ring_svg(_done, size=16)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.4rem;font-size:12px;padding:0.1rem 0;">'
                    f'{_ring}<span style="color:{"var(--j-muted)" if _done else "var(--j-text)"};'
                    f'text-decoration:{"line-through" if _done else "none"};">'
                    f'{html.escape(_hname)}</span></div>',
                    unsafe_allow_html=True,
                )
            with _hc2:
                if not _done:
                    if st.button("Done", key=f"hp_habit_{_hname}", use_container_width=True):
                        memory.log_habit_completion(_hname)
                        st.rerun()
    else:
        st.markdown(
            '<div style="font-size:11px;color:var(--j-muted);margin-bottom:0.5rem;">'
            'No habits yet. Ask June to set up your daily habits.</div>',
            unsafe_allow_html=True,
        )

    # Body snapshot
    _today_m = memory.get_today_body_metrics()
    if _today_m:
        _bits = []
        if _today_m.get("sleep_hours"):
            _bits.append(f"sleep {_today_m['sleep_hours']}h")
        if _today_m.get("energy"):
            _bits.append(f"energy {_today_m['energy']}/5")
        if _today_m.get("steps"):
            _bits.append(f"{int(_today_m['steps']):,} steps")
        if _bits:
            st.markdown(
                '<div style="font-size:9px;color:var(--j-muted);margin-top:0.45rem;">'
                + " · ".join(_bits) + "</div>",
                unsafe_allow_html=True,
            )

    # Quick body + water log
    st.markdown('<div class="june-panel-section"></div>', unsafe_allow_html=True)
    _water = memory.get_water_today()
    with st.expander(f"Log body check-in · water {_water}/8", expanded=False):
        with st.form("hp_body_form", clear_on_submit=False):
            _sl = st.number_input("Sleep hours", min_value=0.0, max_value=24.0, step=0.5,
                                  value=float(_today_m.get("sleep_hours", 0.0)) if _today_m else 0.0)
            _en = st.select_slider("Energy", options=[0, 1, 2, 3, 4, 5],
                                   value=int(_today_m.get("energy", 3)) if _today_m else 3)
            _st = st.select_slider("Stress", options=[0, 1, 2, 3, 4, 5],
                                   value=int(_today_m.get("stress", 0)) if _today_m else 0)
            _so = st.select_slider("Soreness", options=[0, 1, 2, 3, 4, 5],
                                   value=int(_today_m.get("soreness", 0)) if _today_m else 0)
            _sp = st.number_input("Steps", min_value=0, max_value=100000, step=500,
                                  value=int(_today_m.get("steps", 0)) if _today_m else 0)
            _wa = st.number_input("Water (glasses)", min_value=0, max_value=20, step=1,
                                  value=_water)
            if st.form_submit_button("Save", use_container_width=True):
                memory.log_body_metrics(
                    sleep_hours=_sl, energy=_en, stress=_st, soreness=_so, steps=_sp
                )
                if _wa != _water:
                    memory.set_water(_wa)
                append_activity("body | check-in saved from health panel")
                st.rerun()


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

@st.dialog("Settings", width="large")
def open_settings_dialog(
    memory: Memory,
    active_runtime: Any,
    stored_preset: str,
    current_user_id: str,
) -> None:
    """Settings modal: profile, LLM setup, options."""
    _tab_profile, _tab_llm, _tab_options = st.tabs(["Profile", "LLM Setup", "Options"])

    with _tab_profile:
        st.markdown("**Your profile name** — used to separate memory databases.")
        _new_uid = st.text_input("Profile name", value=current_user_id, key="settings_profile_input")
        if st.button("Save profile", use_container_width=True, type="primary"):
            st.session_state["profile_input"] = _new_uid
            st.rerun()

    with _tab_llm:
        _preset_opts = list(runtime_preset_options())
        _preset_keys = [p.key for p in _preset_opts]
        _sel_idx = _preset_keys.index(stored_preset) if stored_preset in _preset_keys else 0
        _chosen = st.selectbox(
            "Runtime",
            options=_preset_keys,
            index=_sel_idx,
            format_func=lambda k: runtime_for_preset(k).label,
            key="settings_runtime_picker",
        )
        _chosen_rt = runtime_for_preset(_chosen)
        _chosen_priv = build_runtime_privacy_status(_chosen_rt)
        st.markdown(
            f'<div style="font-size:11px;color:var(--j-muted);margin:0.4rem 0 0.75rem 0;">'
            f'{_chosen_priv["summary"]}</div>',
            unsafe_allow_html=True,
        )
        # Warn if switching to undownloaded local model
        if _chosen_rt.is_local and not is_model_available(_chosen_rt.model, _chosen_rt.base_url):
            _sz = model_size_label(_chosen_rt.model)
            st.warning(f"`{_chosen_rt.model}` is not downloaded{(' · ' + _sz) if _sz else ''}.")
            if st.button("Download this model", use_container_width=True):
                start_pull(_chosen_rt.model)
                st.info(f"Downloading {_chosen_rt.model} in the background. Switch to this runtime once complete.")
        if st.button("Apply LLM settings", use_container_width=True, type="primary"):
            st.session_state.selected_runtime_preset = _chosen
            memory.set_app_state_value("runtime_preset", _chosen)
            st.rerun()

    with _tab_options:
        privacy_icon = "○" if active_runtime.is_local else "◉"
        _priv = build_runtime_privacy_status(active_runtime)
        st.markdown(
            f'<div style="font-size:12px;margin-bottom:0.75rem;">'
            f'{privacy_icon} <strong>Privacy:</strong> {_priv["privacy_label"]} — {_priv["summary"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        if st.button("Clear conversation", use_container_width=True):
            reset_session_state(st.session_state, current_user_id)
            st.session_state.messages = []
            st.rerun()


# ---------------------------------------------------------------------------
# Calendar dialog
# ---------------------------------------------------------------------------

@st.dialog("Calendar", width="large")
def open_calendar_dialog(memory: Memory, now: datetime) -> None:
    """Calendar dialog with monthly view and agenda list."""
    import calendar as _cal
    from datetime import date, timedelta

    _views = ["Month", "Agenda"]
    _view = st.radio("View", _views, horizontal=True, index=0, label_visibility="collapsed")

    # Navigation state
    if "cal_year" not in st.session_state:
        st.session_state.cal_year = now.year
    if "cal_month" not in st.session_state:
        st.session_state.cal_month = now.month

    _cy, _cm = st.session_state.cal_year, st.session_state.cal_month

    if _view == "Month":
        _nav1, _nav2, _nav3 = st.columns([1, 3, 1])
        with _nav1:
            if st.button("◀", use_container_width=True):
                if _cm == 1:
                    st.session_state.cal_year -= 1
                    st.session_state.cal_month = 12
                else:
                    st.session_state.cal_month -= 1
                st.rerun()
        with _nav2:
            st.markdown(
                f'<div style="text-align:center;font-family:Syne,sans-serif;font-size:1rem;font-weight:600;">'
                f'{date(_cy, _cm, 1).strftime("%B %Y")}</div>',
                unsafe_allow_html=True,
            )
        with _nav3:
            if st.button("▶", use_container_width=True):
                if _cm == 12:
                    st.session_state.cal_year += 1
                    st.session_state.cal_month = 1
                else:
                    st.session_state.cal_month += 1
                st.rerun()

        # Pull calendar items for this month
        _all_items = memory.get_calendar_items(status="", limit=100)
        _month_prefix = date(_cy, _cm, 1).strftime("%Y-%m")
        _by_date: dict[str, list[str]] = {}
        for _ci in _all_items:
            _cd = str(_ci.get("date", ""))
            if _cd.startswith(_month_prefix):
                _by_date.setdefault(_cd, []).append(str(_ci.get("title", "")))

        # Weekday headers
        _dnames = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        _hcols = st.columns(7)
        for _i, _dn in enumerate(_dnames):
            with _hcols[_i]:
                st.markdown(
                    f'<div style="text-align:center;font-size:10px;color:var(--j-muted);'
                    f'font-weight:600;padding:0.25rem 0;">{_dn}</div>',
                    unsafe_allow_html=True,
                )

        # Build calendar grid
        _days_in_month = _cal.monthrange(_cy, _cm)[1]
        _first_wd = date(_cy, _cm, 1).weekday()  # 0=Mon
        _cells: list[date | None] = [None] * _first_wd + [
            date(_cy, _cm, d) for d in range(1, _days_in_month + 1)
        ]
        while len(_cells) % 7 != 0:
            _cells.append(None)

        _today = date.today()
        for _row_s in range(0, len(_cells), 7):
            _row = _cells[_row_s:_row_s + 7]
            _rcols = st.columns(7)
            for _col, _cdate in zip(_rcols, _row):
                with _col:
                    if _cdate is None:
                        st.markdown('<div style="height:2.8rem;"></div>', unsafe_allow_html=True)
                    else:
                        _ds = _cdate.isoformat()
                        _is_today = _cdate == _today
                        _has_event = _ds in _by_date
                        _bg = "var(--j-accent)" if _is_today else ("var(--j-accent-soft)" if _has_event else "transparent")
                        _color = "#fff" if _is_today else "var(--j-text)"
                        _fw = "700" if _is_today else "400"
                        _dot = (
                            '<span style="display:block;width:4px;height:4px;border-radius:50%;'
                            'background:var(--j-accent);margin:2px auto 0;"></span>'
                            if _has_event and not _is_today else ""
                        )
                        _titles = " · ".join(_by_date.get(_ds, []))[:24] if _has_event else ""
                        _title_attr = f' title="{html.escape(_titles)}"' if _titles else ""
                        st.markdown(
                            f'<div{_title_attr} style="text-align:center;padding:0.3rem 0.1rem;'
                            f'background:{_bg};border-radius:8px;min-height:2.8rem;cursor:{"pointer" if _has_event else "default"};">'
                            f'<span style="font-size:12px;color:{_color};font-weight:{_fw};">{_cdate.day}</span>'
                            f'{_dot}</div>',
                            unsafe_allow_html=True,
                        )

    # Agenda list (always shown below calendar, or as own view)
    st.markdown(
        '<hr style="border:none;border-top:1px solid var(--j-line);margin:0.75rem 0 0.5rem 0;">',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="june-panel-label">Upcoming</div>', unsafe_allow_html=True)
    _upcoming = memory.get_upcoming_notifications(limit=12)
    if _upcoming:
        for _u in _upcoming:
            _prefix = "today" if _u["days_until"] == 0 else f"in {_u['days_until']}d"
            _det = f" — {_u['details']}" if _u.get("details") else ""
            st.markdown(
                f'<div style="padding:0.35rem 0;border-bottom:1px solid var(--j-line);font-size:12px;">'
                f'<span style="font-weight:600;">{html.escape(_u["title"])}</span>'
                f'<span style="color:var(--j-muted);margin-left:0.5rem;">{html.escape(_u["when"])} · {_prefix}</span>'
                f'<span style="color:var(--j-muted);font-size:10px;">{html.escape(_det)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="font-size:11px;color:var(--j-muted);">'
            'No upcoming events. Ask June to schedule something.</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# User context — resolved before main layout
# ---------------------------------------------------------------------------

now = current_local_time()
if "profile_input" not in st.session_state:
    st.session_state["profile_input"] = "admin"
user_id = st.session_state["profile_input"]

_init_memory = Memory(user_id)
sidebar_phrase = get_rotating_sidebar_phrase(_init_memory, now)
stored_runtime_preset = str(
    st.session_state.get(
        "selected_runtime_preset",
        _init_memory.get_app_state().get("runtime_preset", RUNTIME_CONFIG.preset_key),
    )
)
if "selected_runtime_preset" not in st.session_state:
    st.session_state.selected_runtime_preset = stored_runtime_preset

# ---------------------------------------------------------------------------
# Left sidebar — hidden; settings go through the dialog
# ---------------------------------------------------------------------------

# Sidebar is hidden via CSS — kept as a stub only so Streamlit doesn't error
# on any lingering widget keys from old sessions.
with st.sidebar:
    pass


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

initialize_session_state(st.session_state, user_id)

if st.session_state.ui_state.get("selected_chapter") and st.session_state.selected_chapter != st.session_state.ui_state.get("selected_chapter"):
    st.session_state.selected_chapter = st.session_state.ui_state.get("selected_chapter", "")
st.session_state.show_right_panel = st.session_state.ui_state.get("show_right_panel", True)
if "show_left_panel" not in st.session_state:
    st.session_state.show_left_panel = True
if "active_panel" not in st.session_state:
    st.session_state.active_panel = "today"
if "show_chat" not in st.session_state:
    st.session_state.show_chat = True
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

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
if "active_tool_names" not in st.session_state:
    st.session_state.active_tool_names = []
active_runtime = runtime_for_preset(str(st.session_state.selected_runtime_preset))
active_privacy = build_runtime_privacy_status(active_runtime)
st.session_state.current_runtime_label = active_runtime.label
st.session_state.current_runtime_model = active_runtime.model

# Compute display variables before rendering anything
snapshot = memory.get_progress_snapshot()
active_skill = SKILLS.get(st.session_state.active_skill_key, SKILLS[DEFAULT_SKILL])
show_right_panel = st.session_state.ui_state.get("show_right_panel", True)
show_left_panel = st.session_state.get("show_left_panel", True)

# Model availability flag — drives chat vs download screen
_models_verified: set = st.session_state.setdefault("_models_verified", set())
_active_model_key = f"{active_runtime.preset_key}:{active_runtime.model}"
_model_ready = (
    not active_runtime.is_local
    or not active_runtime.base_url
    or _active_model_key in _models_verified
    or is_model_available(active_runtime.model, active_runtime.base_url)
)
if _model_ready:
    _models_verified.add(_active_model_key)

# ---------------------------------------------------------------------------
# Nav header: Logo | Today Agenda Plans Gym&Food Health Calendar | dark ⚙
# ---------------------------------------------------------------------------

_date_label = now.strftime("%a, %d %b")
_time_label = now.strftime("%H:%M")
_part_label = current_part_of_day(now)
_privacy_color = "var(--j-accent)" if active_runtime.is_local else "#c07a2a"
_privacy_label = "○ local" if active_runtime.is_local else "◉ cloud"

_NAV_PANELS = [
    ("today",    "Today"),
    ("agenda",   "Agenda"),
    ("plans",    "Plans"),
    ("gym_food", "Gym & Food"),
    ("health",   "Health"),
    ("calendar", "Calendar"),
]
_active_panel = st.session_state.active_panel
_show_chat    = st.session_state.show_chat
_dark_mode    = st.session_state.dark_mode

# Apply dark mode class to body via JS
if _dark_mode:
    components.html(
        "<script>window.parent.document.body.classList.add('june-dark');</script>",
        height=0, width=0,
    )
else:
    components.html(
        "<script>window.parent.document.body.classList.remove('june-dark');</script>",
        height=0, width=0,
    )

# Nav bar: logo | radio nav | model pill + dark + settings
_nav_left, _nav_center, _nav_right = st.columns([1.2, 6, 1.8], gap="small")

with _nav_left:
    st.markdown(
        f'<div style="display:flex;align-items:center;height:44px;">'
        f'<img src="/app/static/june_ai_logo.png" alt="June" '
        f'style="height:22px;width:auto;object-fit:contain;">'
        f'</div>',
        unsafe_allow_html=True,
    )

with _nav_center:
    _nav_labels = [pl for _, pl in _NAV_PANELS]
    _nav_keys   = [pk for pk, _ in _NAV_PANELS]
    _nav_idx    = _nav_keys.index(_active_panel) if _active_panel in _nav_keys else 0
    _selected_label = st.radio(
        "nav", _nav_labels, index=_nav_idx,
        horizontal=True, label_visibility="collapsed",
        key="nav_radio",
    )
    _selected_key = _nav_keys[_nav_labels.index(_selected_label)]
    if _selected_key != st.session_state.active_panel:
        st.session_state.active_panel = _selected_key
        st.rerun()

with _nav_right:
    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:flex-end;'
        f'gap:0.5rem;height:44px;">',
        unsafe_allow_html=True,
    )
    _r1, _r2, _r3 = st.columns([2, 1, 1], gap="small")
    with _r1:
        st.markdown(
            f'<div style="display:flex;align-items:center;height:44px;">'
            f'<span style="font-size:9px;color:{_privacy_color};'
            f'border:1px solid var(--j-line);border-radius:999px;'
            f'padding:0.15rem 0.55rem;white-space:nowrap;">'
            f'{html.escape(active_runtime.model.split(":")[0])} · {_privacy_label}'
            f'</span></div>',
            unsafe_allow_html=True,
        )
    with _r2:
        _dm_label = "☀" if _dark_mode else "☾"
        st.markdown('<div class="june-nav-action">', unsafe_allow_html=True)
        if st.button(_dm_label, key="nav_dark_mode", use_container_width=True, help="Dark mode"):
            st.session_state.dark_mode = not _dark_mode
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with _r3:
        st.markdown('<div class="june-nav-action">', unsafe_allow_html=True)
        if st.button("⚙", key="nav_settings", use_container_width=True, help="Settings"):
            open_settings_dialog(memory, active_runtime, stored_runtime_preset, user_id)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<hr style="margin:0 0 0.5rem 0;border:none;border-top:1px solid var(--j-line);">',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Two-column layout: Chat (1/3, hideable) | Active panel (2/3)
# ---------------------------------------------------------------------------

if _show_chat:
    _chat_col_raw, right_col = st.columns([1, 2], gap="medium")
else:
    _chat_col_raw, right_col = st.empty(), st.container()

# Daily check-in only fires when the model is ready
if _model_ready and not st.session_state.is_generating and memory.should_send_daily_checkin():
    if is_first_run(memory):
        opening_message = build_welcome_message(memory)
        append_activity("welcome | first run")
    else:
        opening_message = build_daily_checkin(memory)
        append_activity("daily check-in | sent")
    st.session_state.messages.append(AIMessage(content=opening_message))
    memory.save_message("assistant", opening_message)
    memory.mark_daily_checkin_sent()

# ── Right: Active panel ───────────────────────────────────────────────────

with right_col:
    st.markdown('<div class="june-panel-area">', unsafe_allow_html=True)
    workspace_placeholder = st.empty()
    activity_placeholder = st.empty()

    if _active_panel == "today":
        render_today_panel(memory, snapshot)
    elif _active_panel == "agenda":
        render_memory_panel(memory)
    elif _active_panel == "plans":
        render_plan_focus(memory)
    elif _active_panel == "gym_food":
        render_habits_focus(memory)
    elif _active_panel == "health":
        render_health_panel(memory)
    elif _active_panel == "calendar":
        open_calendar_dialog(memory, now)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Left: Chat column ─────────────────────────────────────────────────────
# Always use with chat_col — when hidden it goes into st.empty() (discarded)

chat_col = _chat_col_raw

with chat_col:
  st.markdown('<div class="june-chat-toggle">', unsafe_allow_html=True)
  if _show_chat:
    if st.button("← Hide chat", key="toggle_chat", help="Collapse chat panel"):
        st.session_state.show_chat = False
        st.rerun()
  else:
    if st.button("Chat →", key="toggle_chat_open", help="Expand chat panel"):
        st.session_state.show_chat = True
        st.rerun()
  st.markdown('</div>', unsafe_allow_html=True)
  if not _model_ready:
    # ── Model download screen ──────────────────────────────────────────
    import time as _time

    _dl_model = active_runtime.model
    _dl_base_url = active_runtime.base_url
    _size_label = model_size_label(_dl_model)
    _size_str = f" · {_size_label}" if _size_label else ""

    st.markdown(
        f'<div style="text-align:center;padding:2.5rem 0 1rem;">'
        f'<div style="font-family:Syne,sans-serif;font-size:1.25rem;font-weight:600;'
        f'color:var(--j-text);margin-bottom:0.35rem;">Model not downloaded</div>'
        f'<div style="font-size:13px;color:var(--j-muted);margin-bottom:1.5rem;">'
        f'<code>{_dl_model}</code>{_size_str}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.get("_pulling_active_model"):
        _bc1, _bc2 = st.columns(2, gap="small")
        with _bc1:
            if st.button("Download now", key="pull_active_model", use_container_width=True, type="primary"):
                st.session_state["_pulling_active_model"] = _dl_model
                st.rerun()
        with _bc2:
            if st.button("Use Llama 3.2 instead", key="fallback_to_llama", use_container_width=True):
                st.session_state.selected_runtime_preset = "local_llama3_2"
                memory.set_app_state_value("runtime_preset", "local_llama3_2")
                st.rerun()
    else:
        # Kick off download with progress tracking (once)
        if not st.session_state.get("_pull_proc_started"):
            _, _pf = start_pull_with_progress(_dl_model)
            st.session_state["_pull_proc_started"] = True
            st.session_state["_pull_progress_file"] = _pf
            st.session_state["_pull_start_time"] = _time.time()

        _elapsed_s = int(_time.time() - st.session_state.get("_pull_start_time", _time.time()))
        _elapsed_str = (
            f"{_elapsed_s}s" if _elapsed_s < 60
            else f"{_elapsed_s // 60}m {_elapsed_s % 60}s"
        )
        _pf = st.session_state.get("_pull_progress_file", "")
        _pct, _status = read_pull_progress(_pf)

        # ── Error state: Ollama version too old, or other fatal error ──
        if _status.startswith("ERR:"):
            _err_detail = _status[4:]
            _is_version_err = "needs_update" in _err_detail or "version" in _err_detail.lower()
            cleanup_progress_file(_pf)
            for _k in ("_pulling_active_model", "_pull_proc_started",
                       "_pull_start_time", "_pull_progress_file"):
                st.session_state.pop(_k, None)
            if _is_version_err:
                st.error(
                    f"**Ollama is out of date** — `{_dl_model}` requires a newer version.\n\n"
                    f"Click **Upgrade Ollama** below, or run manually in your terminal:"
                )
                st.code("brew upgrade ollama", language="bash")
                _uc1, _uc2 = st.columns(2, gap="small")
                with _uc1:
                    if st.button("Upgrade Ollama", key="err_upgrade_ollama",
                                 use_container_width=True, type="primary"):
                        import subprocess as _sub
                        _brew = _sub.run(["brew", "upgrade", "ollama"],
                                         capture_output=True, text=True)
                        if _brew.returncode == 0:
                            st.success("Ollama upgraded. Click Download now to retry.")
                        else:
                            st.error(f"brew upgrade failed:\n{_brew.stderr[:200]}")
                with _uc2:
                    if st.button("Use Llama 3.2 instead", key="err_fallback_llama",
                                 use_container_width=True):
                        st.session_state.selected_runtime_preset = "local_llama3_2"
                        memory.set_app_state_value("runtime_preset", "local_llama3_2")
                        st.rerun()
            else:
                st.error(f"Download failed: {_err_detail}")
                if st.button("Use Llama 3.2 instead", key="err_fallback_llama2",
                             use_container_width=True):
                    st.session_state.selected_runtime_preset = "local_llama3_2"
                    memory.set_app_state_value("runtime_preset", "local_llama3_2")
                    st.rerun()
        elif not ollama_cli_available():
            st.info("Ollama CLI not found. Run manually:")
            st.code(f"ollama pull {_dl_model}")
        else:
            # Normal in-progress display
            _pct_label = f" — {_pct}%" if _pct > 0 else ""
            _progress_label = f"{_status}{_pct_label} · {_elapsed_str} elapsed"
            st.progress(_pct / 100, text=_progress_label)
            st.markdown(
                f'<div style="font-size:11px;color:var(--j-muted);text-align:center;margin-top:0.4rem;">'
                f'Downloading {html.escape(_dl_model)}'
                f'{(" (" + html.escape(_size_label) + ")") if _size_label else ""} — '
                f'do not close this window.'
                f'</div>',
                unsafe_allow_html=True,
            )
            # Poll every 2 s
            if is_model_available(_dl_model, _dl_base_url):
                cleanup_progress_file(_pf)
                for _k in ("_pulling_active_model", "_pull_proc_started",
                           "_pull_start_time", "_pull_progress_file"):
                    st.session_state.pop(_k, None)
                _models_verified.add(_active_model_key)
                st.rerun()
            else:
                _time.sleep(2)
                st.rerun()

  else:
    # ── Normal chat ──────────────────────────────────────────────────
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

    # Typing indicator + active tool badges
    if st.session_state.is_generating:
        active_tools = st.session_state.get("active_tool_names", [])
        if active_tools:
            badges = "".join(
                f'<span class="june-tool-badge">'
                f'<span class="june-tool-spinner"></span>'
                f'{html.escape(name.replace("_", " "))}'
                f'</span>'
                for name in active_tools[:4]
            )
            st.markdown(
                f'<div class="june-tool-activity">{badges}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="june-typing">'
                '<span class="june-typing-dot"></span>'
                '<span class="june-typing-dot"></span>'
                '<span class="june-typing-dot"></span>'
                '<span class="june-typing-label">June is writing</span>'
                '</div>',
                unsafe_allow_html=True,
            )

    # Cmd+Enter keyboard shortcut
    components.html(
        """
        <script>
        (function() {
            function bindCmdEnter() {
                const doc = window.parent.document;
                const textarea = doc.querySelector('textarea[data-testid="stTextArea"], textarea');
                if (!textarea || textarea.dataset.juneCmdEnterBound === '1') return;
                textarea.dataset.juneCmdEnterBound = '1';
                textarea.addEventListener('keydown', function(e) {
                    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                        e.preventDefault();
                        const submitBtn = doc.querySelector('button[kind="formSubmit"], [data-testid="stFormSubmitButton"] button, button[type="submit"]');
                        if (submitBtn) submitBtn.click();
                    }
                });
            }
            setTimeout(bindCmdEnter, 400);
            setTimeout(bindCmdEnter, 1200);
        })();
        </script>
        """,
        height=0,
        width=0,
    )

    # Input area — sticky at bottom
    st.markdown('<div class="june-input-wrap">', unsafe_allow_html=True)
    with st.form("june_input_form", clear_on_submit=True):
        prompt = st.text_area(
            "Message June",
            value="",
            placeholder="Tell June about your day, plans, goals, feelings, or anything worth remembering.",
            label_visibility="collapsed",
            height=72,
        )
        st.markdown('<div class="june-send-btn">', unsafe_allow_html=True)
        submitted = st.form_submit_button("Send", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="june-input-hint">Cmd+Enter · or press Send</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted and prompt.strip() and not st.session_state.is_generating:
        st.session_state.pending_prompt = prompt.strip()
        st.session_state.active_skill_key = infer_skill_from_text(prompt)
        st.session_state.is_generating = True
        append_activity(f"auto route | {st.session_state.active_skill_key}")
        st.rerun()


# workspace_placeholder / activity_placeholder already set in right_col block above

# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

if _model_ready and st.session_state.is_generating and st.session_state.pending_prompt:
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
            active_runtime.prompt_style,
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
        _exc_str = str(exc)
        _is_model_not_found = (
            "404" in _exc_str
            and ("not found" in _exc_str.lower() or "model" in _exc_str.lower())
        )
        if _is_model_not_found:
            # Clear the verified-models cache so the check above triggers on rerun
            st.session_state.pop("_models_verified", None)
            st.rerun()
        else:
            st.error(f"June ran into an issue: {_exc_str}")
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
    st.session_state.active_tool_names = []
    st.rerun()
