"""JuneAI Streamlit frontend.

Run with: streamlit run app.py
"""

from __future__ import annotations

import html
from datetime import datetime

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from src.agent.graph import june_agent
from src.agent.config import resolve_runtime_config
from src.agent.memory import Memory
from src.agent.skills import DEFAULT_SKILL, SKILLS, infer_skill_from_text
from src.agent.tools import DEFAULT_UI_STATE

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
        background: #ffffff;
        border-right: 1px solid var(--june-line);
        box-shadow: 4px 0 24px rgba(40, 28, 18, 0.04);
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
        font-size: 11px;
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
        background: #ffffff;
        transition: border-color 0.18s;
    }
    .june-item:hover { border-color: rgba(15, 95, 74, 0.22); }

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
        margin: 0.6rem 0 0.3rem 0;
        padding-top: 0.5rem;
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
        background: #ffffff;
        border: 1px solid var(--june-line);
        border-radius: 22px;
        box-shadow: var(--june-shadow-lg);
        padding: 1rem;
        animation: panelIn 0.28s ease both;
        max-height: calc(100vh - 2rem);
        overflow-y: auto;
        position: sticky;
        top: 0.75rem;
    }

    .june-panel-section {
        padding-bottom: 0.8rem;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid var(--june-line);
    }
    .june-panel-section:last-child {
        border-bottom: none;
        margin-bottom: 0;
        padding-bottom: 0;
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
        background: #ffffff;
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

    ul { margin: 0.35rem 0 0 1rem; padding: 0; }

    /* Hide default Streamlit expander arrow styling a bit */
    details summary { font-size: 11px; color: var(--june-muted); }
    </style>
    """,
    unsafe_allow_html=True,
)

WATER_GOAL = 8
RUNTIME_CONFIG = resolve_runtime_config()

CHAPTERS = [
    ("calendar",  "Calendar"),
    ("gym",       "Gym Schedule"),
    ("food",      "Food Schedule"),
    ("trips",     "Trips"),
    ("plans",     "Plans"),
    ("dating",    "Dating/Love"),
    ("family",    "Family"),
    ("birthdays", "Birthdays"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_ui_state() -> dict:
    return {
        "layout":          DEFAULT_UI_STATE["layout"],
        "focus_title":     DEFAULT_UI_STATE["focus_title"],
        "focus_body":      DEFAULT_UI_STATE["focus_body"],
        "checklist_title": DEFAULT_UI_STATE["checklist_title"],
        "checklist_items": list(DEFAULT_UI_STATE["checklist_items"]),
        "notice":          DEFAULT_UI_STATE["notice"],
    }


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def habit_ring_svg(done: bool, size: int = 26) -> str:
    r = 10
    c = size // 2
    circumference = round(2 * 3.14159 * r, 2)
    offset = 0 if done else circumference
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
        f'style="display:block;flex-shrink:0;">'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" '
        f'stroke="rgba(22,20,16,0.10)" stroke-width="2"/>'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" '
        f'stroke="#0f5f4a" stroke-width="2" '
        f'stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" '
        f'transform="rotate(-90 {c} {c})" '
        f'style="transition:stroke-dashoffset 0.55s cubic-bezier(0.34,1.56,0.64,1);"/>'
        f'</svg>'
    )


def water_dots_html(count: int, goal: int = WATER_GOAL) -> str:
    dots = []
    for i in range(goal):
        css = "filled" if i < count else "empty"
        delay = f"animation-delay:{i * 0.04}s;" if i < count else ""
        dots.append(f'<span class="june-water-dot {css}" style="{delay}"></span>')
    return f'<div class="june-water-track">{"".join(dots)}</div>'


def energy_dots_html(value: int, max_val: int = 5) -> str:
    dots = "".join(
        f'<span class="june-metric-dot {"active" if i < value else "inactive"}"></span>'
        for i in range(max_val)
    )
    return f'<div class="june-metric-dots">{dots}</div>'


def transcript_html(messages: list, live_response: str = "") -> str:
    blocks = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            blocks.append(
                '<div class="june-message june-message-user">'
                '<div class="june-message-label">You</div>'
                f"{html.escape(extract_text(msg.content))}"
                "</div>"
            )
        elif isinstance(msg, AIMessage) and msg.content:
            blocks.append(
                '<div class="june-message june-message-assistant">'
                '<div class="june-message-label">June</div>'
                f"{html.escape(extract_text(msg.content))}"
                "</div>"
            )
    if live_response:
        blocks.append(
            '<div class="june-message june-message-assistant">'
            '<div class="june-message-label">June</div>'
            f"{html.escape(live_response)}"
            "</div>"
        )
    return (
        '<div class="june-transcript" id="june-transcript">'
        + "".join(blocks)
        + "</div>"
        + """<script>
        const t = window.parent.document.getElementById("june-transcript");
        if (t) t.scrollTop = t.scrollHeight;
        </script>"""
    )


def render_list(items: list[tuple[str, str]]) -> str:
    if not items:
        return '<div class="june-item-meta">Nothing saved yet.</div>'
    return '<div class="june-list">' + "".join(
        f'<div class="june-item">'
        f'<div class="june-item-title">{html.escape(title)}</div>'
        f'<div class="june-item-meta">{html.escape(meta)}</div>'
        f'</div>'
        for title, meta in items
    ) + "</div>"


def render_workspace(ui_state: dict) -> str:
    checklist_items = ui_state.get("checklist_items", [])
    checklist = (
        "".join(f"<li>{html.escape(item)}</li>" for item in checklist_items)
        if checklist_items
        else "<li>No pinned actions.</li>"
    )
    return (
        f'<div class="june-label">Workspace</div>'
        f'<h3 class="june-title">{html.escape(ui_state.get("focus_title", "Workspace"))}</h3>'
        f'<div class="june-item-meta">{html.escape(ui_state.get("focus_body", ""))}</div>'
        f'<div class="june-label" style="margin-top:0.6rem;">'
        f'{html.escape(ui_state.get("checklist_title", "Next steps"))}</div>'
        f'<div class="june-item-meta"><ul>{checklist}</ul></div>'
        + (
            f'<div class="june-item-meta" style="margin-top:0.5rem;">'
            f'{html.escape(ui_state.get("notice", ""))}</div>'
            if ui_state.get("notice") else ""
        )
    )


def render_activity(activity_log: list[str]) -> str:
    if not activity_log:
        return '<div class="june-item-meta">No activity yet.</div>'
    return '<div class="june-list">' + "".join(
        f'<div class="june-item">'
        f'<div class="june-item-meta">{html.escape(line)}</div>'
        f'</div>'
        for line in activity_log[-10:]
    ) + "</div>"


def append_activity(message: str) -> None:
    st.session_state.activity_log.append(message)


def chapter_items(memory: Memory, chapter_key: str) -> list[tuple[str, str]]:
    if chapter_key == "calendar":
        return [
            (
                item["title"],
                f"{item['date']}{' ' + item['time'] if item.get('time') else ''}"
                f"{' | ' + item['details'] if item.get('details') else ''}",
            )
            for item in memory.get_calendar_items(limit=12)
        ]
    if chapter_key == "gym":
        return [
            (item["name"], f"{item['schedule']}{' | Goal: ' + item['goal'] if item.get('goal') else ''}")
            for item in memory.get_gym_plans(limit=12)
        ]
    if chapter_key == "food":
        return [
            (item["name"], f"{item['daily_structure']}{' | Goal: ' + item['goal'] if item.get('goal') else ''}")
            for item in memory.get_food_programs(limit=12)
        ]
    if chapter_key == "trips":
        return [
            (item["title"], f"{item['date']}{' | ' + item['details'] if item.get('details') else ''}")
            for item in memory.get_calendar_items(limit=30)
            if any(kw in _cal_text(item) for kw in ("trip", "travel", "flight"))
        ]
    if chapter_key == "plans":
        return [
            (item["title"], f"Goal | {item['status']}{' | Next: ' + item['next_step'] if item.get('next_step') else ''}")
            for item in memory.get_goals(status="", limit=20)
        ] + [
            (item["topic"], f"Open loop | {item['status']}{' | Due: ' + item['due_date'] if item.get('due_date') else ''}")
            for item in memory.get_open_loops(status="", limit=20)
        ]
    if chapter_key == "dating":
        return [
            (item["person"], f"{item['relationship']} | {item['summary']}")
            for item in memory.get_relationship_profiles()
            if any(t in item.get("relationship", "").lower()
                   for t in ("dating", "love", "partner", "girlfriend", "boyfriend", "romantic", "spouse"))
        ]
    if chapter_key == "family":
        return [
            (item["person"], f"{item['relationship']} | {item['summary']}")
            for item in memory.get_relationship_profiles()
            if any(t in item.get("relationship", "").lower()
                   for t in ("family", "mother", "father", "mom", "dad", "brother",
                             "sister", "parent", "child", "cousin", "uncle", "aunt"))
        ]
    if chapter_key == "birthdays":
        return [
            (item["title"], f"{item['date']}{' | ' + item['details'] if item.get('details') else ''}")
            for item in memory.get_calendar_items(limit=30)
            if "birthday" in _cal_text(item)
        ]
    return []


def _cal_text(item: dict) -> str:
    return " ".join(str(item.get(f, "")).lower() for f in ("title", "details", "source", "status"))


def chapter_subtitle(chapter_key: str) -> str:
    return {
        "calendar":  "Appointments, reminders, and events.",
        "gym":       "Training routines and weekly splits.",
        "food":      "Food structure and nutrition plans.",
        "trips":     "Travel plans and movement events.",
        "plans":     "Goals, open loops, and next steps.",
        "dating":    "Relationship memory for love and dating.",
        "family":    "Family context and relationship notes.",
        "birthdays": "Birthday reminders and personal dates.",
    }.get(chapter_key, "")


def render_notifications(memory: Memory) -> str:
    notifications = memory.get_upcoming_notifications(limit=5)
    items = []
    for item in notifications:
        prefix = "today" if item["days_until"] == 0 else f"in {item['days_until']}d"
        items.append((
            item["title"],
            f"{item['kind']} · {item['when']} · {prefix}"
            f"{' | ' + item['details'] if item.get('details') else ''}",
        ))
    return render_list(items)


def render_capture_health(memory: Memory, activity_log: list[str]) -> str:
    chapter_counts = {
        "Agenda":    len(memory.get_calendar_items(limit=100)),
        "Birthdays": len(chapter_items(memory, "birthdays")),
        "Trips":     len(chapter_items(memory, "trips")),
        "Gym":       len(memory.get_gym_plans(limit=100)),
        "Food":      len(memory.get_food_programs(limit=100)),
        "Plans":     len(memory.get_goals(status="", limit=100)) + len(memory.get_open_loops(status="", limit=100)),
        "Dating":    len(chapter_items(memory, "dating")),
        "Family":    len(chapter_items(memory, "family")),
    }
    cards = "".join(
        f'<div class="june-stat-card">'
        f'<div class="june-stat-label">{html.escape(label)}</div>'
        f'<div class="june-stat-value">{count}</div>'
        f'</div>'
        for label, count in chapter_counts.items()
    )
    recent_saves = [
        line for line in activity_log[-30:]
        if "save_" in line or "Saved " in line or "saved " in line
    ][-5:]
    save_html = (
        render_list([("Capture", line) for line in recent_saves])
        if recent_saves
        else '<div class="june-item-meta">No recent captures this session.</div>'
    )
    return f'<div class="june-stat-grid">{cards}</div><div style="margin-top:0.6rem;"></div>' + save_html


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
    return state.get("sidebar_phrase_text", generate_sidebar_phrase(now))


def build_daily_checkin(memory: Memory) -> str:
    """Build June's proactive daily opening with a chapter-specific intake question."""
    now = current_local_time()
    part = current_part_of_day(now)
    opening = {"morning": "Good morning.", "afternoon": "Good afternoon.",
               "evening": "Good evening.", "night": "Good evening."}[part]

    notifications = memory.get_upcoming_notifications(limit=3)
    empty_chapters = memory.get_chapters_needing_attention()

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
    }

    if empty_chapters:
        question = chapter_questions.get(empty_chapters[0])
        if question:
            lines.append(question)
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


def handle_stream_chunk(
    mode: str,
    data,
    transcript_placeholder,
    workspace_placeholder,
    activity_placeholder,
) -> None:
    if mode == "custom":
        event = data or {}
        if event.get("event") == "chat_started":
            append_activity(f"route | {event.get('skill')}")
            append_activity(
                "runtime | "
                + f"{event.get('runtime_label', RUNTIME_CONFIG.label)}"
                + f" | {event.get('runtime_model', RUNTIME_CONFIG.model)}"
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
                    workspace_placeholder.markdown(
                        render_workspace(st.session_state.ui_state),
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
            workspace_placeholder.markdown(
                render_workspace(st.session_state.ui_state),
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
    sidebar_phrase = get_rotating_sidebar_phrase(memory_for_sidebar, now)

    st.markdown('<div class="june-brand">June</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="june-copy">{html.escape(sidebar_phrase)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"{now.strftime('%A %d %B')} · {current_part_of_day(now)} · day {now.timetuple().tm_yday}")
    st.caption(f"{RUNTIME_CONFIG.label} · {RUNTIME_CONFIG.model}")

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
    st.markdown('<div class="june-section-label">Body</div>', unsafe_allow_html=True)

    if today_metrics:
        st.markdown(
            f'<div class="june-body-row"><span class="june-body-key">energy</span>'
            f'{energy_dots_html(today_metrics.get("energy", 0))}</div>'
            f'<div class="june-body-row"><span class="june-body-key">sleep</span>'
            f'<span style="font-size:10px;">{today_metrics.get("sleep_hours", 0)}h</span></div>'
            + (
                f'<div class="june-body-row"><span class="june-body-key">weight</span>'
                f'<span style="font-size:10px;">{today_metrics.get("weight_kg", 0)}kg</span></div>'
                if today_metrics.get("weight_kg") else ""
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div style="font-size:10px;color:var(--june-muted);">Not logged today.</div>', unsafe_allow_html=True)

    with st.expander("Log body", expanded=False):
        with st.form("sb_body_form", clear_on_submit=True):
            e_in = st.select_slider("Energy", options=[1, 2, 3, 4, 5], value=3)
            s_in = st.number_input("Sleep h", min_value=0.0, max_value=24.0, step=0.5, value=0.0)
            w_in = st.number_input("Weight kg", min_value=0.0, max_value=300.0, step=0.1, value=0.0)
            if st.form_submit_button("Save", use_container_width=True):
                memory_for_sidebar.log_body_metrics(weight_kg=w_in, sleep_hours=s_in, energy=e_in)
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
        st.session_state.messages = []
        st.session_state.activity_log = []
        st.session_state.ui_state = default_ui_state()
        st.session_state.live_response = ""
        st.session_state.final_state = None
        st.session_state.pending_prompt = ""
        st.session_state.is_generating = False
        st.session_state.active_skill_key = DEFAULT_SKILL
        st.session_state.tool_stats = {"requested": 0, "succeeded": 0, "failed": 0, "last_calls": []}
        st.rerun()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = Memory(user_id).load_chat_messages()
if "last_user_id" not in st.session_state:
    st.session_state.last_user_id = user_id
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []
if "ui_state" not in st.session_state:
    st.session_state.ui_state = default_ui_state()
if "live_response" not in st.session_state:
    st.session_state.live_response = ""
if "final_state" not in st.session_state:
    st.session_state.final_state = None
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = ""
if "active_skill_key" not in st.session_state:
    st.session_state.active_skill_key = DEFAULT_SKILL
if "selected_chapter" not in st.session_state:
    st.session_state.selected_chapter = ""
if "tool_stats" not in st.session_state:
    st.session_state.tool_stats = {"requested": 0, "succeeded": 0, "failed": 0, "last_calls": []}

if st.session_state.last_user_id != user_id:
    st.session_state.messages = Memory(user_id).load_chat_messages()
    st.session_state.activity_log = []
    st.session_state.ui_state = default_ui_state()
    st.session_state.live_response = ""
    st.session_state.final_state = None
    st.session_state.pending_prompt = ""
    st.session_state.is_generating = False
    st.session_state.active_skill_key = DEFAULT_SKILL
    st.session_state.selected_chapter = ""
    st.session_state.tool_stats = {"requested": 0, "succeeded": 0, "failed": 0, "last_calls": []}
    st.session_state.last_user_id = user_id

# ---------------------------------------------------------------------------
# Memory, daily check-in, snapshot
# ---------------------------------------------------------------------------

memory = Memory(user_id)

if not st.session_state.is_generating and memory.should_send_daily_checkin():
    daily_message = build_daily_checkin(memory)
    st.session_state.messages.append(AIMessage(content=daily_message))
    memory.save_message("assistant", daily_message)
    memory.mark_daily_checkin_sent()
    append_activity("daily check-in | sent")

snapshot = memory.get_progress_snapshot()
active_skill = SKILLS.get(st.session_state.active_skill_key, SKILLS[DEFAULT_SKILL])

# ---------------------------------------------------------------------------
# Main layout: Conversation | Right panel
# ---------------------------------------------------------------------------

chat_col, plan_col = st.columns([1.9, 1], gap="medium")

# ── Conversation ──────────────────────────────────────────────────────────

with chat_col:
    st.markdown('<div class="june-surface">', unsafe_allow_html=True)
    st.markdown('<div class="june-label">Conversation</div>', unsafe_allow_html=True)
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
    transcript_placeholder = st.empty()
    transcript_placeholder.markdown(
        transcript_html(st.session_state.messages, st.session_state.live_response),
        unsafe_allow_html=True,
    )
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

with plan_col:
    st.markdown('<div class="june-right-panel">', unsafe_allow_html=True)

    # Upcoming reminders
    st.markdown('<div class="june-panel-section">', unsafe_allow_html=True)
    st.markdown('<div class="june-label">Upcoming</div>', unsafe_allow_html=True)
    st.markdown(render_notifications(memory), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Chapters
    st.markdown('<div class="june-panel-section">', unsafe_allow_html=True)
    st.markdown('<div class="june-label">Chapters</div>', unsafe_allow_html=True)
    st.markdown('<div class="june-chapter-grid">', unsafe_allow_html=True)
    chapter_cols = st.columns(2, gap="small")
    for idx, (chapter_key, chapter_label) in enumerate(CHAPTERS):
        with chapter_cols[idx % 2]:
            if st.button(chapter_label, key=f"ch_{chapter_key}", use_container_width=True):
                st.session_state.selected_chapter = (
                    "" if st.session_state.selected_chapter == chapter_key else chapter_key
                )
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    selected_chapter = st.session_state.selected_chapter
    if selected_chapter:
        selected_label = dict(CHAPTERS)[selected_chapter]
        st.markdown('<div style="margin-top:0.65rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="june-label">Stored memory</div>', unsafe_allow_html=True)
        st.markdown(f'<h3 class="june-title">{html.escape(selected_label)}</h3>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="june-subtitle">{html.escape(chapter_subtitle(selected_chapter))}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(render_list(chapter_items(memory, selected_chapter)), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Workspace (agent-pinned notes)
    st.markdown('<div class="june-panel-section">', unsafe_allow_html=True)
    workspace_placeholder = st.empty()
    workspace_placeholder.markdown(render_workspace(st.session_state.ui_state), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Capture health (collapsible)
    st.markdown('<div class="june-panel-section">', unsafe_allow_html=True)
    with st.expander("Capture health", expanded=False):
        st.markdown(render_capture_health(memory, st.session_state.activity_log), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Agent logs (collapsible)
    st.markdown('<div class="june-panel-section">', unsafe_allow_html=True)
    with st.expander("Agent logs", expanded=False):
        activity_placeholder = st.empty()
        activity_placeholder.markdown(render_activity(st.session_state.activity_log), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close june-right-panel

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
    append_activity(f"user | {prompt}")
    transcript_placeholder.markdown(
        transcript_html(st.session_state.messages, st.session_state.live_response),
        unsafe_allow_html=True,
    )

    # activity_placeholder may be inside a collapsed expander; guard against it
    try:
        activity_placeholder.markdown(render_activity(st.session_state.activity_log), unsafe_allow_html=True)
    except Exception:
        pass

    try:
        for mode, data in june_agent.stream(
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
            transcript_placeholder.markdown(transcript_html(st.session_state.messages), unsafe_allow_html=True)
            memory.save_message("assistant", final_text)

    st.session_state.is_generating = False
    st.rerun()
