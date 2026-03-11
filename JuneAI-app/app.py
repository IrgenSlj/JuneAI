"""JuneAI Streamlit frontend.

Run with: streamlit run app.py
"""

from __future__ import annotations

import html
from datetime import datetime

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from src.agent.graph import june_agent
from src.agent.memory import Memory
from src.agent.skills import DEFAULT_SKILL, SKILLS, infer_skill_from_text
from src.agent.tools import DEFAULT_UI_STATE

st.set_page_config(page_title="June", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --june-text: #161410;
        --june-muted: #6c655d;
        --june-line: rgba(22, 20, 16, 0.08);
        --june-accent: #0f5f4a;
        --june-accent-soft: rgba(15, 95, 74, 0.08);
        --june-user: rgba(15, 95, 74, 0.08);
        --june-panel: #ffffff;
        --june-radius: 24px;
        --june-shadow: 0 14px 40px rgba(40, 28, 18, 0.05);
    }

    html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"] {
        font-family: "IBM Plex Mono", monospace;
        color: var(--june-text);
    }

    [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container {
        background: #ffffff;
    }

    .block-container {
        max-width: 1360px;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--june-line);
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: #ffffff;
        border: 1px solid var(--june-line);
        border-radius: 18px;
        color: var(--june-text);
    }

    .stButton > button, button[kind="primary"], button[kind="secondary"] {
        border-radius: 16px;
        border: 1px solid var(--june-line);
        background: #ffffff;
        color: var(--june-text);
        min-height: 4.8rem;
    }

    .stButton > button:hover, button[kind="primary"]:hover, button[kind="secondary"]:hover {
        border-color: rgba(15, 95, 74, 0.24);
        color: var(--june-accent);
        background: var(--june-accent-soft);
    }

    .june-brand {
        font-family: "Syne", sans-serif;
        letter-spacing: -0.04em;
        font-size: 2.8rem;
        line-height: 0.92;
        margin: 0;
    }

    .june-copy {
        color: var(--june-muted);
        font-size: 12px;
        line-height: 1.55;
    }

    .june-surface {
        background: var(--june-panel);
        border: 1px solid var(--june-line);
        border-radius: var(--june-radius);
        box-shadow: var(--june-shadow);
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .june-label {
        color: var(--june-accent);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 10px;
        margin-bottom: 0.45rem;
    }

    .june-title {
        font-family: "Syne", sans-serif;
        font-size: 1.4rem;
        letter-spacing: -0.03em;
        margin: 0 0 0.75rem 0;
    }

    .june-subtitle {
        color: var(--june-muted);
        font-size: 12px;
        margin-bottom: 0.75rem;
    }

    .june-meta-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 0.8rem;
    }

    .june-chip {
        border: 1px solid var(--june-line);
        border-radius: 999px;
        padding: 0.34rem 0.62rem;
        font-size: 11px;
        color: var(--june-muted);
        background: #ffffff;
    }

    .june-transcript {
        max-height: 68vh;
        overflow-y: auto;
        padding-right: 0.15rem;
    }

    .june-message {
        border: 1px solid var(--june-line);
        border-radius: 18px;
        padding: 0.9rem 0.95rem;
        margin-bottom: 0.75rem;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }

    .june-message-user {
        background: var(--june-user);
    }

    .june-message-assistant {
        background: #ffffff;
    }

    .june-message-label {
        color: var(--june-accent);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 10px;
        margin-bottom: 0.35rem;
    }

    .june-writing {
        color: var(--june-accent);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 10px;
        margin-top: 0.6rem;
    }

    .june-list {
        display: grid;
        gap: 0.65rem;
    }

    .june-item {
        border: 1px solid var(--june-line);
        border-radius: 16px;
        padding: 0.7rem 0.8rem;
        background: #ffffff;
    }

    .june-item-title {
        font-family: "Syne", sans-serif;
        font-size: 1rem;
        margin-bottom: 0.15rem;
    }

    .june-item-meta {
        color: var(--june-muted);
        font-size: 12px;
        line-height: 1.5;
    }

    .june-chapter-grid .stButton > button {
        width: 100%;
        min-height: 5.4rem;
        border-radius: 18px;
        text-align: left;
        padding: 0.8rem;
    }

    .june-stat-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.6rem;
    }

    .june-stat-card {
        border: 1px solid var(--june-line);
        border-radius: 16px;
        padding: 0.75rem;
        background: #ffffff;
    }

    .june-stat-label {
        color: var(--june-muted);
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.35rem;
    }

    .june-stat-value {
        font-family: "Syne", sans-serif;
        font-size: 1.25rem;
        line-height: 1;
    }

    ul {
        margin: 0.5rem 0 0 1rem;
        padding: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CHAPTERS = [
    ("calendar", "Calendar"),
    ("gym", "Gym Schedule"),
    ("food", "Food Schedule"),
    ("trips", "Trips"),
    ("plans", "Plans"),
    ("dating", "Dating/Love"),
    ("family", "Family"),
    ("birthdays", "Birthdays"),
]


def default_ui_state() -> dict:
    """Return a fresh default UI state."""
    return {
        "layout": DEFAULT_UI_STATE["layout"],
        "focus_title": DEFAULT_UI_STATE["focus_title"],
        "focus_body": DEFAULT_UI_STATE["focus_body"],
        "checklist_title": DEFAULT_UI_STATE["checklist_title"],
        "checklist_items": list(DEFAULT_UI_STATE["checklist_items"]),
        "notice": DEFAULT_UI_STATE["notice"],
    }


def extract_text(content) -> str:
    """Extract plain text from LangChain message content blocks."""
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


def transcript_html(messages: list, live_response: str = "") -> str:
    """Render the transcript."""
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
        + """
        <script>
        const juneTranscript = window.parent.document.getElementById("june-transcript");
        if (juneTranscript) {
            juneTranscript.scrollTop = juneTranscript.scrollHeight;
        }
        </script>
        """
    )


def render_list(items: list[tuple[str, str]]) -> str:
    """Render a simple titled list."""
    if not items:
        return '<div class="june-item-meta">Nothing saved yet.</div>'
    return '<div class="june-list">' + "".join(
        (
            '<div class="june-item">'
            f'<div class="june-item-title">{html.escape(title)}</div>'
            f'<div class="june-item-meta">{html.escape(meta)}</div>'
            "</div>"
        )
        for title, meta in items
    ) + "</div>"


def render_workspace(ui_state: dict) -> str:
    """Render the workspace board."""
    checklist_items = ui_state.get("checklist_items", [])
    checklist = (
        "".join(f"<li>{html.escape(item)}</li>" for item in checklist_items)
        if checklist_items
        else "<li>No pinned actions.</li>"
    )
    return (
        '<div class="june-surface">'
        '<div class="june-label">Workspace</div>'
        f'<h3 class="june-title">{html.escape(ui_state.get("focus_title", "Workspace"))}</h3>'
        f'<div class="june-item-meta">{html.escape(ui_state.get("focus_body", ""))}</div>'
        f'<div class="june-label" style="margin-top:0.9rem;">{html.escape(ui_state.get("checklist_title", "Next steps"))}</div>'
        f'<div class="june-item-meta"><ul>{checklist}</ul></div>'
        f'<div class="june-item-meta" style="margin-top:0.75rem;">{html.escape(ui_state.get("notice", ""))}</div>'
        "</div>"
    )


def render_activity(activity_log: list[str]) -> str:
    """Render recent agent activity."""
    if not activity_log:
        return '<div class="june-item-meta">No activity yet.</div>'
    return '<div class="june-list">' + "".join(
        (
            '<div class="june-item">'
            f'<div class="june-item-meta">{html.escape(line)}</div>'
            "</div>"
        )
        for line in activity_log[-12:]
    ) + "</div>"


def append_activity(message: str) -> None:
    """Append one activity line."""
    st.session_state.activity_log.append(message)


def chapter_items(memory: Memory, chapter_key: str) -> list[tuple[str, str]]:
    """Return items for the selected chapter."""
    if chapter_key == "calendar":
        return [
            (
                item["title"],
                f"{item['date']}{' ' + item['time'] if item.get('time') else ''}{' | ' + item['details'] if item.get('details') else ''}",
            )
            for item in memory.get_calendar_items(limit=12)
        ]
    if chapter_key == "gym":
        return [
            (
                item["name"],
                f"{item['schedule']}{' | Goal: ' + item['goal'] if item.get('goal') else ''}",
            )
            for item in memory.get_gym_plans(limit=12)
        ]
    if chapter_key == "food":
        return [
            (
                item["name"],
                f"{item['daily_structure']}{' | Goal: ' + item['goal'] if item.get('goal') else ''}",
            )
            for item in memory.get_food_programs(limit=12)
        ]
    if chapter_key == "trips":
        return [
            (
                item["title"],
                f"{item['date']}{' | ' + item['details'] if item.get('details') else ''}",
            )
            for item in memory.get_calendar_items(limit=30)
            if "trip" in _calendar_text(item) or "travel" in _calendar_text(item) or "flight" in _calendar_text(item)
        ]
    if chapter_key == "plans":
        goal_items = [
            (
                item["title"],
                f"Goal | {item['status']}{' | Next: ' + item['next_step'] if item.get('next_step') else ''}",
            )
            for item in memory.get_goals(status="", limit=20)
        ]
        loop_items = [
            (
                item["topic"],
                f"Open loop | {item['status']}{' | Due: ' + item['due_date'] if item.get('due_date') else ''}",
            )
            for item in memory.get_open_loops(status="", limit=20)
        ]
        return goal_items + loop_items
    if chapter_key == "dating":
        return [
            (
                item["person"],
                f"{item['relationship']} | {item['summary']}",
            )
            for item in memory.get_relationship_profiles()
            if any(term in item.get("relationship", "").lower() for term in ["dating", "love", "partner", "girlfriend", "boyfriend", "romantic", "spouse"])
        ]
    if chapter_key == "family":
        return [
            (
                item["person"],
                f"{item['relationship']} | {item['summary']}",
            )
            for item in memory.get_relationship_profiles()
            if any(term in item.get("relationship", "").lower() for term in ["family", "mother", "father", "mom", "dad", "brother", "sister", "parent", "child", "cousin", "uncle", "aunt"])
        ]
    if chapter_key == "birthdays":
        return [
            (
                item["title"],
                f"{item['date']}{' | ' + item['details'] if item.get('details') else ''}",
            )
            for item in memory.get_calendar_items(limit=30)
            if "birthday" in _calendar_text(item)
        ]
    return []


def _calendar_text(item: dict) -> str:
    return " ".join(str(item.get(field, "")).lower() for field in ("title", "details", "source", "status"))


def chapter_subtitle(chapter_key: str) -> str:
    """Return the chapter subtitle."""
    subtitles = {
        "calendar": "Appointments, reminders, and captured events.",
        "gym": "Saved training routines and weekly splits.",
        "food": "Saved food structure and nutrition plans.",
        "trips": "Travel plans and movement-related events.",
        "plans": "Goals, open loops, and next steps.",
        "dating": "Relationship memory for love and dating.",
        "family": "Family context and important relationship notes.",
        "birthdays": "Birthday reminders and personal dates.",
    }
    return subtitles.get(chapter_key, "")


def render_notifications(memory: Memory) -> str:
    """Render local upcoming reminders."""
    notifications = memory.get_upcoming_notifications(limit=6)
    items = []
    for item in notifications:
        prefix = "today" if item["days_until"] == 0 else f"in {item['days_until']} days"
        items.append(
            (
                item["title"],
                f"{item['kind']} | {item['when']} | {prefix}{' | ' + item['details'] if item.get('details') else ''}",
            )
        )
    return render_list(items)


def render_capture_health(memory: Memory, activity_log: list[str]) -> str:
    """Render storage coverage and recent save-tool activity."""
    chapter_counts = {
        "Agenda": len(memory.get_calendar_items(limit=100)),
        "Birthdays": len(chapter_items(memory, "birthdays")),
        "Trips": len(chapter_items(memory, "trips")),
        "Gym": len(memory.get_gym_plans(limit=100)),
        "Food": len(memory.get_food_programs(limit=100)),
        "Plans": len(memory.get_goals(status="", limit=100)) + len(memory.get_open_loops(status="", limit=100)),
        "Dating": len(chapter_items(memory, "dating")),
        "Family": len(chapter_items(memory, "family")),
    }
    cards = "".join(
        (
            '<div class="june-stat-card">'
            f'<div class="june-stat-label">{html.escape(label)}</div>'
            f'<div class="june-stat-value">{count}</div>'
            "</div>"
        )
        for label, count in chapter_counts.items()
    )
    recent_save_logs = [
        line for line in activity_log[-30:]
        if "save_" in line or "Saved " in line or "saved " in line
    ][-6:]
    save_log_html = (
        render_list([("Recent capture", line) for line in recent_save_logs])
        if recent_save_logs
        else '<div class="june-item-meta">No recent save-tool activity in this session.</div>'
    )
    return (
        f'<div class="june-stat-grid">{cards}</div>'
        '<div style="margin-top:0.9rem;"></div>'
        + save_log_html
    )


def current_local_time() -> datetime:
    """Return the current local timestamp with timezone."""
    return datetime.now().astimezone()


def current_part_of_day(now: datetime) -> str:
    """Return the natural language part of day."""
    if 5 <= now.hour < 12:
        return "morning"
    if 12 <= now.hour < 17:
        return "afternoon"
    if 17 <= now.hour < 22:
        return "evening"
    return "night"


def phrase_bucket(now: datetime) -> str:
    """Return the 15-minute refresh bucket key."""
    return f"{now.strftime('%Y-%m-%d')}-{now.hour:02d}-{now.minute // 15}"


def generate_sidebar_phrase(now: datetime) -> str:
    """Generate an original philosophy-inspired health phrase offline."""
    openings = [
        "Health begins with gentle attention.",
        "A good life grows from small kindnesses to the body.",
        "Energy is built in quiet disciplines.",
        "Strength lasts longer when it is renewed by care.",
        "Well-being lives between motion, rest, and meaning.",
        "A clear day starts with one honest choice.",
    ]
    middles = [
        "Let today favor steady movement.",
        "Protect your peace like a serious habit.",
        "Feed the body with rhythm.",
        "Choose what leaves you lighter by tonight.",
        "Walk toward what strengthens you quietly.",
        "Keep your routines human and durable.",
    ]

    seed = now.timetuple().tm_yday + (now.hour * 4) + (now.minute // 15)
    opening = openings[seed % len(openings)]
    middle = middles[(seed * 3 + now.weekday()) % len(middles)]
    return f"{opening} {middle}"


def get_rotating_sidebar_phrase(memory: Memory, now: datetime) -> str:
    """Get or refresh the sidebar phrase every 15 minutes."""
    bucket = phrase_bucket(now)
    state = memory.get_app_state()
    if state.get("sidebar_phrase_bucket") != bucket:
        phrase = generate_sidebar_phrase(now)
        memory.set_app_state_value("sidebar_phrase_text", phrase)
        memory.set_app_state_value("sidebar_phrase_bucket", bucket)
        return phrase
    return state.get("sidebar_phrase_text", generate_sidebar_phrase(now))


def build_daily_checkin(memory: Memory) -> str:
    """Build June's proactive daily opening message."""
    now = current_local_time()
    opening = {
        "morning": "Good morning.",
        "afternoon": "Good afternoon.",
        "evening": "Good evening.",
        "night": "Good evening.",
    }[current_part_of_day(now)]
    notifications = memory.get_upcoming_notifications(limit=3)
    lines = [
        f"{opening} It's {now.strftime('%A')}, day {now.timetuple().tm_yday} of the year.",
        "How is your day going, what are your plans, and how are you feeling?",
    ]
    if notifications:
        reminder_parts = []
        for item in notifications:
            prefix = "today" if item["days_until"] == 0 else f"in {item['days_until']} days"
            reminder_parts.append(f"{item['title']} ({prefix})")
        lines.append("Important reminders: " + ", ".join(reminder_parts) + ".")
    return "\n\n".join(lines)


def handle_stream_chunk(
    mode: str,
    data,
    transcript_placeholder,
    workspace_placeholder,
    activity_placeholder,
) -> None:
    """Process one streamed graph event."""
    if mode == "custom":
        event = data or {}
        if event.get("event") == "chat_started":
            append_activity(f"route | {event.get('skill')}")
        elif event.get("event") == "tool_calls_requested":
            append_activity("tool request | " + ", ".join(event.get("tools", [])))
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
                        for tool_call in getattr(message, "tool_calls", []) or []:
                            append_activity(f"tool args | {tool_call.get('name')} {tool_call.get('args')}")
        activity_placeholder.markdown(render_activity(st.session_state.activity_log), unsafe_allow_html=True)
        return

    if mode == "values" and isinstance(data, dict):
        st.session_state.final_state = data
        if "ui_state" in data:
            st.session_state.ui_state = data["ui_state"]
            workspace_placeholder.markdown(
                render_workspace(st.session_state.ui_state),
                unsafe_allow_html=True,
            )


with st.sidebar:
    now = current_local_time()
    st.markdown('<div class="june-brand">June</div>', unsafe_allow_html=True)
    st.markdown(
        '<script>setTimeout(function(){ window.parent.location.reload(); }, 900000);</script>',
        unsafe_allow_html=True,
    )
    user_id = st.session_state.get("profile_input", "admin")
    memory_for_sidebar = Memory(user_id)
    sidebar_phrase = get_rotating_sidebar_phrase(memory_for_sidebar, now)
    st.markdown(
        f'<div class="june-copy">{html.escape(sidebar_phrase)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"{now.strftime('%A %d %B')} • {current_part_of_day(now)} • day {now.timetuple().tm_yday}")
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
        st.rerun()


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
    st.session_state.last_user_id = user_id

memory = Memory(user_id)

if not st.session_state.is_generating and memory.should_send_daily_checkin():
    daily_message = build_daily_checkin(memory)
    st.session_state.messages.append(AIMessage(content=daily_message))
    memory.save_message("assistant", daily_message)
    memory.mark_daily_checkin_sent()
    append_activity("daily check-in | sent")

snapshot = memory.get_progress_snapshot()
active_skill = SKILLS.get(st.session_state.active_skill_key, SKILLS[DEFAULT_SKILL])

left_col, right_col = st.columns([1.7, 1.1], gap="large")

with left_col:
    st.markdown('<div class="june-surface">', unsafe_allow_html=True)
    st.markdown('<div class="june-label">Conversation</div>', unsafe_allow_html=True)
    st.markdown(
        (
            '<div class="june-meta-row">'
            f'<div class="june-chip">profile: {html.escape(user_id)}</div>'
            f'<div class="june-chip">route: {html.escape(active_skill.label)}</div>'
            f'<div class="june-chip">agenda: {snapshot["calendar_count"]}</div>'
            f'<div class="june-chip">plans: {snapshot["goal_count"] + snapshot["open_loop_count"]}</div>'
            "</div>"
        ),
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
        inferred_skill = infer_skill_from_text(prompt)
        st.session_state.pending_prompt = prompt.strip()
        st.session_state.active_skill_key = inferred_skill
        st.session_state.is_generating = True
        append_activity(f"auto route | {inferred_skill}")
        st.rerun()

with right_col:
    st.markdown('<div class="june-surface">', unsafe_allow_html=True)
    st.markdown('<div class="june-label">Notifications</div>', unsafe_allow_html=True)
    st.markdown('<h3 class="june-title">Upcoming reminders</h3>', unsafe_allow_html=True)
    st.markdown(render_notifications(memory), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="june-surface">', unsafe_allow_html=True)
    st.markdown('<div class="june-label">Chapters</div>', unsafe_allow_html=True)
    st.markdown('<div class="june-chapter-grid">', unsafe_allow_html=True)
    chapter_columns = st.columns(2, gap="small")
    for index, (chapter_key, chapter_label) in enumerate(CHAPTERS):
        with chapter_columns[index % 2]:
            if st.button(chapter_label, key=f"chapter_{chapter_key}", use_container_width=True):
                if st.session_state.selected_chapter == chapter_key:
                    st.session_state.selected_chapter = ""
                else:
                    st.session_state.selected_chapter = chapter_key
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    selected_chapter = st.session_state.selected_chapter
    if selected_chapter:
        selected_label = dict(CHAPTERS)[selected_chapter]
        st.markdown('<div style="margin-top:0.9rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="june-label">Stored memory</div>', unsafe_allow_html=True)
        st.markdown(f'<h3 class="june-title">{html.escape(selected_label)}</h3>', unsafe_allow_html=True)
        st.markdown(f'<div class="june-subtitle">{html.escape(chapter_subtitle(selected_chapter))}</div>', unsafe_allow_html=True)
        st.markdown(render_list(chapter_items(memory, selected_chapter)), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="june-surface">', unsafe_allow_html=True)
    st.markdown('<div class="june-label">Capture health</div>', unsafe_allow_html=True)
    st.markdown('<h3 class="june-title">Stored by chapter</h3>', unsafe_allow_html=True)
    st.markdown(render_capture_health(memory, st.session_state.activity_log), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    workspace_placeholder = st.empty()
    workspace_placeholder.markdown(render_workspace(st.session_state.ui_state), unsafe_allow_html=True)

    activity_placeholder = st.empty()
    st.markdown('<div class="june-surface">', unsafe_allow_html=True)
    st.markdown('<div class="june-label">Logs</div>', unsafe_allow_html=True)
    st.markdown('<h3 class="june-title">Tool activity</h3>', unsafe_allow_html=True)
    activity_placeholder.markdown(render_activity(st.session_state.activity_log), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

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
    activity_placeholder.markdown(render_activity(st.session_state.activity_log), unsafe_allow_html=True)

    try:
        for mode, data in june_agent.stream(
            {
                "messages": st.session_state.messages,
                "user_id": user_id,
                "skill": st.session_state.active_skill_key,
                "ui_state": st.session_state.ui_state,
            },
            stream_mode=["messages", "updates", "custom", "values"],
        ):
            handle_stream_chunk(
                mode,
                data,
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
            (
                message for message in reversed(result["messages"])
                if isinstance(message, AIMessage) and message.content
            ),
            None,
        )
        if response:
            final_text = extract_text(response.content)
            st.session_state.messages = result["messages"]
            st.session_state.live_response = ""
            transcript_placeholder.markdown(
                transcript_html(st.session_state.messages),
                unsafe_allow_html=True,
            )
            memory.save_message("assistant", final_text)

    st.session_state.is_generating = False
    st.rerun()
