"""JuneAI Streamlit frontend.

Run with: streamlit run app.py
"""

from __future__ import annotations

import html

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from src.agent.graph import june_agent
from src.agent.memory import Memory
from src.agent.skills import DEFAULT_SKILL, SKILLS
from src.agent.tools import DEFAULT_UI_STATE

st.set_page_config(page_title="June", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --june-bg: #ffffff;
        --june-panel: rgba(255, 255, 255, 0.9);
        --june-panel-strong: rgba(255, 255, 255, 0.98);
        --june-text: #161410;
        --june-muted: #6e665d;
        --june-line: rgba(22, 20, 16, 0.08);
        --june-accent: #0f5f4a;
        --june-accent-soft: rgba(15, 95, 74, 0.1);
        --june-warm: #d96c43;
        --june-radius: 26px;
        --june-shadow: 0 18px 60px rgba(56, 38, 20, 0.08);
    }

    html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"] {
        font-family: "IBM Plex Mono", monospace;
        color: var(--june-text);
    }

    [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container {
        background: #ffffff;
    }

    .block-container {
        max-width: 1440px;
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
    }

    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.94);
        border-right: 1px solid var(--june-line);
    }

    [data-testid="stSidebar"] * {
        color: var(--june-text);
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: rgba(255, 255, 255, 0.55);
        border: 1px solid var(--june-line);
        border-radius: 18px;
        color: var(--june-text);
    }

    .stButton > button, button[kind="primary"], button[kind="secondary"] {
        border-radius: 999px;
        border: 1px solid var(--june-line);
        background: rgba(255, 255, 255, 0.6);
        color: var(--june-text);
        min-height: 2.5rem;
    }

    .stButton > button:hover, button[kind="primary"]:hover, button[kind="secondary"]:hover {
        border-color: rgba(15, 95, 74, 0.26);
        color: var(--june-accent);
        background: rgba(15, 95, 74, 0.06);
    }

    .june-shell {
        margin-bottom: 1rem;
    }

    .june-hero {
        background: rgba(255, 255, 255, 0.98);
        border: 1px solid rgba(22, 20, 16, 0.06);
        border-radius: 34px;
        box-shadow: var(--june-shadow);
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }

    .june-kicker, .june-panel-kicker {
        color: var(--june-accent);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 11px;
        margin-bottom: 0.55rem;
    }

    .june-title, .june-brand {
        font-family: "Syne", sans-serif;
        letter-spacing: -0.04em;
        line-height: 0.95;
        margin: 0;
    }

    .june-title {
        font-size: 3.6rem;
        margin-bottom: 0.8rem;
    }

    .june-brand {
        font-size: 2.7rem;
        margin-bottom: 0.6rem;
    }

    .june-subtitle, .june-meta {
        color: var(--june-muted);
        font-size: 13px;
        line-height: 1.6;
    }

    .june-surface {
        background: var(--june-panel);
        border: 1px solid rgba(22, 20, 16, 0.06);
        border-radius: var(--june-radius);
        box-shadow: var(--june-shadow);
        padding: 1rem 1rem 1.1rem 1rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1rem;
    }

    .june-surface-strong {
        background: var(--june-panel-strong);
    }

    .june-panel-title {
        font-family: "Syne", sans-serif;
        font-size: 1.4rem;
        margin: 0 0 0.9rem 0;
        letter-spacing: -0.03em;
    }

    .june-stat-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.7rem;
    }

    .june-stat {
        border: 1px solid rgba(22, 20, 16, 0.06);
        border-radius: 18px;
        padding: 0.8rem 0.9rem;
        background: rgba(255, 255, 255, 0.38);
    }

    .june-stat-label {
        color: var(--june-muted);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.35rem;
    }

    .june-stat-value {
        font-family: "Syne", sans-serif;
        font-size: 1.8rem;
        line-height: 1;
    }

    .june-list {
        display: grid;
        gap: 0.65rem;
    }

    .june-item {
        border-top: 1px solid var(--june-line);
        padding-top: 0.65rem;
    }

    .june-item:first-child {
        border-top: none;
        padding-top: 0;
    }

    .june-item-title {
        font-family: "Syne", sans-serif;
        font-size: 1.05rem;
        margin-bottom: 0.18rem;
    }

    .june-item-meta {
        color: var(--june-muted);
        font-size: 12px;
        line-height: 1.55;
    }

    .june-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }

    .june-chip {
        border-radius: 999px;
        padding: 0.38rem 0.68rem;
        background: rgba(15, 95, 74, 0.08);
        color: var(--june-accent);
        font-size: 12px;
        border: 1px solid rgba(15, 95, 74, 0.1);
    }

    .june-transcript {
        max-height: 60vh;
        overflow-y: auto;
        padding-right: 0.2rem;
    }

    .june-message {
        margin-bottom: 0.95rem;
        padding: 0.95rem 1rem;
        border-radius: 20px;
        border: 1px solid rgba(22, 20, 16, 0.05);
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }

    .june-message-user {
        background: rgba(217, 108, 67, 0.10);
    }

    .june-message-assistant {
        background: rgba(255, 255, 255, 0.52);
    }

    .june-message-label {
        color: var(--june-accent);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 10px;
        margin-bottom: 0.4rem;
    }

    .june-live {
        color: var(--june-accent);
    }

    .june-writing {
        color: var(--june-accent);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 11px;
        margin-top: 0.6rem;
    }

    hr, [data-testid="stSidebar"] hr {
        border: none;
        border-top: 1px solid var(--june-line);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def has_active_notes(ui_state: dict) -> bool:
    """Return whether the workspace contains meaningful model notes."""
    if ui_state.get("focus_title") not in {"", "Workspace"}:
        return True
    if ui_state.get("focus_body") not in {"", DEFAULT_UI_STATE["focus_body"]}:
        return True
    if ui_state.get("checklist_items"):
        return True
    if ui_state.get("notice"):
        return True
    return False


def transcript_html(messages: list, live_response: str = "") -> str:
    """Render the full transcript as a single styled surface."""
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
            '<div class="june-message june-message-assistant june-live">'
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


def append_activity(message: str) -> None:
    """Append an item to the live activity feed."""
    st.session_state.activity_log.append(message)


def render_stat_grid(snapshot: dict) -> str:
    """Render the snapshot tiles."""
    stats = [
        ("Calendar", snapshot["calendar_count"]),
        ("Favorites", snapshot["favorite_count"]),
        ("Goals", snapshot["goal_count"]),
        ("Plans", snapshot["gym_plan_count"] + snapshot["food_program_count"]),
    ]
    tiles = "".join(
        (
            '<div class="june-stat">'
            f'<div class="june-stat-label">{html.escape(label)}</div>'
            f'<div class="june-stat-value">{value}</div>'
            "</div>"
        )
        for label, value in stats
    )
    return f'<div class="june-stat-grid">{tiles}</div>'


def render_list(items: list[tuple[str, str]]) -> str:
    """Render a generic list of titled items."""
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


def render_activity(activity_log: list[str]) -> str:
    """Render the live activity feed."""
    if not activity_log:
        return '<div class="june-item-meta">No activity yet.</div>'
    return render_list([("Agent activity", line) for line in activity_log[-12:]])


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
            append_activity(
                f"skill={event.get('skill')} | messages={event.get('message_count')}"
            )
        elif event.get("event") == "tool_calls_requested":
            tools = ", ".join(event.get("tools", []))
            append_activity(f"tool calls requested | {tools}")
        elif event.get("event") == "response_completed":
            append_activity("direct response completed")
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
                    append_activity(f"planning tool | {name}")
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
                            append_activity(
                                f"tool request | {tool_call.get('name')} {tool_call.get('args')}"
                            )
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


def render_workspace(ui_state: dict) -> str:
    """Render the workspace board."""
    checklist_items = ui_state.get("checklist_items", [])
    checklist = (
        "".join(f"<li>{html.escape(item)}</li>" for item in checklist_items)
        if checklist_items
        else "<li>No pinned actions.</li>"
    )
    return (
        '<div class="june-surface june-surface-strong">'
        '<div class="june-panel-kicker">Workspace</div>'
        f'<h3 class="june-panel-title">{html.escape(ui_state.get("focus_title", "Workspace"))}</h3>'
        f'<div class="june-item-meta">{html.escape(ui_state.get("focus_body", ""))}</div>'
        f'<div class="june-panel-kicker" style="margin-top:1rem;">{html.escape(ui_state.get("checklist_title", "Next steps"))}</div>'
        f'<div class="june-item-meta"><ul>{checklist}</ul></div>'
        f'<div class="june-item-meta" style="margin-top:0.8rem;">{html.escape(ui_state.get("notice", ""))}</div>'
        "</div>"
    )


with st.sidebar:
    if "selected_skill_key" not in st.session_state:
        st.session_state.selected_skill_key = DEFAULT_SKILL

    st.markdown('<div class="june-brand">June</div>', unsafe_allow_html=True)
    st.markdown(
        '',
        unsafe_allow_html=True,
    )
    st.write("")
    user_id = st.text_input("Profile", value="admin")
    st.caption("Mode")

    skill_keys = list(SKILLS.keys())
    skill_cols = st.columns(2, gap="small")
    for index, key in enumerate(skill_keys):
        with skill_cols[index % 2]:
            if st.button(SKILLS[key].label, key=f"skill_{key}", use_container_width=True):
                st.session_state.selected_skill_key = key
                st.rerun()

    selected_skill = st.session_state.selected_skill_key
    skill = SKILLS[selected_skill]
    st.write("")
    st.caption(skill.sidebar_caption)
    st.markdown(f"**{skill.intro}**")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.activity_log = []
        st.session_state.ui_state = default_ui_state()
        st.session_state.live_response = ""
        st.session_state.final_state = None
        st.session_state.pending_prompt = ""
        st.session_state.is_generating = False
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = Memory(user_id).load_chat_messages()
if "last_user_id" not in st.session_state:
    st.session_state.last_user_id = user_id
if "last_skill" not in st.session_state:
    st.session_state.last_skill = selected_skill
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

if st.session_state.last_user_id != user_id:
    st.session_state.messages = Memory(user_id).load_chat_messages()
    st.session_state.activity_log = []
    st.session_state.ui_state = default_ui_state()
    st.session_state.live_response = ""
    st.session_state.final_state = None
    st.session_state.pending_prompt = ""
    st.session_state.is_generating = False
    st.session_state.last_user_id = user_id

if st.session_state.last_skill != selected_skill:
    st.session_state.last_skill = selected_skill

memory = Memory(user_id)
snapshot = memory.get_progress_snapshot()
preferences = memory.get_preferences(limit=8)
calendar_items = memory.get_calendar_items(limit=8)
favorites = memory.get_favorites(limit=8)
gym_plans = memory.get_gym_plans(limit=4)
food_programs = memory.get_food_programs(limit=4)

st.markdown('<div class="june-shell">', unsafe_allow_html=True)
st.markdown(
    (
        '<div class="june-hero">'
        '<div class="june-kicker">June</div>'
        '<h1 class="june-title">Your assistant.</h1>'
        f'<div class="june-subtitle">{html.escape(skill.hint)}</div>'
        "</div>"
    ),
    unsafe_allow_html=True,
)

left_col, center_col, right_col = st.columns([1.0, 1.45, 1.0], gap="large")

with left_col:
    st.markdown('<div class="june-surface">', unsafe_allow_html=True)
    st.markdown('<div class="june-panel-kicker">Snapshot</div>', unsafe_allow_html=True)
    st.markdown(render_stat_grid(snapshot), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    pref_chips = (
        '<div class="june-chip-row">'
        + "".join(
            f'<div class="june-chip">{html.escape(item["category"])}: {html.escape(item["value"])}</div>'
            for item in preferences
        )
        + "</div>"
        if preferences
        else '<div class="june-item-meta">No saved preferences yet.</div>'
    )
    st.markdown('<div class="june-surface">', unsafe_allow_html=True)
    st.markdown('<div class="june-panel-kicker">Taste and Preferences</div>', unsafe_allow_html=True)
    st.markdown('<h3 class="june-panel-title">What June has learned</h3>', unsafe_allow_html=True)
    st.markdown(pref_chips, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    routine_items = [
        (plan["name"], f"{plan['schedule']} | Goal: {plan['goal'] or 'not set'}")
        for plan in gym_plans
    ] + [
        (program["name"], f"{program['goal']} | {program['daily_structure']}")
        for program in food_programs
    ]
    st.markdown('<div class="june-surface">', unsafe_allow_html=True)
    st.markdown('<div class="june-panel-kicker">Routines</div>', unsafe_allow_html=True)
    st.markdown('<h3 class="june-panel-title">Gym and food programs</h3>', unsafe_allow_html=True)
    st.markdown(render_list(routine_items), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with center_col:
    transcript_placeholder = st.empty()
    st.markdown('<div class="june-surface june-surface-strong">', unsafe_allow_html=True)
    st.markdown('<div class="june-panel-kicker">Conversation</div>', unsafe_allow_html=True)
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
            placeholder=skill.hint,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if submitted and prompt.strip() and not st.session_state.is_generating:
        st.session_state.pending_prompt = prompt.strip()
        st.session_state.is_generating = True
        st.rerun()

with right_col:
    workspace_placeholder = st.empty()
    workspace_placeholder.markdown(render_workspace(st.session_state.ui_state), unsafe_allow_html=True)

    calendar_list = [
        (
            item["title"],
            f"{item['date']}{' ' + item['time'] if item.get('time') else ''} | {item.get('details') or item.get('status', '')}",
        )
        for item in calendar_items
    ]
    st.markdown('<div class="june-surface">', unsafe_allow_html=True)
    st.markdown('<div class="june-panel-kicker">Calendar</div>', unsafe_allow_html=True)
    st.markdown('<h3 class="june-panel-title">Upcoming and captured</h3>', unsafe_allow_html=True)
    st.markdown(render_list(calendar_list), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    favorite_list = [
        (
            item["title"],
            f"{item['category']}{' | ' + item['creator'] if item.get('creator') else ''}{' | ' + item['reason'] if item.get('reason') else ''}",
        )
        for item in favorites
    ]
    st.markdown('<div class="june-surface">', unsafe_allow_html=True)
    st.markdown('<div class="june-panel-kicker">Favorites</div>', unsafe_allow_html=True)
    st.markdown('<h3 class="june-panel-title">Books, films, and saved picks</h3>', unsafe_allow_html=True)
    st.markdown(render_list(favorite_list), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    activity_placeholder = st.empty()
    st.markdown('<div class="june-surface">', unsafe_allow_html=True)
    st.markdown('<div class="june-panel-kicker">Live Activity</div>', unsafe_allow_html=True)
    activity_placeholder.markdown(render_activity(st.session_state.activity_log), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

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
                "skill": selected_skill,
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
