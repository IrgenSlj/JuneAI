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

st.set_page_config(
    page_title="JuneAI",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    :root {
        --june-bg: #ffffff;
        --june-panel: rgba(255, 255, 255, 0.92);
        --june-text: #111111;
        --june-green: #1f6b3a;
        --june-green-soft: #eef8f0;
        --june-green-border: #d9ead9;
        --june-radius: 18px;
        --june-font: "Share Tech Mono", "IBM Plex Mono", "Menlo", monospace;
    }
    html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"],
    [data-testid="stText"], [data-testid="stButton"], [data-testid="stRadio"], [data-testid="stChatMessage"] {
        font-family: var(--june-font);
        color: var(--june-text);
    }
    [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container {
        background: var(--june-bg);
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1180px;
    }
    [data-testid="stSidebar"] {
        background: var(--june-bg);
    }
    [data-testid="stSidebar"] * {
        color: var(--june-text);
        font-family: var(--june-font);
    }
    [data-testid="stSidebar"] > div {
        background: var(--june-bg);
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: var(--june-green);
    }
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stChatInputTextArea"] textarea {
        background: var(--june-bg);
        color: var(--june-text);
        border: none;
        border-radius: 999px;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stChatInputTextArea"] textarea:focus {
        border-color: var(--june-green);
        box-shadow: 0 0 0 1px var(--june-green-soft);
    }
    .june-console, .june-workspace {
        background: var(--june-panel);
        border: none;
        border-radius: var(--june-radius);
        padding: 18px;
        color: var(--june-text);
        font-family: var(--june-font);
    }
    .june-console {
        min-height: 360px;
        font-size: 12px;
        line-height: 1.45;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }
    .june-console-header, .june-workspace-header {
        font-family: var(--june-font);
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 12px;
        color: var(--june-green);
        border-bottom: 1px solid rgba(31, 107, 58, 0.12);
        padding-bottom: 8px;
    }
    .june-workspace h3, .june-workspace h4, .june-workspace p, .june-workspace li,
    .june-console, .june-console * {
        color: var(--june-text);
    }
    .june-workspace h3 {
        margin-bottom: 0.7rem;
        font-size: 1.9rem;
    }
    .june-workspace h4 {
        margin-top: 1.5rem;
        margin-bottom: 0.45rem;
        color: var(--june-green);
    }
    .june-workspace ul {
        margin: 0;
        padding-left: 18px;
    }
    .june-chat-shell {
        max-width: 720px;
    }
    .june-transcript {
        height: 70vh;
        overflow-y: auto;
        padding-right: 1rem;
        mask-image: linear-gradient(to bottom, transparent 0%, black 18%, black 100%);
    }
    .june-message {
        margin-bottom: 1.25rem;
        color: var(--june-text);
        animation: june-rise 180ms ease;
    }
    .june-message-user {
        font-weight: 700;
        color: var(--june-green);
    }
    .june-message-assistant {
        font-weight: 400;
        color: var(--june-text);
    }
    .june-message-live {
        color: var(--june-green);
    }
    .june-input-wrap {
        margin-top: 1.2rem;
        background: transparent;
    }
    .june-writing {
        color: var(--june-green);
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 1rem;
    }
    @keyframes june-rise {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .june-drawer {
        position: fixed;
        right: 1rem;
        width: 380px;
        overflow-y: auto;
        padding: 0.25rem 0.25rem 1rem 0.25rem;
        background: transparent;
        z-index: 30;
        transform: translateX(calc(100% + 2rem));
        transition: transform 180ms ease;
    }
    .june-drawer.open {
        transform: translateX(0);
    }
    .june-workspace-drawer {
        top: 4.5rem;
        max-height: calc(52vh - 3rem);
    }
    .june-activity-drawer {
        bottom: 1rem;
        max-height: calc(42vh - 1rem);
    }
    .june-drawer-controls {
        position: fixed;
        top: 4.5rem;
        right: 1rem;
        width: 220px;
        z-index: 40;
    }
    button[kind="secondary"], button[kind="primary"], .stButton > button {
        background: var(--june-bg);
        color: var(--june-text);
        border: none;
        border-radius: 999px;
        transition: border-color 140ms ease, color 140ms ease, background 140ms ease;
    }
    button[kind="secondary"]:hover, button[kind="primary"]:hover, .stButton > button:hover {
        color: var(--june-green);
        background: var(--june-green-soft);
    }
    button[kind="secondary"]:focus, button[kind="primary"]:focus, .stButton > button:focus {
        box-shadow: 0 0 0 1px var(--june-green-soft);
    }
    [data-testid="stRadio"] label {
        color: var(--june-text);
    }
    [data-testid="stRadio"] input:checked + div,
    [data-testid="stRadio"] input:checked + div p {
        color: var(--june-green);
    }
    [data-testid="stRadio"] svg {
        color: var(--june-green);
    }
    [data-testid="stChatInput"] {
        background: transparent;
    }
    hr, [data-testid="stSidebar"] hr {
        border: none;
        border-top: 1px solid rgba(31, 107, 58, 0.12);
    }
    .june-console-floating {
        background: transparent;
        backdrop-filter: blur(6px);
        mask-image: linear-gradient(to bottom, transparent 0%, black 16%, black 100%);
    }
    .june-input-wrap [data-testid="stForm"] {
        background: transparent;
        border: none;
        padding: 0;
    }
    .june-input-wrap input {
        color: var(--june-green);
        font-weight: 700;
    }
    .june-input-wrap input::placeholder {
        color: rgba(31, 107, 58, 0.45);
    }
    .june-brand {
        color: var(--june-green);
        font-size: 2.2rem;
        line-height: 1;
        margin: 0 0 0.8rem 0;
        letter-spacing: 0.02em;
    }
    .june-flashline {
        position: relative;
        color: rgba(31, 107, 58, 0.34);
        overflow: hidden;
        display: inline-block;
        line-height: 1.5;
        margin-bottom: 1.4rem;
    }
    .june-flashline::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(
            110deg,
            rgba(255,255,255,0) 0%,
            rgba(255,255,255,0) 38%,
            rgba(31,107,58,0.18) 48%,
            rgba(255,255,255,0) 58%,
            rgba(255,255,255,0) 100%
        );
        transform: translateX(-130%);
        animation: june-flash 4.6s ease-in-out infinite;
    }
    @keyframes june-flash {
        0% { transform: translateX(-130%); }
        45% { transform: translateX(-130%); }
        70% { transform: translateX(130%); }
        100% { transform: translateX(130%); }
    }
    .june-side-label {
        color: var(--june-green);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 12px;
        margin-bottom: 0.4rem;
    }
    .june-skill-wheel {
        text-align: center;
        margin: 0.5rem 0 1.2rem 0;
    }
    .june-skill-current {
        color: var(--june-green);
        font-weight: 700;
        font-size: 1.1rem;
        line-height: 1.5;
        margin: 0.3rem 0;
    }
    .june-skill-neighbor {
        color: rgba(31, 107, 58, 0.34);
        font-size: 0.92rem;
        line-height: 1.3;
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


def workspace_drawer_class() -> str:
    """Return the CSS class for the workspace drawer."""
    base = "june-drawer june-workspace-drawer"
    if st.session_state.workspace_drawer_open:
        base += " open"
    return base


def activity_drawer_class() -> str:
    """Return the CSS class for the activity drawer."""
    base = "june-drawer june-activity-drawer"
    if st.session_state.activity_drawer_open:
        base += " open"
    return base


def render_workspace_drawer(container, ui_state: dict) -> None:
    """Render the workspace inside a fixed right-side drawer."""
    items = "".join(
        f"<li>{html.escape(item)}</li>"
        for item in ui_state.get("checklist_items", [])
    )
    drawer_html = (
        f'<div class="{workspace_drawer_class()}">'
        '<div class="june-workspace">'
        '<div class="june-workspace-header">Workspace / '
        f"{html.escape(ui_state.get('layout', 'split'))}"
        "</div>"
        f"<h3>{html.escape(ui_state.get('focus_title', 'Workspace'))}</h3>"
        f"<p>{html.escape(ui_state.get('focus_body', ''))}</p>"
        f"<h4>{html.escape(ui_state.get('checklist_title', 'Next steps'))}</h4>"
        f"<ul>{items}</ul>"
        f"<p><strong>Status:</strong> {html.escape(ui_state.get('notice', ''))}</p>"
        "</div>"
        "</div>"
    )
    container.markdown(drawer_html, unsafe_allow_html=True)


def render_activity_drawer(container, activity_log: list[str]) -> None:
    """Render the activity console inside a fixed right-side drawer."""
    lines = "\n".join(activity_log[-40:]) or "No activity yet."
    drawer_html = (
        f'<div class="{activity_drawer_class()}">'
        '<div class="june-console june-console-floating">'
        '<div class="june-console-header">Live Activity</div>'
        f"{html.escape(lines)}"
        "</div>"
        "</div>"
    )
    container.markdown(drawer_html, unsafe_allow_html=True)


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
                f"{html.escape(extract_text(msg.content))}"
                "</div>"
            )
        elif isinstance(msg, AIMessage) and msg.content:
            blocks.append(
                '<div class="june-message june-message-assistant">'
                f"{html.escape(extract_text(msg.content))}"
                "</div>"
            )
    if live_response:
        blocks.append(
            '<div class="june-message june-message-assistant june-message-live">'
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
                f"[chat] skill={event.get('skill')} messages={event.get('message_count')}"
            )
        elif event.get("event") == "tool_calls_requested":
            tools = ", ".join(event.get("tools", []))
            append_activity(f"[chat] tool calls requested: {tools}")
        elif event.get("event") == "response_completed":
            append_activity("[chat] direct response completed")
        render_activity_drawer(activity_placeholder, st.session_state.activity_log)
        return

    if mode == "messages":
        message, _metadata = data
        if isinstance(message, AIMessageChunk):
            token_text = extract_text(message.content)
            if token_text:
                st.session_state.live_response += token_text
                transcript_placeholder.markdown(
                    transcript_html(
                        st.session_state.messages,
                        st.session_state.live_response,
                    ),
                    unsafe_allow_html=True,
                )
            for chunk in getattr(message, "tool_call_chunks", []) or []:
                name = chunk.get("name")
                if name:
                    append_activity(f"[token] planning tool: {name}")
        render_activity_drawer(activity_placeholder, st.session_state.activity_log)
        return

    if mode == "updates":
        for node_name, payload in (data or {}).items():
            append_activity(f"[node] {node_name}")
            if isinstance(payload, dict):
                if "ui_state" in payload:
                    st.session_state.ui_state = payload["ui_state"]
                    if has_active_notes(st.session_state.ui_state):
                        render_workspace_drawer(
                            workspace_placeholder,
                            st.session_state.ui_state,
                        )
                    else:
                        workspace_placeholder.empty()
                    append_activity(
                        f"[ui] layout={st.session_state.ui_state.get('layout')} "
                        f"title={st.session_state.ui_state.get('focus_title')}"
                    )
                for message in payload.get("messages", []):
                    if isinstance(message, ToolMessage):
                        append_activity(
                            f"[tool] {message.name}: {extract_text(message.content)}"
                        )
                    elif isinstance(message, AIMessage):
                        for tool_call in getattr(message, "tool_calls", []) or []:
                            append_activity(
                                f"[tool-request] {tool_call.get('name')} "
                                f"args={tool_call.get('args')}"
                            )
        render_activity_drawer(activity_placeholder, st.session_state.activity_log)
        return

    if mode == "values" and isinstance(data, dict):
        st.session_state.final_state = data
        if "ui_state" in data:
            st.session_state.ui_state = data["ui_state"]
            if has_active_notes(st.session_state.ui_state):
                render_workspace_drawer(workspace_placeholder, st.session_state.ui_state)
            else:
                workspace_placeholder.empty()


with st.sidebar:
    skill_keys = list(SKILLS.keys())
    if "selected_skill_key" not in st.session_state:
        st.session_state.selected_skill_key = DEFAULT_SKILL

    st.markdown('<div class="june-brand">JuneAI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="june-flashline">Your companion for love,<br>life and growth</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="june-side-label">You are:</div>', unsafe_allow_html=True)

    user_id = st.text_input(
        "You are:",
        value="friend",
        key="user_id_input",
        label_visibility="collapsed",
    )

    current_skill_index = skill_keys.index(st.session_state.selected_skill_key)
    prev_skill = SKILLS[skill_keys[(current_skill_index - 1) % len(skill_keys)]].label
    current_skill = SKILLS[skill_keys[current_skill_index]].label
    next_skill = SKILLS[skill_keys[(current_skill_index + 1) % len(skill_keys)]].label

    st.markdown(
        '<div class="june-side-label">What do you need?</div>',
        unsafe_allow_html=True,
    )
    skill_left, skill_mid, skill_right = st.columns([0.18, 0.64, 0.18], gap="small")
    with skill_left:
        if st.button("<", use_container_width=True):
            st.session_state.selected_skill_key = skill_keys[
                (current_skill_index - 1) % len(skill_keys)
            ]
            st.rerun()
    with skill_mid:
        st.markdown(
            (
                '<div class="june-skill-wheel">'
                f'<div class="june-skill-neighbor">{html.escape(prev_skill)}</div>'
                f'<div class="june-skill-current">{html.escape(current_skill)}</div>'
                f'<div class="june-skill-neighbor">{html.escape(next_skill)}</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with skill_right:
        if st.button(">", use_container_width=True):
            st.session_state.selected_skill_key = skill_keys[
                (current_skill_index + 1) % len(skill_keys)
            ]
            st.rerun()

    selected_skill = st.session_state.selected_skill_key

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.activity_log = []
        st.session_state.ui_state = default_ui_state()
        st.rerun()

    mem_preview = Memory(user_id)
    history = mem_preview.get_mood_history(5)
    if history:
        st.caption("Recent moods")
        for mood in reversed(history):
            st.write(f"**{mood['timestamp'][:10]}** - {mood['mood']}")
            if mood.get("note"):
                st.caption(mood["note"])

    loops = mem_preview.get_open_loops(limit=5)
    if loops:
        st.caption("Open loops")
        for loop in reversed(loops):
            text = loop["topic"]
            if loop.get("next_step"):
                text += f" | Next: {loop['next_step']}"
            st.write(text)

skill = SKILLS[selected_skill]

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

if "workspace_drawer_open" not in st.session_state:
    st.session_state.workspace_drawer_open = True

if "activity_drawer_open" not in st.session_state:
    st.session_state.activity_drawer_open = True

if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = ""

if st.session_state.last_user_id != user_id:
    st.session_state.messages = Memory(user_id).load_chat_messages()
    st.session_state.activity_log = []
    st.session_state.ui_state = default_ui_state()
    st.session_state.live_response = ""
    st.session_state.pending_prompt = ""
    st.session_state.is_generating = False
    st.session_state.last_user_id = user_id

memory = Memory(user_id)

if st.session_state.last_skill != selected_skill:
    st.session_state.last_skill = selected_skill

control_spacer, control_col = st.columns([0.78, 0.22], gap="small")
with control_col:
    workspace_label = (
        "Hide Workspace" if st.session_state.workspace_drawer_open else "Show Workspace"
    )
    activity_label = (
        "Hide Activity" if st.session_state.activity_drawer_open else "Show Activity"
    )
    if st.button(workspace_label, use_container_width=True):
        st.session_state.workspace_drawer_open = not st.session_state.workspace_drawer_open
        st.rerun()
    if st.button(activity_label, use_container_width=True):
        st.session_state.activity_drawer_open = not st.session_state.activity_drawer_open
        st.rerun()

st.markdown('<div class="june-chat-shell">', unsafe_allow_html=True)
transcript_placeholder = st.empty()
transcript_placeholder.markdown(
    transcript_html(st.session_state.messages, st.session_state.live_response),
    unsafe_allow_html=True,
)
if not st.session_state.is_generating:
    st.markdown('<div class="june-input-wrap">', unsafe_allow_html=True)
    with st.form("june_input_form", clear_on_submit=True):
        prompt = st.text_input(
            "Message June",
            value="",
            placeholder=skill.hint,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if submitted and prompt.strip():
        st.session_state.pending_prompt = prompt.strip()
        st.session_state.is_generating = True
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

workspace_placeholder = st.empty()
if has_active_notes(st.session_state.ui_state):
    render_workspace_drawer(workspace_placeholder, st.session_state.ui_state)
else:
    workspace_placeholder.empty()

activity_placeholder = st.empty()
render_activity_drawer(activity_placeholder, st.session_state.activity_log)

if st.session_state.is_generating:
    st.markdown('<div class="june-writing">June is writing</div>', unsafe_allow_html=True)

if st.session_state.is_generating and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    memory.save_message("user", prompt)
    st.session_state.pending_prompt = ""
    st.session_state.live_response = ""
    st.session_state.final_state = None
    append_activity(f"[user] {prompt}")
    transcript_placeholder.markdown(
        transcript_html(st.session_state.messages, st.session_state.live_response),
        unsafe_allow_html=True,
    )
    render_activity_drawer(activity_placeholder, st.session_state.activity_log)

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
    except Exception as e:
        st.session_state.is_generating = False
        st.error(f"June ran into an issue: {e}")
        st.stop()

    result = st.session_state.final_state
    if result:
        response = next(
            (
                m for m in reversed(result["messages"])
                if isinstance(m, AIMessage) and m.content
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
