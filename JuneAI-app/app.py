"""JuneAI — Streamlit frontend.

Run with:  streamlit run app.py
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.graph import june_agent
from src.agent.memory import Memory
from src.agent.tools import set_memory

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JuneAI",
    page_icon="🌸",
    layout="centered",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌸 JuneAI")
    st.caption("Your companion for love, life & growth")
    st.divider()

    user_id = st.text_input(
        "Your name (used to save your memory)",
        value="friend",
        key="user_id_input",
    )

    mode = st.radio(
        "What do you need today?",
        ["💬 Friend & Therapist", "💘 Dating Coach", "📓 Mood Tracker"],
        index=0,
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Show mood history in sidebar when in Mood Tracker mode
    if mode == "📓 Mood Tracker":
        mem_preview = Memory(user_id)
        history = mem_preview.get_mood_history(10)
        if history:
            st.subheader("Your recent moods")
            for m in reversed(history[-5:]):
                st.write(f"**{m['timestamp'][:10]}** — {m['mood']}")
                if m.get("note"):
                    st.caption(m["note"])
        else:
            st.caption("No moods logged yet. Start chatting!")

# ── Main area ─────────────────────────────────────────────────────────────────
mode_config = {
    "💬 Friend & Therapist": {
        "intro": "Hey, I'm June 🌸 I'm here to listen, support, and help you grow. What's on your mind?",
        "hint": "Share what's on your mind...",
    },
    "💘 Dating Coach": {
        "intro": "Hey, I'm June 🌸 Let's talk love! I can help with compatibility, profiles, or figuring out what you really want.",
        "hint": "Ask about compatibility, conversation starters, dating advice...",
    },
    "📓 Mood Tracker": {
        "intro": "Hey, I'm June 🌸 Tell me how you're feeling — I'll help you track and understand your emotional journey.",
        "hint": "How are you feeling today?",
    },
}

config = mode_config[mode]

st.markdown(f"### {config['intro']}")
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_user_id" not in st.session_state:
    st.session_state.last_user_id = user_id

# Reset chat if user changes their name
if st.session_state.last_user_id != user_id:
    st.session_state.messages = []
    st.session_state.last_user_id = user_id

# Inject memory into tools for this user
memory = Memory(user_id)
set_memory(memory)

# ── Display chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant", avatar="🌸"):
            st.write(msg.content)

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input(config["hint"]):
    # Display user message immediately
    with st.chat_message("user"):
        st.write(prompt)

    # Add to state
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    memory.save_message("user", prompt)

    # Run the agent
    with st.chat_message("assistant", avatar="🌸"):
        with st.spinner("June is thinking..."):
            result = june_agent.invoke({
                "messages": st.session_state.messages,
                "user_id": user_id,
            })

        # Extract the last AI message that has actual text content
        # (tool call messages have no content, so we skip them)
        response = next(
            (
                m for m in reversed(result["messages"])
                if isinstance(m, AIMessage) and m.content
            ),
            None,
        )

        if response:
            st.write(response.content)
            st.session_state.messages = result["messages"]
            memory.save_message("assistant", response.content)
