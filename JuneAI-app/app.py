"""JuneAI Streamlit frontend.

Run with:  streamlit run app.py
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.graph import june_agent
from src.agent.memory import Memory
from src.agent.skills import DEFAULT_SKILL, SKILLS

st.set_page_config(
    page_title="JuneAI",
    layout="centered",
)

with st.sidebar:
    st.title("JuneAI")
    st.caption("Your companion for love, life and growth")
    st.divider()

    user_id = st.text_input(
        "Your name (used to save your memory)",
        value="friend",
        key="user_id_input",
    )

    skill_labels = [skill.label for skill in SKILLS.values()]
    selected_label = st.radio(
        "What do you need today?",
        skill_labels,
        index=skill_labels.index(SKILLS[DEFAULT_SKILL].label),
    )
    selected_skill = next(
        key for key, skill in SKILLS.items() if skill.label == selected_label
    )

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    mem_preview = Memory(user_id)
    st.subheader("Saved context")

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

st.markdown(f"### {skill.intro}")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = Memory(user_id).load_chat_messages()

if "last_user_id" not in st.session_state:
    st.session_state.last_user_id = user_id

if "last_skill" not in st.session_state:
    st.session_state.last_skill = selected_skill

if st.session_state.last_user_id != user_id:
    st.session_state.messages = Memory(user_id).load_chat_messages()
    st.session_state.last_user_id = user_id

memory = Memory(user_id)

if st.session_state.last_skill != selected_skill:
    st.session_state.last_skill = selected_skill

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.write(msg.content)

if prompt := st.chat_input(skill.hint):
    with st.chat_message("user"):
        st.write(prompt)

    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    memory.save_message("user", prompt)

    with st.chat_message("assistant"):
        with st.spinner("June is thinking..."):
            try:
                result = june_agent.invoke({
                    "messages": st.session_state.messages,
                    "user_id": user_id,
                    "skill": selected_skill,
                })
            except Exception as e:
                st.error(f"June ran into an issue: {e}")
                st.stop()

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
