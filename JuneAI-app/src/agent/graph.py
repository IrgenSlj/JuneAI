"""JuneAI LangGraph agent."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .config import LLM_API_KEY, LLM_BASE_URL, MODEL_NAME
from .skills import DEFAULT_SKILL, build_system_prompt
from .tools import JUNE_TOOLS


class AgentState(TypedDict):
    """The state passed between every node in the graph.

    'messages' uses operator.add so each node appends to the list
    rather than replacing it. This keeps the full conversation history.
    """

    messages: Annotated[list, operator.add]
    user_id: str
    skill: str


def create_june_agent():
    """Build and compile the JuneAI LangGraph agent."""

    # LLM client configured through an OpenAI-compatible API.
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=LLM_API_KEY,
        openai_api_base=LLM_BASE_URL,
        temperature=0.8,
    ).bind_tools(JUNE_TOOLS)

    tool_node = ToolNode(JUNE_TOOLS)

    def chat(state: AgentState) -> dict:
        """Main chat node."""
        skill = state.get("skill", DEFAULT_SKILL)
        prompt = build_system_prompt(skill)
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("chat", chat)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("chat")

    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")

    return graph.compile()


# Single compiled agent instance reused across sessions
june_agent = create_june_agent()
