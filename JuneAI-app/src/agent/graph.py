"""JuneAI LangGraph agent.

How it works (the ReAct loop):
  1. User sends a message
  2. 'chat' node: Gemini reads the conversation + system prompt and decides:
     a. Reply directly  →  END
     b. Call a tool     →  'tools' node
  3. 'tools' node: executes the tool, returns result as a ToolMessage
  4. Back to 'chat' node: Gemini reads the tool result and replies
  5. Loop continues until Gemini stops calling tools

This is the standard agent pattern — simple, powerful, easy to extend.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .config import GEMINI_API_KEY, MODEL_NAME
from .prompts import JUNE_SYSTEM_PROMPT
from .tools import JUNE_TOOLS


class AgentState(TypedDict):
    """The state passed between every node in the graph.

    'messages' uses operator.add so each node appends to the list
    rather than replacing it — this keeps the full conversation history.
    """

    messages: Annotated[list, operator.add]
    user_id: str


def create_june_agent():
    """Build and compile the JuneAI LangGraph agent."""

    # Gemini with tools bound — it will decide when to call them
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0.8,  # Slightly creative for warmth
    ).bind_tools(JUNE_TOOLS)

    # ToolNode handles executing whichever tool Gemini chose
    tool_node = ToolNode(JUNE_TOOLS)

    def chat(state: AgentState) -> dict:
        """Main chat node — prepends system prompt and calls Gemini."""
        messages = [SystemMessage(content=JUNE_SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("chat", chat)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("chat")

    # tools_condition: if Gemini returned tool_calls → go to tools, else END
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")  # After tool runs, back to chat

    return graph.compile()


# Single compiled agent instance — reused across sessions
june_agent = create_june_agent()
