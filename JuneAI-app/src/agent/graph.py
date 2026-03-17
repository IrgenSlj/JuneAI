"""JuneAI LangGraph agent."""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import StreamWriter

from .config import RuntimeConfig, resolve_runtime_config
from .models import build_chat_model
from .skills import DEFAULT_SKILL, build_system_prompt
from .tools import JUNE_TOOLS


class AgentState(TypedDict):
    """The state passed between every node in the graph."""

    messages: Annotated[list, operator.add]
    user_id: str
    skill: str
    ui_state: dict
    tool_stats: dict


def _extract_text(content) -> str:
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


def _summarize_tool_messages(tool_messages: list[ToolMessage], requested_calls: list[dict], existing: dict | None) -> dict:
    existing = existing or {}
    by_id = {message.tool_call_id: message for message in tool_messages}
    call_results = []
    success_count = 0
    error_count = 0

    for call in requested_calls:
        message = by_id.get(call.get("id", ""))
        status = getattr(message, "status", "error") if message else "error"
        preview = _extract_text(message.content)[:160] if message else "No tool message returned."
        call_results.append(
            {
                "id": call.get("id", ""),
                "name": call.get("name", ""),
                "status": status,
                "preview": preview,
            }
        )
        if status == "success":
            success_count += 1
        else:
            error_count += 1

    return {
        "requested": existing.get("requested", 0) + len(requested_calls),
        "succeeded": existing.get("succeeded", 0) + success_count,
        "failed": existing.get("failed", 0) + error_count,
        "last_calls": call_results,
    }


def create_june_agent(llm=None, runtime: RuntimeConfig | None = None):
    """Build and compile the JuneAI LangGraph agent."""

    runtime = runtime or resolve_runtime_config()
    llm = llm or build_chat_model(runtime)
    if hasattr(llm, "bind_tools"):
        llm = llm.bind_tools(JUNE_TOOLS)

    tool_node = ToolNode(JUNE_TOOLS, handle_tool_errors=True)

    def chat(state: AgentState, writer: StreamWriter) -> dict:
        """Main chat node."""
        skill = state.get("skill", DEFAULT_SKILL)
        now = datetime.now().astimezone()
        prompt = build_system_prompt(skill, now=now, runtime=runtime)
        writer(
            {
                "event": "chat_started",
                "skill": skill,
                "message_count": len(state["messages"]),
                "local_time": now.strftime("%Y-%m-%d %H:%M"),
                "runtime_label": runtime.label,
                "runtime_provider": runtime.provider,
                "runtime_model": runtime.model,
            }
        )
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = llm.invoke(messages)
        if getattr(response, "tool_calls", None):
            writer(
                {
                    "event": "tool_calls_requested",
                    "tools": [call["name"] for call in response.tool_calls],
                }
            )
        else:
            writer({"event": "response_completed"})
        return {"messages": [response]}

    def run_tools(state: AgentState, writer: StreamWriter) -> dict:
        """Execute tools and report structured diagnostics."""
        last_message = state["messages"][-1] if state.get("messages") else None
        requested_calls = list(getattr(last_message, "tool_calls", []) or [])
        result = tool_node.invoke(state)
        tool_messages = [
            message for message in result.get("messages", []) if isinstance(message, ToolMessage)
        ]
        tool_stats = _summarize_tool_messages(tool_messages, requested_calls, state.get("tool_stats"))
        writer({"event": "tool_results", "calls": tool_stats["last_calls"], "summary": tool_stats})
        result["tool_stats"] = tool_stats
        return result

    graph = StateGraph(AgentState)
    graph.add_node("chat", chat)
    graph.add_node("tools", run_tools)

    graph.set_entry_point("chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")

    return graph.compile()


june_agent = create_june_agent()
