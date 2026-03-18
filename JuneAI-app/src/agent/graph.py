"""JuneAI LangGraph agent."""

from __future__ import annotations

import json
import operator
from html import unescape
from json import JSONDecodeError
from datetime import datetime
from typing import Annotated, TypedDict
from uuid import uuid4

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


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_json_payload(text: str):
    """Extract the outermost JSON object or array from free-form model text."""
    cleaned = unescape(_strip_code_fence(text)).strip()
    if not cleaned:
        return None

    pairs = []
    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")
    if object_start != -1 and object_end != -1 and object_end > object_start:
        pairs.append((object_start, object_end))
    array_start = cleaned.find("[")
    array_end = cleaned.rfind("]")
    if array_start != -1 and array_end != -1 and array_end > array_start:
        pairs.append((array_start, array_end))
    if not pairs:
        return None

    for start, end in sorted(pairs, key=lambda item: item[0]):
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except JSONDecodeError:
            normalized = candidate.replace("\n", " ").replace("\t", " ")
            try:
                return json.loads(normalized)
            except JSONDecodeError:
                continue

    return None


def _coerce_tool_calls(payload) -> list[tuple[str, dict]]:
    """Convert model-emitted JSON into normalized tool calls."""
    if isinstance(payload, dict):
        if isinstance(payload.get("tool_calls"), list):
            items = payload["tool_calls"]
        elif isinstance(payload.get("calls"), list):
            items = payload["calls"]
        elif isinstance(payload.get("tools"), list):
            items = payload["tools"]
        else:
            items = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        return []

    normalized_calls = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("tool") or "").strip()
        parameters = (
            item.get("parameters")
            or item.get("args")
            or item.get("arguments")
            or item.get("input")
            or {}
        )
        if isinstance(parameters, str):
            parsed = _extract_json_payload(parameters)
            parameters = parsed if isinstance(parsed, dict) else {}
        if name and isinstance(parameters, dict):
            normalized_calls.append((name, parameters))
    return normalized_calls


def _normalize_tool_call(name: str, args: dict) -> tuple[str, dict]:
    """Correct common local-model tool formatting mistakes."""
    args = dict(args or {})

    if name == "save_calendar_item":
        title = args.get("title") or args.get("event") or args.get("name") or ""
        date = args.get("date") or args.get("day") or args.get("when") or ""
        details = args.get("details") or args.get("note") or args.get("description") or ""
        time = args.get("time") or args.get("at") or ""
        normalized = {"title": title, "date": date, "time": time, "details": details}
        return name, normalized

    if name == "save_journal_entry":
        entry = args.get("entry", "")
        if isinstance(entry, str):
            payload = _extract_json_payload(entry)
            if payload is None:
                return name, args

            date = str(payload.get("date", "")).strip()
            if date:
                title = str(
                    payload.get("title")
                    or payload.get("event")
                    or payload.get("name")
                    or "Saved reminder"
                ).strip()
                details = str(payload.get("details") or payload.get("note") or "").strip()
                blob = " ".join(str(value).lower() for value in payload.values())
                if "birthday" in blob and "birthday" not in title.lower():
                    title = f"{title} birthday"
                return "save_calendar_item", {
                    "title": title,
                    "date": date,
                    "details": details,
                    "source": "conversation",
                }

    return name, args


def _recover_tool_call(response: AIMessage) -> AIMessage:
    """Recover tool calls when a local model emits JSON text instead of structured tool metadata."""
    if getattr(response, "tool_calls", None):
        return response

    raw_text = _extract_text(response.content)
    payload = _extract_json_payload(raw_text)
    if payload is None:
        return response

    available_tools = {tool.name for tool in JUNE_TOOLS}
    recovered_calls = []
    for name, parameters in _coerce_tool_calls(payload):
        name, parameters = _normalize_tool_call(name, parameters)
        if name in available_tools:
            recovered_calls.append(
                {
                    "name": name,
                    "args": parameters,
                    "id": f"recovered_{uuid4().hex[:10]}",
                    "type": "tool_call",
                }
            )

    if not recovered_calls:
        return response

    return AIMessage(
        content="",
        tool_calls=recovered_calls,
    )


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
        response = _recover_tool_call(llm.invoke(messages))
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
        if isinstance(result, list):
            base_ui_state = dict(state.get("ui_state") or {})
            combined = {"messages": []}
            current_ui_state = dict(base_ui_state)
            for item in result:
                update = getattr(item, "update", None)
                if isinstance(update, dict):
                    if "ui_state" in update and isinstance(update["ui_state"], dict):
                        for key, value in update["ui_state"].items():
                            if key not in base_ui_state or base_ui_state.get(key) != value:
                                current_ui_state[key] = value
                        combined["ui_state"] = current_ui_state
                    if "messages" in update:
                        combined["messages"].extend(update.get("messages", []))
                    for key, value in update.items():
                        if key in {"ui_state", "messages"}:
                            continue
                        combined[key] = value
                else:
                    combined["messages"].append(item)
            result = combined
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
