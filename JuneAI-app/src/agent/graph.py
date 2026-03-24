"""JuneAI LangGraph agent."""

from __future__ import annotations

import ast
import json
import operator
from datetime import datetime
from html import unescape
from json import JSONDecodeError
from typing import Any, Annotated, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import StreamWriter

from .config import RuntimeConfig, resolve_runtime_config
from .memory import Memory
from .models import build_chat_model
from .skills import DEFAULT_SKILL, build_system_prompt
from .tools import JUNE_TOOLS


class AgentState(TypedDict):
    """The state passed between every node in the graph."""

    messages: Annotated[list[AnyMessage], operator.add]
    user_id: str
    skill: str
    ui_state: dict[str, Any]
    tool_stats: dict[str, Any]


def _extract_text(content: Any) -> str:
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


def _summarize_tool_messages(
    tool_messages: list[ToolMessage],
    requested_calls: list[dict[str, Any]],
    existing: dict[str, Any] | None,
) -> dict[str, Any]:
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


def _extract_json_payload(text: str) -> Any | None:
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
                try:
                    return ast.literal_eval(candidate)
                except (SyntaxError, ValueError):
                    try:
                        return ast.literal_eval(normalized)
                    except (SyntaxError, ValueError):
                        continue

    return None


def _coerce_tool_calls(payload: Any) -> list[tuple[str, dict[str, Any]]]:
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
        function_block = item.get("function")
        if isinstance(function_block, dict):
            name = str(
                item.get("name")
                or item.get("tool")
                or function_block.get("name")
                or ""
            ).strip()
            parameters = (
                item.get("parameters")
                or item.get("args")
                or item.get("arguments")
                or item.get("input")
                or function_block.get("arguments")
                or {}
            )
        else:
            name = str(item.get("name") or item.get("tool") or "").strip()
            parameters = (
                item.get("parameters")
                or item.get("args")
                or item.get("arguments")
                or item.get("input")
                or {}
            )
        if not name and isinstance(item.get("tool_name"), str):
            name = item["tool_name"].strip()
        parameters = (
            item.get("parameters")
            or item.get("args")
            or item.get("arguments")
            or item.get("input")
            or parameters
        )
        if isinstance(parameters, str):
            parsed = _extract_json_payload(parameters)
            parameters = parsed if isinstance(parsed, dict) else {}
        if name and isinstance(parameters, dict):
            normalized_calls.append((name, parameters))
    return normalized_calls


def _normalize_tool_call(name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Correct common local-model tool formatting mistakes."""
    args = dict(args or {})
    alias_map = {
        "save_goal": "track_goal",
        "create_goal": "track_goal",
        "add_goal": "track_goal",
        "save_reminder": "save_calendar_item",
        "add_calendar_item": "save_calendar_item",
        "create_calendar_item": "save_calendar_item",
        "save_trip": "save_calendar_item",
        "save_birthday": "save_calendar_item",
        "save_preference": "save_user_preference",
        "save_favorite": "save_favorite_recommendation",
        "add_favorite": "save_favorite_recommendation",
        "save_workout": "log_workout_session",
        "record_workout": "log_workout_session",
        "save_meal": "log_nutrition",
        "record_meal": "log_nutrition",
        "record_water": "log_water",
        "set_chapter": "set_ui_chapter",
        "open_chapter": "set_ui_chapter",
        "focus_workspace": "set_ui_focus",
        "update_workspace": "set_ui_focus",
    }
    name = alias_map.get(name, name)

    if name == "save_calendar_item":
        title = args.get("title") or args.get("event") or args.get("name") or ""
        date = args.get("date") or args.get("day") or args.get("when") or ""
        details = args.get("details") or args.get("note") or args.get("description") or ""
        time = args.get("time") or args.get("at") or ""
        if title and "birthday" in str(args).lower() and "birthday" not in str(title).lower():
            title = f"{title} birthday"
        if title and any(word in str(args).lower() for word in ("trip", "travel", "flight")):
            details = details or "trip"
        normalized = {"title": title, "date": date, "time": time, "details": details}
        return name, normalized

    if name == "track_goal":
        return name, {
            "title": args.get("title") or args.get("goal") or args.get("name") or "",
            "category": args.get("category") or args.get("area") or "personal",
            "target_date": args.get("target_date") or args.get("deadline") or args.get("date") or "",
            "next_step": args.get("next_step") or args.get("next") or args.get("action") or "",
            "status": args.get("status") or "active",
        }

    if name == "save_open_loop":
        return name, {
            "topic": args.get("topic") or args.get("title") or args.get("name") or "",
            "next_step": args.get("next_step") or args.get("next") or args.get("action") or "",
            "due_date": args.get("due_date") or args.get("deadline") or args.get("date") or "",
            "status": args.get("status") or "open",
        }

    if name == "save_gym_plan":
        return name, {
            "name": args.get("name") or args.get("title") or args.get("plan_name") or "Gym Plan",
            "schedule": args.get("schedule") or args.get("split") or args.get("routine") or "",
            "goal": args.get("goal") or args.get("focus") or "",
            "notes": args.get("notes") or args.get("details") or "",
            "status": args.get("status") or "active",
        }

    if name == "save_food_program":
        return name, {
            "name": args.get("name") or args.get("title") or args.get("program_name") or "Food Program",
            "goal": args.get("goal") or args.get("focus") or "",
            "daily_structure": (
                args.get("daily_structure")
                or args.get("structure")
                or args.get("meal_structure")
                or args.get("plan")
                or ""
            ),
            "notes": args.get("notes") or args.get("details") or "",
            "status": args.get("status") or "active",
        }

    if name == "save_relationship_profile":
        return name, {
            "person": args.get("person") or args.get("name") or "",
            "relationship": args.get("relationship") or args.get("relation") or "",
            "summary": args.get("summary") or args.get("context") or args.get("details") or "",
            "user_needs": args.get("user_needs") or args.get("needs") or "",
            "cautions": args.get("cautions") or args.get("warnings") or "",
        }

    if name == "save_user_preference":
        return name, {
            "category": args.get("category") or args.get("type") or "general",
            "value": args.get("value") or args.get("preference") or args.get("title") or "",
            "context": args.get("context") or args.get("details") or args.get("reason") or "",
        }

    if name == "save_favorite_recommendation":
        return name, {
            "category": args.get("category") or args.get("type") or "general",
            "title": args.get("title") or args.get("name") or "",
            "reason": args.get("reason") or args.get("details") or "",
            "creator": args.get("creator") or args.get("author") or args.get("artist") or "",
            "status": args.get("status") or "saved",
        }

    if name == "log_workout_session":
        return name, {
            "plan_name": args.get("plan_name") or args.get("name") or args.get("title") or "Workout",
            "exercises": args.get("exercises") or args.get("workout") or args.get("details") or "",
            "duration_min": args.get("duration_min") or args.get("duration") or 0,
            "notes": args.get("notes") or "",
            "energy_rating": args.get("energy_rating") or args.get("energy") or 0,
        }

    if name == "log_body_metrics":
        return name, {
            "weight_kg": args.get("weight_kg") or args.get("weight") or 0.0,
            "sleep_hours": args.get("sleep_hours") or args.get("sleep") or 0.0,
            "sleep_quality": args.get("sleep_quality") or args.get("sleep_score") or 0,
            "energy": args.get("energy") or args.get("energy_rating") or 0,
            "stress": args.get("stress") or args.get("stress_level") or 0,
            "soreness": args.get("soreness") or args.get("muscle_soreness") or 0,
            "resting_hr": args.get("resting_hr") or args.get("heart_rate") or args.get("rest_hr") or 0,
            "steps": args.get("steps") or args.get("step_count") or 0,
            "notes": args.get("notes") or "",
        }

    if name == "create_habit":
        return name, {
            "name": args.get("name") or args.get("habit") or args.get("title") or "",
            "category": args.get("category") or "health",
            "target_days": args.get("target_days") or args.get("frequency") or "daily",
        }

    if name == "log_habit_completion":
        return name, {"habit_name": args.get("habit_name") or args.get("name") or args.get("habit") or ""}

    if name == "log_nutrition":
        return name, {
            "meal": args.get("meal") or args.get("meal_type") or "meal",
            "description": args.get("description") or args.get("details") or args.get("food") or "",
            "calories_est": args.get("calories_est") or args.get("calories") or 0,
            "protein_est": args.get("protein_est") or args.get("protein") or 0,
        }

    if name == "log_water":
        return name, {"glasses": args.get("glasses") or args.get("count") or args.get("amount") or 1}

    if name == "set_ui_focus":
        return name, {
            "title": args.get("title") or args.get("heading") or "Workspace",
            "body": args.get("body") or args.get("content") or args.get("text") or "",
            "footer": args.get("footer") or args.get("notice") or "",
        }

    if name == "set_ui_checklist":
        items = args.get("items") or args.get("checklist") or args.get("lines") or ""
        if isinstance(items, list):
            items = "\n".join(str(item) for item in items)
        return name, {
            "title": args.get("title") or args.get("heading") or "Next steps",
            "items": items,
        }

    if name == "set_ui_layout":
        return name, {
            "layout": args.get("layout") or args.get("mode") or "split",
            "notice": args.get("notice") or "",
        }

    if name == "set_ui_chapter":
        return name, {
            "chapter": args.get("chapter") or args.get("name") or args.get("section") or "",
            "notice": args.get("notice") or "",
        }

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


def _build_memory_context(user_id: str) -> str:
    """Build compact memory context for the current turn."""
    memory = Memory(user_id)
    summary = memory.get_today_summary()
    lines = ["Current user context:"]

    body = summary.get("body_metrics")
    if body:
        body_parts = []
        if body.get("weight_kg"):
            body_parts.append(f"weight {body['weight_kg']}kg")
        if body.get("sleep_hours"):
            body_parts.append(f"sleep {body['sleep_hours']}h")
        if body.get("sleep_quality"):
            body_parts.append(f"sleep quality {body['sleep_quality']}/5")
        if body.get("energy"):
            body_parts.append(f"energy {body['energy']}/5")
        if body.get("stress"):
            body_parts.append(f"stress {body['stress']}/5")
        if body.get("soreness"):
            body_parts.append(f"soreness {body['soreness']}/5")
        if body.get("resting_hr"):
            body_parts.append(f"resting HR {body['resting_hr']}")
        if body.get("steps"):
            body_parts.append(f"steps {body['steps']}")
        if body.get("notes"):
            body_parts.append(f"notes: {body['notes']}")
        lines.append("Today's body metrics: " + ", ".join(body_parts))
    else:
        lines.append("Today's body metrics: not logged.")

    workout = summary.get("workout")
    if workout:
        lines.append(
            "Today's workout: "
            + f"{workout.get('plan_name', 'Workout')}"
            + (f" for {workout.get('duration_min')} min" if workout.get("duration_min") else "")
        )
    else:
        lines.append("Today's workout: not logged.")

    lines.append(f"Today's water: {summary.get('water_glasses', 0)} glasses.")
    lines.append(
        "Today's habits: "
        + f"{summary.get('habits_done', 0)}/{summary.get('habits_total', 0)} done."
    )

    if summary.get("meals_logged"):
        lines.append(
            "Today's nutrition: "
            + f"{summary.get('meals_logged', 0)} meals"
            + (
                f", ~{summary.get('calories_est', 0)} kcal, ~{summary.get('protein_est', 0)}g protein."
            )
        )

    recent_metrics = summary.get("recent_body_metrics", []) or []
    if len(recent_metrics) > 1:
        previous = recent_metrics[-2]
        previous_parts = []
        if previous.get("sleep_hours"):
            previous_parts.append(f"sleep {previous['sleep_hours']}h")
        if previous.get("energy"):
            previous_parts.append(f"energy {previous['energy']}/5")
        if previous.get("stress"):
            previous_parts.append(f"stress {previous['stress']}/5")
        if previous.get("soreness"):
            previous_parts.append(f"soreness {previous['soreness']}/5")
        if previous_parts:
            lines.append(
                f"Most recent prior body check ({previous.get('date', 'unknown date')}): "
                + ", ".join(previous_parts)
            )

    lines.append("Use this context in reasoning, recommendations, and follow-up questions.")
    return "\n".join(lines)


def create_june_agent(llm: Any = None, runtime: RuntimeConfig | None = None) -> Any:
    """Build and compile the JuneAI LangGraph agent."""

    runtime = runtime or resolve_runtime_config()
    llm = llm or build_chat_model(runtime)
    if hasattr(llm, "bind_tools"):
        llm = llm.bind_tools(JUNE_TOOLS)

    tool_node = ToolNode(JUNE_TOOLS, handle_tool_errors=True)

    def chat(state: AgentState, writer: StreamWriter) -> dict[str, Any]:
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
        memory_context = _build_memory_context(state["user_id"])
        messages = [
            SystemMessage(content=prompt),
            SystemMessage(content=memory_context),
        ] + state["messages"]
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

    def run_tools(state: AgentState, writer: StreamWriter) -> dict[str, Any]:
        """Execute tools and report structured diagnostics."""
        last_message = state["messages"][-1] if state.get("messages") else None
        requested_calls = list(getattr(last_message, "tool_calls", []) or [])
        result = tool_node.invoke(state)
        if isinstance(result, list):
            base_ui_state = dict(state.get("ui_state") or {})
            combined: dict[str, Any] = {"messages": []}
            current_ui_state = dict(base_ui_state)
            for item in result:
                if isinstance(item, dict):
                    if "ui_state" in item and isinstance(item["ui_state"], dict):
                        current_ui_state.update(item["ui_state"])
                        combined["ui_state"] = current_ui_state
                    if "messages" in item:
                        combined["messages"].extend(item.get("messages", []))
                    for key, value in item.items():
                        if key in {"ui_state", "messages"}:
                            continue
                        combined[key] = value
                    continue
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
        return dict(result)

    graph = StateGraph(AgentState)
    graph.add_node("chat", chat)
    graph.add_node("tools", run_tools)

    graph.set_entry_point("chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")

    return graph.compile()


june_agent = create_june_agent()
