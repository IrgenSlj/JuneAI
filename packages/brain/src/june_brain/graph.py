"""JuneAI LangGraph agent."""

from __future__ import annotations

import json
import logging
import operator
import re
import time
from datetime import datetime
from html import unescape
from json import JSONDecodeError
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import StreamWriter

from .config import RuntimeConfig, resolve_runtime_config
from .context_intelligence import (
    build_active_commitments_summary,
    build_recovery_readiness_summary,
    format_active_commitments_summary,
    format_recovery_readiness_summary,
)
from .memory import Memory, MemoryManager
from .models import build_chat_model
from .skills import DEFAULT_SKILL, build_system_prompt
from .telemetry import record_route_selection, record_save_event, record_tool_call
from .tools import JUNE_TOOLS, JUNE_TOOLS_CORE, JUNE_TOOLS_GEMMA

_CONTEXT_CACHE: dict[str, tuple[float, str]] = {}  # user_id -> (timestamp, context)
_CONTEXT_TTL = 30.0  # seconds


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


def _append_messages(target: list[Any], payload: Any) -> None:
    """Append a tool-node message payload without assuming a fixed container type."""
    if payload is None:
        return
    if isinstance(payload, list):
        target.extend(payload)
        return
    target.append(payload)


def _normalize_tool_node_result(result: Any, base_ui_state: dict[str, Any]) -> dict[str, Any]:
    """Flatten ToolNode results into a dict we can summarize and return safely."""
    combined: dict[str, Any] = {"messages": []}
    current_ui_state = dict(base_ui_state)
    ui_state_seen = False

    def merge(item: Any) -> None:
        nonlocal ui_state_seen
        if isinstance(item, dict):
            if "ui_state" in item and isinstance(item["ui_state"], dict):
                for key, value in item["ui_state"].items():
                    if key not in base_ui_state or base_ui_state.get(key) != value:
                        current_ui_state[key] = value
                ui_state_seen = True
            if "messages" in item:
                _append_messages(combined["messages"], item.get("messages"))
            for key, value in item.items():
                if key in {"ui_state", "messages"}:
                    continue
                combined[key] = value
            return

        update = getattr(item, "update", None)
        if isinstance(update, dict):
            merge(update)
            return

        _append_messages(combined["messages"], item)

    if isinstance(result, list):
        for item in result:
            merge(item)
    else:
        merge(result)

    if ui_state_seen:
        combined["ui_state"] = current_ui_state
    return combined


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _strip_internal_thoughts(text: str) -> str:
    """Remove Gemma thought-channel markup before display or reuse."""
    cleaned = re.sub(r"<\|channel>thought\s*.*?<channel\|>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _extract_json_payload(text: str) -> Any | None:
    """Extract the outermost JSON object or array from free-form model text."""
    cleaned = unescape(_strip_internal_thoughts(_strip_code_fence(text))).strip()
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
        # strategy 1: direct parse
        try:
            return json.loads(candidate)
        except JSONDecodeError:
            pass
        # strategy 2: collapse whitespace
        normalized = candidate.replace("\n", " ").replace("\t", " ")
        try:
            return json.loads(normalized)
        except JSONDecodeError:
            pass
        # strategy 3: replace single quotes with double quotes (common local-model mistake)
        single_to_double = re.sub(r"(?<!\\)'", '"', normalized)
        try:
            return json.loads(single_to_double)
        except JSONDecodeError:
            pass
        # strategy 4: unescape HTML entities and retry
        unescaped = unescape(normalized)
        try:
            return json.loads(unescaped)
        except JSONDecodeError:
            continue

    logging.warning(
        "_extract_json_payload: all JSON strategies failed. Raw (first 120 chars): %s",
        text[:120],
    )
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


def _recover_tool_call(response: AIMessage, bound_tool_names: set[str]) -> AIMessage:
    """Recover tool calls when a local model emits JSON text instead of structured tool metadata.

    bound_tool_names must be the set of tools actually bound to this agent instance —
    not the global JUNE_TOOLS — so recovery never constructs calls for tools that
    ToolNode cannot dispatch.
    """
    if getattr(response, "tool_calls", None):
        return response

    raw_text = _extract_text(response.content)
    payload = _extract_json_payload(raw_text)
    if payload is None:
        return response

    recovered_calls = []
    for name, parameters in _coerce_tool_calls(payload):
        name, parameters = _normalize_tool_call(name, parameters)
        if name in bound_tool_names:
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


def invalidate_context_cache(user_id: str) -> None:
    """Remove a user's cached memory context so the next turn re-reads from disk."""
    _CONTEXT_CACHE.pop(user_id, None)


def _compute_memory_context(memory: Memory) -> str:
    """Read memory files and build the compact context string."""
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

    lines.append("")
    lines.append(format_recovery_readiness_summary(build_recovery_readiness_summary(memory)))
    lines.append("")
    lines.append(format_active_commitments_summary(build_active_commitments_summary(memory)))
    lines.append("Use this context in reasoning, recommendations, and follow-up questions.")
    return "\n".join(lines)


def _build_memory_context(user_id: str) -> str:
    """Return cached memory context, refreshing from disk when the TTL has expired."""
    now = time.monotonic()
    cached = _CONTEXT_CACHE.get(user_id)
    if cached and (now - cached[0]) < _CONTEXT_TTL:
        return cached[1]
    memory = Memory(user_id)
    result = _compute_memory_context(memory)
    _CONTEXT_CACHE[user_id] = (now, result)
    return result


def _build_recall_block(user_id: str, query: str, k: int = 5) -> str:
    """Fresh per-turn fan-out across the three memory stores.

    Not cached because the query changes every turn and the individual
    lookups are cheap (vector search ~tens of ms, graph + sqlite keyword
    scans ~ms). If recall raises, we swallow and return an empty block so
    a broken memory store never takes down the chat loop.
    """
    query = (query or "").strip()
    if not query:
        return ""
    try:
        manager = MemoryManager(user_id)
        hits = manager.recall(query, k=k)
    except Exception as exc:  # noqa: BLE001
        logging.warning("recall block failed for user=%s: %s", user_id, exc)
        return ""
    return manager.format_for_prompt(hits)


def _latest_user_text(messages: list[AnyMessage]) -> str:
    """Find the most recent human message content in the agent state."""
    for message in reversed(messages or []):
        if getattr(message, "type", None) == "human" or message.__class__.__name__ == "HumanMessage":
            return _extract_text(message.content)
    return ""


def _select_tools_for_runtime(runtime: RuntimeConfig) -> list[Any]:
    """Choose a tool set sized for the active runtime.

    Gemma 4 (local, small) runs with the trimmed tool schema to keep prompts
    lean. Gemini (cloud) gets the full tool set.
    """
    if runtime.preset_key == "gemma":
        return JUNE_TOOLS_GEMMA
    return JUNE_TOOLS


_MESSAGE_WINDOW = 24  # messages kept verbatim (must be even so pairs stay intact)


def _trim_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Keep the tail of the conversation within the context window.

    Rules:
    - Always keep the most recent _MESSAGE_WINDOW messages verbatim.
    - If older messages exist, prepend a single SystemMessage summarising them
      so June retains broad conversational continuity without token explosion.
    - Tool call / tool result pairs are kept together: a ToolMessage is only
      included if the preceding AIMessage that requested it is also included.
    """
    if len(messages) <= _MESSAGE_WINDOW:
        return messages

    recent = messages[-_MESSAGE_WINDOW:]
    older = messages[:-_MESSAGE_WINDOW]

    # Count turns and topics from older messages for a lightweight summary
    user_texts = [
        _extract_text(m.content)
        for m in older
        if getattr(m, "type", None) == "human" or m.__class__.__name__ == "HumanMessage"
    ]
    assistant_texts = [
        _extract_text(m.content)
        for m in older
        if (getattr(m, "type", None) == "ai" or m.__class__.__name__ == "AIMessage")
        and _extract_text(m.content).strip()
    ]
    turn_count = len(user_texts)
    topic_snippets = ". ".join(t[:80] for t in user_texts[-4:] if t.strip())
    summary_lines = [
        f"[Earlier in this session — {turn_count} turns not shown in full]",
    ]
    if topic_snippets:
        summary_lines.append(f"Topics discussed: {topic_snippets}.")
    if assistant_texts:
        last_ai = assistant_texts[-1][:120]
        summary_lines.append(f"June's last substantive response before this window: {last_ai}")

    summary_msg = SystemMessage(content="\n".join(summary_lines))
    return [summary_msg] + list(recent)


def create_june_agent(llm: Any = None, runtime: RuntimeConfig | None = None) -> Any:
    """Build and compile the JuneAI LangGraph agent."""
    runtime = runtime or resolve_runtime_config()
    llm = llm or build_chat_model(runtime)
    _tools = _select_tools_for_runtime(runtime)
    if hasattr(llm, "bind_tools"):
        llm = llm.bind_tools(_tools)

    tool_node = ToolNode(_tools, handle_tool_errors=True)
    # Compute once at agent-compile time — stable for the lifetime of this agent instance.
    _bound_tool_names: set[str] = {t.name for t in _tools}

    def chat(state: AgentState, writer: StreamWriter) -> dict[str, Any]:
        """Run the main chat node."""
        skill = state.get("skill", DEFAULT_SKILL)
        now = datetime.now().astimezone()
        _chat_memory = Memory(state["user_id"])
        prompt = build_system_prompt(skill, now=now, runtime=runtime, memory=_chat_memory)
        record_route_selection(
            _chat_memory,
            skill,
            source="graph",
            payload={
                "runtime_label": runtime.label,
                "runtime_provider": runtime.provider,
                "runtime_model": runtime.model,
            },
        )
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
        recall_block = _build_recall_block(
            state["user_id"], _latest_user_text(state["messages"])
        )
        system_messages = [
            SystemMessage(content=prompt),
            SystemMessage(content=memory_context),
        ]
        if recall_block:
            system_messages.append(SystemMessage(content=recall_block))
        messages = system_messages + _trim_messages(state["messages"])
        raw_response = llm.invoke(messages)
        if isinstance(getattr(raw_response, "content", None), str):
            raw_response.content = _strip_internal_thoughts(str(raw_response.content))
        # "native" and "balanced_reasoning" (Claude) both emit proper tool_calls —
        # only "recovery" models need JSON extraction from free-form text.
        if runtime.tool_strategy in ("native", "balanced_reasoning") and getattr(raw_response, "tool_calls", None):
            response = raw_response
        else:
            response = _recover_tool_call(raw_response, _bound_tool_names)
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
        memory = Memory(state["user_id"])
        for call in requested_calls:
            record_tool_call(
                memory,
                call.get("name", ""),
                status="requested",
                source="graph",
                route=state.get("skill", DEFAULT_SKILL),
                payload={"tool_call_id": call.get("id", "")},
            )
        result = _normalize_tool_node_result(tool_node.invoke(state), state.get("ui_state") or {})
        tool_messages = [
            message for message in result.get("messages", []) if isinstance(message, ToolMessage)
        ]
        tool_stats = _summarize_tool_messages(tool_messages, requested_calls, state.get("tool_stats"))
        for call in tool_stats.get("last_calls", []):
            tool_name = call.get("name", "")
            payload = {
                "tool_call_id": call.get("id", ""),
                "preview": call.get("preview", ""),
            }
            record_tool_call(
                memory,
                tool_name,
                status=call.get("status", "error"),
                source="graph",
                route=state.get("skill", DEFAULT_SKILL),
                payload=payload,
            )
            if call.get("status") == "success" and tool_name.startswith(("save_", "log_", "track_", "set_ui_")):
                record_save_event(
                    memory,
                    kind=tool_name,
                    name=tool_name,
                    status="saved",
                    source="graph",
                    route=state.get("skill", DEFAULT_SKILL),
                    payload=payload,
                )
        writer({"event": "tool_results", "calls": tool_stats["last_calls"], "summary": tool_stats})
        result["tool_stats"] = tool_stats
        invalidate_context_cache(state["user_id"])
        return dict(result)

    graph = StateGraph(AgentState)
    graph.add_node("chat", chat)
    graph.add_node("tools", run_tools)

    graph.set_entry_point("chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")

    return graph.compile()


startup_error: str | None = None
try:
    june_agent = create_june_agent()
except Exception as exc:
    june_agent = None
    startup_error = str(exc)
