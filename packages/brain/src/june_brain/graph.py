"""JuneAI LangGraph agent."""

from __future__ import annotations

import json
import logging
import operator
import os
import re
import threading
import time
from datetime import datetime
from hashlib import sha256
from html import unescape
from json import JSONDecodeError
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime
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
from .skills import DEFAULT_SKILL, build_system_prompt, load_skill_tools
from .telemetry import record_route_selection, record_save_event, record_tool_call
from .tools import JUNE_TOOLS, JUNE_TOOLS_GEMMA

_CONTEXT_CACHE: dict[str, tuple[float, str]] = {}  # user_id -> (timestamp, context)
_CONTEXT_TTL = 30.0  # seconds
_CONTEXT_CACHE_MAX = 256  # FIFO cap so the dict can't grow unbounded


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
    """Correct common local-model tool formatting mistakes.

    Delegates to the data-driven table in ``tool_aliases.py``.
    """
    from .tool_aliases import resolve_tool_call

    return resolve_tool_call(name, args)


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
    if len(_CONTEXT_CACHE) >= _CONTEXT_CACHE_MAX and user_id not in _CONTEXT_CACHE:
        # FIFO eviction: drop the oldest entry to keep the cache bounded.
        oldest = min(_CONTEXT_CACHE.items(), key=lambda kv: kv[1][0])[0]
        _CONTEXT_CACHE.pop(oldest, None)
    _CONTEXT_CACHE[user_id] = (now, result)
    return result


def _normalize_recall_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Make a recall hit's ``ref`` match the prefix scheme used in /memory.

    ``MemoryManager.recall`` returns vector hits with bare ``fact_id`` and
    graph hits with bare ``node_id``; the memory snapshot route emits
    those as ``semantic:<id>`` and ``node:<id>``. We normalize here so the
    UI can deep-link recalled memories into the browser without each
    surface re-implementing the prefix logic.
    """
    source = hit.get("source")
    ref = hit.get("ref", "")
    kind = hit.get("kind", "")
    if source == "vector":
        ref = f"semantic:{ref}"
    elif source == "graph":
        ref = f"edge:{ref}" if kind.startswith("edge:") else f"node:{ref}"
    return {
        "ref": ref,
        "text": hit.get("text", ""),
        "source": str(source or ""),
        "kind": str(kind or ""),
        "score": hit.get("score"),
        "feedback": str(hit.get("feedback") or ""),
    }


def _recall_with_hits(
    user_id: str, query: str, k: int = 5
) -> tuple[str, list[dict[str, Any]]]:
    """Fresh per-turn fan-out across the three memory stores.

    Returns the formatted prompt block AND the raw hits, so the caller can
    both inject the block into the system prompt and stream the hits to
    the UI for a "memories used" disclosure.

    Not cached because the query changes every turn and the individual
    lookups are cheap (vector search ~tens of ms, graph + sqlite keyword
    scans ~ms). If recall raises we swallow and return empties so a broken
    memory store never takes down the chat loop.
    """
    query = (query or "").strip()
    if not query:
        return "", []
    try:
        manager = MemoryManager(user_id)
        hits = manager.recall(query, k=k)
    except Exception:
        logging.exception("recall block failed for user=%s", user_id)
        return "", []
    return manager.format_for_prompt(hits), [_normalize_recall_hit(h) for h in hits]


def _latest_user_text(messages: list[AnyMessage]) -> str:
    """Find the most recent human message content in the agent state."""
    for message in reversed(messages or []):
        if getattr(message, "type", None) == "human" or message.__class__.__name__ == "HumanMessage":
            return _extract_text(message.content)
    return ""


def _select_tools_for_runtime(runtime: RuntimeConfig) -> list[Any]:
    """Choose a tool set sized for the active runtime.

    Gemma 4 (local, small) runs with the trimmed tool schema to keep prompts
    lean. Gemini (cloud) gets the full tool set. Enabled MCP skills append
    their tools on top, so toggling a skill on the /skills page adds or
    removes its capabilities on the next agent build.

    Native tools win on name collision — the skill is shadowed until the
    native copy is removed. That matters while we're mid-migration: we do
    not want two implementations of e.g. ``log_water`` competing for the
    same call.
    """
    if runtime.preset_key == "gemma":
        native = JUNE_TOOLS_GEMMA
    else:
        native = JUNE_TOOLS

    native_names = {getattr(t, "name", "") for t in native}
    try:
        skill_tools = [t for t in load_skill_tools() if t.name not in native_names]
    except Exception:
        logging.exception("Failed to load skill tools")
        skill_tools = []

    return list(native) + skill_tools


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
        recall_block, recall_hits = _recall_with_hits(
            state["user_id"], _latest_user_text(state["messages"])
        )
        if recall_hits:
            writer({"event": "recall", "hits": recall_hits})
        system_messages = [
            SystemMessage(content=prompt),
            SystemMessage(content=memory_context),
        ]
        if recall_block:
            system_messages.append(SystemMessage(content=recall_block))
        messages = system_messages + _trim_messages(state["messages"])
        _invoke_started_at = time.monotonic()
        raw_response = llm.invoke(messages)
        _invoke_latency_ms = max(0, int((time.monotonic() - _invoke_started_at) * 1000))
        writer(
            {
                "event": "provenance",
                "provider": runtime.preset_key,
                "model": runtime.model,
                "tier": runtime.preset_key,
                "latency_ms": _invoke_latency_ms,
            }
        )
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
        result = _normalize_tool_node_result(
            tool_node._func(state, {}, Runtime(stream_writer=writer)),
            state.get("ui_state") or {},
        )
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


_agent_lock = threading.Lock()
startup_error: str | None = None
june_agent: Any | None = None
_agent_fingerprint: tuple[Any, ...] | None = None


def _runtime_fingerprint(runtime: RuntimeConfig) -> tuple[Any, ...]:
    """Return the runtime fields that require a compiled agent refresh."""
    api_key_digest = sha256(runtime.api_key.encode("utf-8")).hexdigest() if runtime.api_key else ""
    return (
        runtime.preset_key,
        runtime.provider,
        runtime.model,
        runtime.base_url,
        runtime.temperature,
        runtime.max_tokens,
        runtime.tool_strategy,
        runtime.prompt_style,
        api_key_digest,
    )


def get_or_create_agent() -> Any:
    """Return the cached agent, rebuilding it when runtime config changes.

    Agent construction is lazy so importing ``june_brain.graph`` never locks
    the process to a provider before stored config has been applied.
    """
    global june_agent, startup_error, _agent_fingerprint
    if os.environ.get("JUNE_IS_SKILL_SUBPROCESS") == "1":
        raise RuntimeError("June agent is unavailable inside a skill subprocess.")

    try:
        runtime = resolve_runtime_config()
        fingerprint = _runtime_fingerprint(runtime)
    except Exception as exc:
        with _agent_lock:
            june_agent = None
            _agent_fingerprint = None
            startup_error = str(exc)
        logging.exception("get_or_create_agent config resolution failed")
        raise

    with _agent_lock:
        if june_agent is not None and _agent_fingerprint == fingerprint:
            return june_agent
        try:
            june_agent = create_june_agent(runtime=runtime)
            _agent_fingerprint = fingerprint
            startup_error = None
            return june_agent
        except Exception as exc:
            june_agent = None
            _agent_fingerprint = None
            startup_error = str(exc)
            logging.exception("get_or_create_agent build failed")
            raise


def invalidate_agent() -> None:
    """Clear the cached agent so the next chat request rebuilds from env."""
    global june_agent, startup_error, _agent_fingerprint
    with _agent_lock:
        june_agent = None
        startup_error = None
        _agent_fingerprint = None


def run_agent_sync(prompt: str, user_id: str) -> str:
    """Synchronous, non-streaming agent invocation for scheduled tasks.

    Builds a minimal system prompt + the user ``prompt``, invokes the
    current LLM, and returns the text reply. No tools are bound —
    scheduled invocations are informational (briefings, reminders, etc.)
    and shouldn't trigger side effects.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from june_brain.memory import Memory
    from june_brain.prompts import build_system_prompt

    agent = get_or_create_agent()
    if agent is None:
        return "error: agent not available"
    memory = Memory(user_id)
    system = build_system_prompt("default", memory=memory)
    response = agent.invoke({
        "messages": [
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ],
        "user_id": user_id,
        "skill": "default",
    })
    last = response["messages"][-1] if response.get("messages") else None
    return str(getattr(last, "content", "")) if last else ""


def reload_agent() -> None:
    """Rebuild :data:`june_agent` in place. Call after a skill toggle so the
    next chat turn sees the updated tool surface.

    The module-level ``june_agent`` reference is replaced atomically. Callers
    that already captured a reference to the old agent keep a working copy
    until their request completes.
    """
    global june_agent, startup_error, _agent_fingerprint
    with _agent_lock:
        try:
            runtime = resolve_runtime_config()
            june_agent = create_june_agent(runtime=runtime)
            _agent_fingerprint = _runtime_fingerprint(runtime)
            startup_error = None
        except Exception as exc:
            june_agent = None
            _agent_fingerprint = None
            startup_error = str(exc)
            logging.exception("reload_agent failed")
