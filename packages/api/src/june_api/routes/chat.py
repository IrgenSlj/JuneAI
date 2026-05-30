"""POST /chat — SSE streaming chat endpoint."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Iterable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from june_brain import graph as brain_graph
from june_brain.loop.interface import HarnessLoop, SessionState, TurnProvenance, TurnResult
from june_brain.memory import Memory, MemoryManager
from june_brain.providers.base import Message
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from starlette.background import BackgroundTask

from ..schemas import ChatEvent, ChatHistory, ChatHistoryMessage, ChatRequest, RecallHit

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def get_agent() -> Any:
    """Resolve the LangGraph agent.

    Exposed as a FastAPI dependency so tests can stub it via
    ``app.dependency_overrides[get_agent] = ...`` without touching
    process-global state.
    """
    try:
        return brain_graph.get_or_create_agent()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"June agent failed to start: {brain_graph.startup_error or exc}",
        ) from exc


def get_harness_loop() -> HarnessLoop:
    """Resolve the live-chat HarnessLoop implementation.

    Exposed as a FastAPI dependency so tests can stub it via
    ``app.dependency_overrides[get_harness_loop] = ...`` without touching
    process-global state.  Returns the active engine from
    ``june_brain.loop.engine.get_loop()`` (hand-written loop by default).
    """
    from june_brain.loop.engine import get_loop

    return get_loop()


def _chat_use_harness() -> bool:
    return os.getenv("JUNE_CHAT_USE_HARNESS", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _event_to_sse(event: ChatEvent) -> str:
    """Format a ChatEvent as one SSE data frame."""
    return f"data: {event.model_dump_json()}\n\n"


def _safe_tool_args(call: Any) -> dict[str, Any]:
    """Coerce a tool-call ``args`` blob into a dict, never raising.

    LangChain typically yields a dict here, but a fragile local model
    can emit ``args`` as a JSON string, a list, or ``None``. The chat
    SSE loop must not die on one bad call — surface the best-effort
    dict (possibly empty) so the rest of the turn still streams.
    """
    raw = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        import json

        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            return {}
    return {}


def _empty_ui_state() -> dict[str, Any]:
    return {
        "layout": "split",
        "focus_title": "Workspace",
        "focus_body": "",
        "checklist_title": "Next steps",
        "checklist_items": [],
        "notice": "",
    }


def _tool_call_events(message: AIMessage) -> Iterable[ChatEvent]:
    for call in getattr(message, "tool_calls", None) or []:
        yield ChatEvent(
            type="tool_call",
            tool_name=str(call.get("name", "")),
            tool_args=_safe_tool_args(call),
        )


def _tool_result_event(message: ToolMessage) -> ChatEvent:
    content = message.content
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return ChatEvent(
        type="tool_result",
        tool_name=str(getattr(message, "name", "") or ""),
        tool_result=str(content)[:4000],
    )


def _messages_for_turn(user_id: str, user_text: str) -> list[Any]:
    """Persist the user turn and return the conversation context for the agent."""
    if not user_text.strip():
        return [HumanMessage(content=user_text)]
    try:
        memory = Memory(user_id)
        memory.save_message("user", user_text)
        return memory.load_chat_messages()
    except Exception:
        logger.exception("chat history load failed for user=%s", user_id)
        return [HumanMessage(content=user_text)]


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _provider_message(message: Any) -> Message | None:
    message_type = str(getattr(message, "type", "") or "")
    raw_role = str(getattr(message, "role", "") or "")
    role = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
    }.get(message_type, raw_role)
    if role not in {"system", "user", "assistant", "tool"}:
        return None
    return Message(
        role=role,
        content=_message_content_text(getattr(message, "content", "")),
    )


def _harness_turn_inputs(request: ChatRequest) -> tuple[SessionState, Message]:
    history = [
        msg
        for raw in _messages_for_turn(request.user_id, request.message)
        if (msg := _provider_message(raw)) is not None
    ]
    if (
        history
        and history[-1].role == "user"
        and history[-1].content == request.message
    ):
        history = history[:-1]
    return (
        SessionState(
            user_id=request.user_id,
            messages=history,
            skill=request.skill,
        ),
        Message(role="user", content=request.message),
    )


def _turn_provenance_dict(prov: TurnProvenance) -> dict[str, Any]:
    """Build the provenance payload dict from a TurnProvenance object."""
    tiers = list(prov.tiers_used)
    models = list(prov.model_ids)
    return {
        "provider": "cloud" if prov.cloud_call else "local",
        "model": models[0] if models else "",
        "model_ids": models,
        "tier": tiers[0] if tiers else "",
        "tiers_used": tiers,
        "latency_ms": prov.latency_ms,
        "cloud_call": prov.cloud_call,
        "cloud_payload_summary": prov.cloud_payload_summary,
        "memories_recalled": prov.memories_recalled,
        "skills_called": list(prov.skills_called),
        "rationale": prov.rationale,
    }


def _turn_result_provenance(result: TurnResult) -> dict[str, Any]:
    d = _turn_provenance_dict(result.provenance)
    d["input_tokens"] = result.tokens.input_tokens
    d["output_tokens"] = result.tokens.output_tokens
    d["compacted"] = result.compacted
    return d


def _turn_result_events(result: TurnResult) -> Iterable[ChatEvent]:
    content = result.assistant_msg.content
    if content:
        yield ChatEvent(type="token", content=content)
    for call in result.tool_calls:
        yield ChatEvent(type="tool_call", tool_name=call.name, tool_args=call.args)
    yield ChatEvent(type="provenance", provenance=_turn_result_provenance(result))
    yield ChatEvent(type="done")


async def _iter_events(
    agent: Any,
    request: ChatRequest,
    assistant_buffer: list[str],
) -> AsyncIterator[str]:
    state = {
        "messages": _messages_for_turn(request.user_id, request.message),
        "user_id": request.user_id,
        "skill": request.skill,
        "ui_state": _empty_ui_state(),
        "tool_stats": {"requested": 0, "succeeded": 0, "failed": 0, "last_calls": []},
    }

    emitted_tool_call_ids: set[str] = set()
    try:
        async for mode, chunk in agent.astream(
            state, stream_mode=["messages", "updates", "custom"]
        ):
            if mode == "messages":
                message, _metadata = chunk
                text = getattr(message, "content", "")
                if isinstance(text, str) and text:
                    assistant_buffer.append(text)
                    yield _event_to_sse(ChatEvent(type="token", content=text))
            elif mode == "updates":
                for _node, update in chunk.items():
                    for msg in update.get("messages") or []:
                        if isinstance(msg, AIMessage):
                            for call in msg.tool_calls or []:
                                call_id = str(call.get("id") or "")
                                if call_id and call_id in emitted_tool_call_ids:
                                    continue
                                if call_id:
                                    emitted_tool_call_ids.add(call_id)
                                yield _event_to_sse(
                                    ChatEvent(
                                        type="tool_call",
                                        tool_name=str(call.get("name", "")),
                                        tool_args=_safe_tool_args(call),
                                    )
                                )
                        elif isinstance(msg, ToolMessage):
                            yield _event_to_sse(_tool_result_event(msg))
            elif mode == "custom":
                if isinstance(chunk, dict) and chunk.get("event") == "recall":
                    raw_hits = chunk.get("hits") or []
                    hits = [RecallHit(**h) for h in raw_hits if isinstance(h, dict)]
                    if hits:
                        yield _event_to_sse(
                            ChatEvent(type="recall", recall_hits=hits)
                        )
                elif isinstance(chunk, dict) and chunk.get("event") == "provenance":
                    yield _event_to_sse(
                        ChatEvent(
                            type="provenance",
                            provenance={
                                "provider": chunk.get("provider", ""),
                                "model": chunk.get("model", ""),
                                "tier": chunk.get("tier", ""),
                                "latency_ms": chunk.get("latency_ms", 0),
                                "cloud_call": chunk.get("cloud_call", False),
                                "cloud_payload_summary": chunk.get("cloud_payload_summary"),
                                "memories_recalled": chunk.get("memories_recalled", 0),
                                "skills_called": chunk.get("skills_called", []),
                                "rationale": chunk.get("rationale", ""),
                            },
                        )
                    )

        yield _event_to_sse(ChatEvent(type="done"))
    except Exception as exc:
        logger.exception("chat stream failed for user=%s", request.user_id)
        yield _event_to_sse(ChatEvent(type="error", content=str(exc)))


async def _iter_harness_events(
    loop: HarnessLoop,
    request: ChatRequest,
    assistant_buffer: list[str],
) -> AsyncIterator[str]:
    session, user_msg = _harness_turn_inputs(request)
    try:
        async for ev in loop.stream_turn(session, user_msg):
            if ev.type == "token":
                assistant_buffer.append(ev.content)
                yield _event_to_sse(ChatEvent(type="token", content=ev.content))
            elif ev.type == "tool_call":
                yield _event_to_sse(
                    ChatEvent(type="tool_call", tool_name=ev.tool_name, tool_args=ev.tool_args)
                )
            elif ev.type == "tool_result":
                yield _event_to_sse(
                    ChatEvent(type="tool_result", tool_name=ev.tool_name, tool_result=ev.tool_result)
                )
            elif ev.type == "recall":
                hits: list[RecallHit] = []
                for h in ev.recall_hits:
                    if isinstance(h, dict):
                        try:
                            hits.append(RecallHit(**h))
                        except Exception:
                            pass
                if hits:
                    yield _event_to_sse(ChatEvent(type="recall", recall_hits=hits))
            elif ev.type == "provenance" and ev.provenance is not None:
                yield _event_to_sse(
                    ChatEvent(
                        type="provenance",
                        provenance=_turn_provenance_dict(ev.provenance),
                    )
                )
            elif ev.type == "reasoning":
                # Reasoning is NOT added to assistant_buffer — it is not the answer
                # and must not be persisted or fed to memory extraction.
                yield _event_to_sse(ChatEvent(type="reasoning", content=ev.content))
            elif ev.type == "done":
                yield _event_to_sse(ChatEvent(type="done"))
    except Exception as exc:
        logger.exception("harness chat stream failed for user=%s", request.user_id)
        yield _event_to_sse(ChatEvent(type="error", content=str(exc)))


def _run_post_chat(user_id: str, user_text: str, assistant_buffer: list[str]) -> None:
    """Post-stream background task: persist the answer and extract memory.

    Runs after the SSE response is fully sent so the user never waits for
    the extractor LLM call. Any error here stays out of the user's view —
    memory extraction is best-effort; the chat already succeeded.
    """
    assistant_text = "".join(assistant_buffer).strip()
    if assistant_text:
        try:
            Memory(user_id).save_message("assistant", assistant_text)
        except Exception:
            logger.exception("assistant message save failed for user=%s", user_id)
    if not user_text.strip() and not assistant_text:
        return
    try:
        manager = MemoryManager(user_id)
        manager.extract({"user": user_text, "assistant": assistant_text})
    except Exception:
        logger.exception("post-chat extract failed for user=%s", user_id)


@router.get("/chat/history/{user_id}", response_model=ChatHistory)
def chat_history(user_id: str) -> ChatHistory:
    """Return the persisted chat transcript (oldest first) so the UI can rehydrate."""
    try:
        rows = Memory(user_id).load_chat()
    except Exception:
        logger.exception("chat history load failed for user=%s", user_id)
        rows = []
    return ChatHistory(
        messages=[
            ChatHistoryMessage(
                role=str(r.get("role", "")),
                content=str(r.get("content", "")),
                timestamp=str(r.get("timestamp", "")),
            )
            for r in rows
            if r.get("role") in ("user", "assistant")
        ]
    )


@router.post(
    "/chat",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Server-Sent Events stream. See ChatEvent for the frame payload schema.",
        }
    },
)
async def chat(
    request: ChatRequest,
    agent: Any = Depends(get_agent),
    harness_loop: HarnessLoop = Depends(get_harness_loop),
) -> StreamingResponse:
    """Stream June's reply as SSE. Each frame is a JSON-encoded ChatEvent."""
    assistant_buffer: list[str] = []
    stream = (
        _iter_harness_events(harness_loop, request, assistant_buffer)
        if _chat_use_harness()
        else _iter_events(agent, request, assistant_buffer)
    )
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
        background=BackgroundTask(
            _run_post_chat, request.user_id, request.message, assistant_buffer
        ),
    )
