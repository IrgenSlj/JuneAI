"""POST /chat — SSE streaming chat endpoint."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from june_brain import graph as brain_graph
from june_brain.memory import Memory, MemoryManager
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from starlette.background import BackgroundTask

from ..schemas import ChatEvent, ChatRequest, RecallHit

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


def _event_to_sse(event: ChatEvent) -> str:
    """Format a ChatEvent as one SSE data frame."""
    return f"data: {event.model_dump_json()}\n\n"


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
            tool_args=dict(call.get("args", {})),
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
    except Exception as exc:  # noqa: BLE001
        logger.warning("chat history load failed for user=%s: %s", user_id, exc)
        return [HumanMessage(content=user_text)]


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
                                        tool_args=dict(call.get("args", {})),
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

        yield _event_to_sse(ChatEvent(type="done"))
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat stream failed for user=%s", request.user_id)
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
        except Exception as exc:  # noqa: BLE001
            logger.warning("assistant message save failed for user=%s: %s", user_id, exc)
    if not user_text.strip() and not assistant_text:
        return
    try:
        manager = MemoryManager(user_id)
        manager.extract({"user": user_text, "assistant": assistant_text})
    except Exception as exc:  # noqa: BLE001
        logger.warning("post-chat extract failed for user=%s: %s", user_id, exc)


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
) -> StreamingResponse:
    """Stream June's reply as SSE. Each frame is a JSON-encoded ChatEvent."""
    assistant_buffer: list[str] = []
    return StreamingResponse(
        _iter_events(agent, request, assistant_buffer),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
        background=BackgroundTask(
            _run_post_chat, request.user_id, request.message, assistant_buffer
        ),
    )
