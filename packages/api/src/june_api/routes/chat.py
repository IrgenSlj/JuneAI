"""POST /chat — SSE streaming chat endpoint."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from june_brain import graph as brain_graph

from ..schemas import ChatEvent, ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def get_agent() -> Any:
    """Resolve the LangGraph agent.

    Exposed as a FastAPI dependency so tests can stub it via
    ``app.dependency_overrides[get_agent] = ...`` without touching
    process-global state.
    """
    if brain_graph.june_agent is not None:
        return brain_graph.june_agent
    if brain_graph.startup_error:
        raise HTTPException(
            status_code=503,
            detail=f"June agent failed to start: {brain_graph.startup_error}",
        )
    return brain_graph.create_june_agent()


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


async def _iter_events(agent: Any, request: ChatRequest) -> AsyncIterator[str]:
    state = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        "skill": request.skill,
        "ui_state": _empty_ui_state(),
        "tool_stats": {"requested": 0, "succeeded": 0, "failed": 0, "last_calls": []},
    }

    emitted_tool_call_ids: set[str] = set()
    try:
        async for mode, chunk in agent.astream(
            state, stream_mode=["messages", "updates"]
        ):
            if mode == "messages":
                message, _metadata = chunk
                text = getattr(message, "content", "")
                if isinstance(text, str) and text:
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

        yield _event_to_sse(ChatEvent(type="done"))
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat stream failed for user=%s", request.user_id)
        yield _event_to_sse(ChatEvent(type="error", content=str(exc)))


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
    return StreamingResponse(
        _iter_events(agent, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
