"""POST /chat — SSE streaming chat endpoint."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from june_brain.loop.interface import HarnessLoop, SessionState, TurnProvenance, TurnResult
from june_brain.memory import Memory, MemoryManager
from june_brain.providers.base import Message
from starlette.background import BackgroundTask

from ..schemas import ChatEvent, ChatHistory, ChatHistoryMessage, ChatRequest, RecallHit

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def get_harness_loop() -> HarnessLoop:
    """Resolve the live-chat HarnessLoop implementation.

    Exposed as a FastAPI dependency so tests can stub it via
    ``app.dependency_overrides[get_harness_loop] = ...`` without touching
    process-global state.  Returns the hand-written loop — the one engine
    (ADR 0018).
    """
    from june_brain.loop.engine import get_loop

    return get_loop()


def _event_to_sse(event: ChatEvent) -> str:
    """Format a ChatEvent as one SSE data frame."""
    return f"data: {event.model_dump_json()}\n\n"


def _messages_for_turn(user_id: str, user_text: str) -> list[Any]:
    """Persist the user turn and return the conversation context for the loop."""
    if not user_text.strip():
        return [Message(role="user", content=user_text)]
    try:
        memory = Memory(user_id)
        memory.save_message("user", user_text)
        return memory.load_chat_messages()
    except Exception:
        logger.exception("chat history load failed for user=%s", user_id)
        return [Message(role="user", content=user_text)]


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
        "egress": list(getattr(prov, "egress", []) or []),
        "difficulty": getattr(prov, "difficulty", "") or "",
        "difficulty_source": getattr(prov, "difficulty_source", "") or "",
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
                    ChatEvent(
                        type="tool_call",
                        tool_name=ev.tool_name,
                        tool_args=ev.tool_args,
                        detail=ev.detail,
                        network=ev.network,
                    )
                )
            elif ev.type == "tool_result":
                yield _event_to_sse(
                    ChatEvent(
                        type="tool_result",
                        tool_name=ev.tool_name,
                        tool_result=ev.tool_result,
                        detail=ev.detail,
                    )
                )
            elif ev.type in ("prompt", "iteration", "compaction"):
                # Glass-box trace events: collapsed line in content, full body
                # in detail. Never added to assistant_buffer (not the answer).
                yield _event_to_sse(
                    ChatEvent(type=ev.type, content=ev.content, detail=ev.detail)
                )
            elif ev.type == "tool_blocked":
                # A networked tool was blocked by Local-only mode. The UI renders
                # the one-click "switch and retry" affordance from this.
                yield _event_to_sse(
                    ChatEvent(
                        type="tool_blocked",
                        tool_name=ev.tool_name,
                        tool_args=ev.tool_args,
                        detail=ev.detail,
                        network=True,
                    )
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
    harness_loop: HarnessLoop = Depends(get_harness_loop),
) -> StreamingResponse:
    """Stream June's reply as SSE. Each frame is a JSON-encoded ChatEvent."""
    assistant_buffer: list[str] = []
    stream = _iter_harness_events(harness_loop, request, assistant_buffer)
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
