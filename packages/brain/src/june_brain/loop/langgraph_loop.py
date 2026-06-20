"""LangGraph engine wrapped behind the HarnessLoop interface.

This module wraps the existing compiled LangGraph agent so both engines can be
measured side-by-side in the C.2 CLEAR experiment.  Do NOT modify graph.py.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from june_brain.providers.base import Message

from .interface import (
    SessionState,
    StreamEvent,
    TokenAccounting,
    TurnProvenance,
    TurnResult,
)

log = logging.getLogger(__name__)


def _to_lc_message(msg: Message) -> Any:
    """Convert a providers.base.Message to a LangChain message object."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    if msg.role == "system":
        return SystemMessage(content=msg.content)
    if msg.role == "assistant":
        return AIMessage(content=msg.content)
    return HumanMessage(content=msg.content)


class LangGraphLoop:
    """Wraps the existing LangGraph agent behind the HarnessLoop interface.

    The agent is injected (default: lazy call to get_or_create_agent() on the
    first run_turn) so tests can pass a fake without touching the real agent.
    """

    def __init__(self, agent: Any | None = None) -> None:
        self._agent = agent
        self._agent_resolved = agent is not None

    def _get_agent(self) -> Any:
        if not self._agent_resolved:
            from june_brain.graph import get_or_create_agent

            self._agent = get_or_create_agent()
            self._agent_resolved = True
        return self._agent

    @staticmethod
    def _build_token_collector() -> Any:
        """Build a non-streaming chat model to measure token usage.

        The streaming model used by the agent loses ``token_usage`` (OpenAI
        returns it only in the final non-streaming response), so we build a
        separate non-streaming copy just for token counting.
        """
        from langchain_openai import ChatOpenAI

        from june_brain.config import resolve_runtime_config

        rc = resolve_runtime_config()
        return ChatOpenAI(
            model=rc.model,
            api_key=rc.api_key or "unused",  # type: ignore[arg-type]
            base_url=rc.base_url,
            temperature=rc.temperature,
            max_completion_tokens=rc.max_tokens,
            streaming=False,
            timeout=120,
        )

    async def run_turn(self, session: SessionState, user_msg: Message) -> TurnResult:
        _start = time.monotonic()
        try:
            agent = self._get_agent()
            lc_messages = [_to_lc_message(m) for m in session.messages]
            lc_messages.append(_to_lc_message(user_msg))

            response = agent.invoke(
                {
                    "messages": lc_messages,
                    "user_id": session.user_id,
                    "skill": session.skill,
                }
            )
            last_msg = response["messages"][-1]
            reply_text = str(last_msg.content)

            tokens = TokenAccounting()
            try:
                tc = self._build_token_collector()
                tc_resp = tc.invoke(lc_messages)
                rm = tc_resp.response_metadata or {}
                tu = rm.get("token_usage", {}) or {}
                tokens.input_tokens = tu.get("prompt_tokens", 0)
                tokens.output_tokens = tu.get("completion_tokens", 0)
            except Exception:
                log.warning("token collection failed", exc_info=True)
        except Exception:
            log.exception("LangGraphLoop.run_turn failed")
            reply_text = "error: LangGraph agent call failed"
            tokens = TokenAccounting()

        try:
            from june_brain.config import resolve_runtime_config

            rc = resolve_runtime_config()
            cloud_call = rc.is_api
            tiers = ["cloud-capable"] if cloud_call else ["local-fast"]
            model_ids = [rc.model]
        except Exception:
            cloud_call = False
            tiers = ["local-fast"]
            model_ids = ["unknown"]

        _tier = tiers[0]
        provenance = TurnProvenance(
            tiers_used=tiers,
            cloud_call=cloud_call,
            model_ids=model_ids,
            latency_ms=max(0, int((time.monotonic() - _start) * 1000)),
            rationale=f"Handled by LangGraph agent via {model_ids[0]} ({_tier}).",
        )

        return TurnResult(
            assistant_msg=Message(role="assistant", content=reply_text),
            tool_calls=[],
            provenance=provenance,
            tokens=tokens,
            compacted=False,
        )

    async def stream_turn(
        self, session: SessionState, user_msg: Message
    ) -> AsyncIterator[StreamEvent]:
        """Minimal streaming wrapper: delegates to run_turn then yields events."""
        result = await self.run_turn(session, user_msg)
        yield StreamEvent(type="token", content=result.assistant_msg.content)
        for tc in result.tool_calls:
            yield StreamEvent(type="tool_call", tool_name=tc.name, tool_args=tc.args)
        yield StreamEvent(type="provenance", provenance=result.provenance)
        yield StreamEvent(type="done")
