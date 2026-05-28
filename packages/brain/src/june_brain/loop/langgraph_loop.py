"""LangGraph engine wrapped behind the HarnessLoop interface.

This module wraps the existing compiled LangGraph agent so both engines can be
measured side-by-side in the C.2 CLEAR experiment.  Do NOT modify graph.py.
"""

from __future__ import annotations

import logging
from typing import Any

from june_brain.providers.base import Message

from .interface import (
    SessionState,
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

    async def run_turn(self, session: SessionState, user_msg: Message) -> TurnResult:
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
            reply_text = str(response["messages"][-1].content)
        except Exception:
            log.exception("LangGraphLoop.run_turn failed")
            reply_text = "error: LangGraph agent call failed"

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

        provenance = TurnProvenance(
            tiers_used=tiers,
            cloud_call=cloud_call,
            model_ids=model_ids,
            rationale="Handled by LangGraph agent.",
        )

        return TurnResult(
            assistant_msg=Message(role="assistant", content=reply_text),
            tool_calls=[],
            provenance=provenance,
            tokens=TokenAccounting(),
            compacted=False,
        )
