"""Hand-written harness loop implementing the fixed shape:

    assemble_context -> call_provider -> (tool calls? dispatch -> observe -> repeat : done)
                     -> maybe_compact

Dynamic choices (tier, skills) flow as DATA through this fixed shape — the shape itself
never changes and is never self-modified.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from june_brain.providers.base import GenerateRequest, GenerateResult, Message
from june_brain.providers.registry import ProviderRegistry, get_registry

from .interface import (
    HarnessLoop,
    SessionState,
    TokenAccounting,
    ToolCall,
    TurnProvenance,
    TurnResult,
)

_SYSTEM_PROMPT = (
    "You are June, a personal assistant. Be helpful, honest, and concise."
)


def _default_assemble_context(session: SessionState, user_msg: Message) -> list[Message]:
    """Minimal context assembler: system + history + new user message.

    The full C.3 assembler (pinned state, salience-ranked recall, character block)
    slots in here as a drop-in replacement.
    """
    system = Message(role="system", content=_SYSTEM_PROMPT)
    return [system, *session.messages, user_msg]


def _default_extract_tool_calls(result: GenerateResult) -> list[ToolCall]:
    """Extension point for tool-call parsing.

    The C.1 provider returns text only; real structured tool-call parsing
    arrives with later tasks and replaces this function via injection.
    """
    return []


def _default_maybe_compact(session: SessionState) -> bool:  # noqa: ARG001
    """Compaction stub — real logic is C.3.  Always returns False."""
    return False


class HandwrittenLoop:
    """A plain async while-loop implementing the fixed harness shape.

    All seams are injectable so the class is fully testable without Ollama.
    """

    def __init__(
        self,
        registry: ProviderRegistry | None = None,
        role: str = "local-fast",
        assemble_context: Callable[[SessionState, Message], list[Message]] | None = None,
        extract_tool_calls: Callable[[GenerateResult], list[ToolCall]] | None = None,
        dispatch: (
            Callable[[list[ToolCall], SessionState], Awaitable[list[Message]]] | None
        ) = None,
        maybe_compact: Callable[[SessionState], bool] | None = None,
        max_iterations: int = 6,
    ) -> None:
        self._registry = registry if registry is not None else get_registry()
        self._role = role
        self._assemble_context = assemble_context or _default_assemble_context
        self._extract_tool_calls = extract_tool_calls or _default_extract_tool_calls
        self._dispatch = dispatch
        self._maybe_compact = maybe_compact or _default_maybe_compact
        self._max_iterations = max_iterations

    async def run_turn(self, session: SessionState, user_msg: Message) -> TurnResult:
        provider = self._registry.get(self._role)
        tokens = TokenAccounting()
        all_tool_calls: list[ToolCall] = []
        last_result: GenerateResult | None = None

        for _ in range(self._max_iterations):
            ctx = self._assemble_context(session, user_msg)
            result = await provider.generate(
                GenerateRequest(messages=ctx, max_tokens=512)
            )
            last_result = result
            tokens.input_tokens += result.input_tokens
            tokens.output_tokens += result.output_tokens

            tool_calls = self._extract_tool_calls(result)
            if tool_calls and self._dispatch is not None:
                all_tool_calls.extend(tool_calls)
                observations: list[Message] = await self._dispatch(tool_calls, session)
                tool_turn = Message(role="assistant", content=result.text)
                session.messages.append(tool_turn)
                session.messages.extend(observations)
            else:
                break

        assert last_result is not None
        compacted = self._maybe_compact(session)

        provenance = TurnProvenance(
            tiers_used=[self._role],
            cloud_call=(provider.tier == "cloud-capable"),
            model_ids=[provider.model_id],
            rationale=f"Handled locally via {provider.model_id} ({self._role}).",
        )

        return TurnResult(
            assistant_msg=Message(role="assistant", content=last_result.text),
            tool_calls=all_tool_calls,
            provenance=provenance,
            tokens=tokens,
            compacted=compacted,
        )


# Satisfy the Protocol at import time (runtime_checkable check in tests).
def _assert_protocol() -> None:
    assert isinstance(HandwrittenLoop(), HarnessLoop)
