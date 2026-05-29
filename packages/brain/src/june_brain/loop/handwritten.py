"""Hand-written harness loop implementing the fixed shape:

    assemble_context -> call_provider -> (tool calls? dispatch -> observe -> repeat : done)
                     -> maybe_compact

Dynamic choices (tier, skills) flow as DATA through this fixed shape — the shape itself
never changes and is never self-modified.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from june_brain.context.assembler import ContextAssembler
from june_brain.context.compactor import Compactor
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
        maybe_compact: Callable[[SessionState], Awaitable[bool]] | None = None,
        max_iterations: int = 6,
    ) -> None:
        self._registry = registry if registry is not None else get_registry()
        self._role = role

        # --- recall state (shared mutable dict populated by each assemble call) ---
        self._recall_state: dict[str, Any] = {"memories_recalled": 0, "recall_hits": []}

        if assemble_context is not None:
            self._assemble_context = assemble_context
            self._recall_state_external = True  # caller owns recall tracking
        else:
            self._recall_state_external = False
            character_block: str | None = None
            try:
                from june_brain.character import load_or_seed, shaping_section

                block = load_or_seed()
                character_block = block.to_block() + "\n\n" + shaping_section(block)
            except Exception:
                pass

            from .wiring import make_recall_fn

            recall_fn, self._recall_state = make_recall_fn()
            _assembler = ContextAssembler(
                character_block=character_block,
                recall=recall_fn,
            )
            self._assemble_context = _assembler.assemble

        # --- tool-call extraction ---
        if extract_tool_calls is not None:
            self._extract_tool_calls = extract_tool_calls
        else:
            from .wiring import make_extract_tool_calls_fn

            self._extract_tool_calls = make_extract_tool_calls_fn()

        # --- dispatch ---
        self._dispatched_names: list[str] = []
        if dispatch is not None:
            self._dispatch: Callable[[list[ToolCall], SessionState], Awaitable[list[Message]]] | None = dispatch
        else:
            from .wiring import make_dispatch_fn

            self._dispatched_names = []
            self._dispatch = make_dispatch_fn(self._dispatched_names)

        # --- compaction ---
        if maybe_compact is not None:
            self._maybe_compact = maybe_compact
        else:
            _compactor = Compactor(registry=self._registry, role=self._role)
            self._maybe_compact = _compactor.compact

        self._max_iterations = max_iterations

    async def run_turn(self, session: SessionState, user_msg: Message) -> TurnResult:
        _start = time.monotonic()

        # Reset per-turn tracking
        self._dispatched_names.clear()
        self._recall_state["memories_recalled"] = 0
        self._recall_state["recall_hits"] = []

        # --- difficulty-based role/tier routing ---
        try:
            from june_brain.router.difficulty import heuristic_difficulty, tier_for_difficulty

            routed_role = tier_for_difficulty(heuristic_difficulty(user_msg.content))
        except Exception:
            routed_role = self._role

        # Fall back to self._role if the routed role is not in the registry
        try:
            provider = self._registry.get(routed_role)
            chosen_role = routed_role
        except Exception:
            provider = self._registry.get(self._role)
            chosen_role = self._role

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
        compacted = await self._maybe_compact(session)

        # --- provenance enrichment ---
        memories_recalled = self._recall_state.get("memories_recalled", 0)
        skills_called = list(self._dispatched_names)
        is_cloud = provider.tier == "cloud-capable"

        if is_cloud:
            rationale = (
                f"Escalated to {provider.model_id} ({chosen_role})."
            )
            ctx_len = len(all_tool_calls) + 1  # rough context message count
            cloud_payload_summary: str | None = (
                f"Sent {ctx_len} context messages"
                + (f" + recalled {memories_recalled} memories" if memories_recalled else "")
                + f" to {provider.model_id}."
            )
        else:
            rationale = (
                f"Handled locally via {provider.model_id} ({chosen_role});"
                f" recalled {memories_recalled} memories."
            )
            cloud_payload_summary = None

        provenance = TurnProvenance(
            tiers_used=[chosen_role],
            cloud_call=is_cloud,
            model_ids=[provider.model_id],
            memories_recalled=memories_recalled,
            skills_called=skills_called,
            latency_ms=max(0, int((time.monotonic() - _start) * 1000)),
            rationale=rationale,
            cloud_payload_summary=cloud_payload_summary,
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
