"""Hand-written harness loop implementing the fixed shape:

    assemble_context -> call_provider -> (tool calls? dispatch -> observe -> repeat : done)
                     -> maybe_compact

Dynamic choices (tier, skills) flow as DATA through this fixed shape — the shape itself
never changes and is never self-modified.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from june_brain.context.assembler import ContextAssembler
from june_brain.context.compactor import Compactor
from june_brain.context.tokens import estimate_tokens
from june_brain.providers.base import GenerateRequest, GenerateResult, Message
from june_brain.providers.registry import ProviderRegistry, get_registry

from .interface import (
    HarnessLoop,
    SessionState,
    StreamEvent,
    TokenAccounting,
    ToolCall,
    TurnProvenance,
    TurnResult,
)
from .reasoning import ReasoningSplitter, split_reasoning
from .trace import TraceStore, TurnTrace
from .wiring import is_network_tool as _is_network_tool

# Per-call token cap. A cap, not a target — short answers still finish early.
# Set high enough that thinking models (qwen3) can reason AND answer within one
# call rather than spending the whole budget in <think> and emitting nothing.
_MAX_TOKENS = 2048


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
        classify: Callable[[str], Awaitable[Any]] | None = None,
        max_iterations: int = 6,
    ) -> None:
        self._registry = registry if registry is not None else get_registry()
        self._role = role
        # Difficulty classifier seam: defaults to the registry-backed model
        # classifier; injectable so tests with scripted provider responses are
        # not perturbed by the routing classification call.
        self._classify = classify

        # --- recall state (shared mutable dict populated by each assemble call) ---
        self._recall_state: dict[str, Any] = {"memories_recalled": 0, "recall_hits": []}

        # The real assembler instance (None when assemble_context is injected),
        # so reasoning can be gated per turn via set_reason without changing the
        # injected callable's signature.
        self._assembler: ContextAssembler | None = None
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

            from .wiring import make_recall_fn, make_tools_block

            recall_fn, self._recall_state = make_recall_fn()
            _assembler = ContextAssembler(
                character_block=character_block,
                recall=recall_fn,
                tools_block=make_tools_block(),
                reason=False,  # gated per turn by difficulty (S4.3)
            )
            self._assembler = _assembler
            self._assemble_context = _assembler.assemble

        # --- tool-call extraction ---
        if extract_tool_calls is not None:
            self._extract_tool_calls = extract_tool_calls
        else:
            from .wiring import make_extract_tool_calls_fn

            self._extract_tool_calls = make_extract_tool_calls_fn()

        # --- dispatch ---
        self._dispatched_names: list[str] = []
        self._network_calls: list[str] = []
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
        self._trace_store = TraceStore()

    # ------------------------------------------------------------------
    # Private helpers shared by run_turn and stream_turn
    # ------------------------------------------------------------------

    @staticmethod
    def _egress_blocked() -> bool:
        """True when the user's privacy dial is Local-only — networked tools are
        blocked and the loop offers to switch instead of silently egressing."""
        try:
            from june_brain.config_store import get_privacy_dial
            from june_brain.routing import UserPrivacyDial

            return get_privacy_dial() == UserPrivacyDial.LOCAL_ONLY
        except Exception:
            return False

    @staticmethod
    def _render_prompt(ctx: list[Message]) -> str:
        """Render the assembled context exactly as the model receives it.

        This is the "LLM factory" input — the full prompt including character,
        recalled memories, pinned state, and history.
        """
        return "\n\n".join(f"[{m.role}]\n{m.content}" for m in ctx)

    def _reset_per_turn(self) -> None:
        """Reset per-turn tracking state."""
        self._dispatched_names.clear()
        self._network_calls.clear()
        self._recall_state["memories_recalled"] = 0
        self._recall_state["recall_hits"] = []

    async def _route(self, user_msg: Message) -> tuple[Any, str, Any]:
        """Difficulty-based role/tier routing with registry fallback.

        Classifies difficulty with the model classifier (cached, time-boxed,
        heuristic fallback), picks the tier, and resolves the provider.
        Returns (provider, chosen_role, DifficultyResult).
        """
        from june_brain.router.difficulty import (
            DifficultyResult,
            classify_difficulty_detailed,
            heuristic_difficulty,
            tier_for_difficulty,
        )

        try:
            if self._classify is not None:
                difficulty = await self._classify(user_msg.content)
            else:
                difficulty = await classify_difficulty_detailed(
                    user_msg.content, registry=self._registry
                )
        except Exception:
            difficulty = DifficultyResult(
                heuristic_difficulty(user_msg.content), "heuristic"
            )

        try:
            routed_role = tier_for_difficulty(difficulty.label)
        except Exception:
            routed_role = self._role

        try:
            provider = self._registry.get(routed_role)
            chosen_role = routed_role
        except Exception:
            provider = self._registry.get(self._role)
            chosen_role = self._role

        return provider, chosen_role, difficulty

    def _apply_reason(self, label: str) -> None:
        """Gate the <think> reasoning instruction by difficulty.

        Trivial/standard turns skip the thinking pass (latency win); hard and
        creative keep it. No-op when the assembler is injected (tests own it).
        """
        if self._assembler is not None:
            self._assembler.set_reason(label in ("hard", "creative"))

    def _build_provenance(
        self,
        provider: Any,
        chosen_role: str,
        all_tool_calls: list[ToolCall],
        start_time: float,
        tokens: TokenAccounting,
        difficulty: Any = None,
    ) -> TurnProvenance:
        """Build TurnProvenance from accumulated state."""
        difficulty_label = getattr(difficulty, "label", "") or ""
        difficulty_source = getattr(difficulty, "source", "") or ""
        memories_recalled = self._recall_state.get("memories_recalled", 0)
        skills_called = list(self._dispatched_names)
        egress = list(dict.fromkeys(self._network_calls))  # de-dupe, keep order
        is_cloud = provider.tier == "cloud-capable"

        egress_note = (
            f" Network egress via {', '.join(egress)}." if egress else ""
        )

        if is_cloud:
            rationale = f"Escalated to {provider.model_id} ({chosen_role})." + egress_note
            ctx_len = len(all_tool_calls) + 1
            cloud_payload_summary: str | None = (
                f"Sent {ctx_len} context messages"
                + (f" + recalled {memories_recalled} memories" if memories_recalled else "")
                + f" to {provider.model_id}."
            )
        else:
            rationale = (
                f"Handled locally via {provider.model_id} ({chosen_role});"
                f" recalled {memories_recalled} memories." + egress_note
            )
            cloud_payload_summary = None

        if difficulty_label:
            src = f" via {difficulty_source}" if difficulty_source else ""
            rationale = f"{rationale} Difficulty: {difficulty_label}{src}."

        return TurnProvenance(
            tiers_used=[chosen_role],
            cloud_call=is_cloud,
            model_ids=[provider.model_id],
            memories_recalled=memories_recalled,
            skills_called=skills_called,
            latency_ms=max(0, int((time.monotonic() - start_time) * 1000)),
            rationale=rationale,
            cloud_payload_summary=cloud_payload_summary,
            egress=egress,
            difficulty=difficulty_label,
            difficulty_source=difficulty_source,
        )

    # ------------------------------------------------------------------
    # run_turn — non-streaming path (used by CLEAR experiment and callers
    # that don't need streaming)
    # ------------------------------------------------------------------

    async def run_turn(self, session: SessionState, user_msg: Message) -> TurnResult:
        _start = time.monotonic()

        self._reset_per_turn()
        provider, chosen_role, difficulty = await self._route(user_msg)
        self._apply_reason(difficulty.label)

        tokens = TokenAccounting()
        all_tool_calls: list[ToolCall] = []
        last_result: GenerateResult | None = None

        for _ in range(self._max_iterations):
            ctx = self._assemble_context(session, user_msg)
            result = await provider.generate(
                GenerateRequest(messages=ctx, max_tokens=_MAX_TOKENS)
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

        provenance = self._build_provenance(
            provider, chosen_role, all_tool_calls, _start, tokens, difficulty
        )

        _, answer_text = split_reasoning(last_result.text)

        return TurnResult(
            assistant_msg=Message(role="assistant", content=answer_text),
            tool_calls=all_tool_calls,
            provenance=provenance,
            tokens=tokens,
            compacted=compacted,
        )

    # ------------------------------------------------------------------
    # stream_turn — true token-streaming path
    # ------------------------------------------------------------------

    async def stream_turn(
        self, session: SessionState, user_msg: Message
    ) -> AsyncIterator[StreamEvent]:
        _start = time.monotonic()

        self._reset_per_turn()
        trace = TurnTrace(turn_id=uuid.uuid4().hex, user_id=session.user_id)
        block_egress = self._egress_blocked()
        try:
            provider, chosen_role, difficulty = await self._route(user_msg)
        except Exception:
            # Graceful degradation: no provider found
            yield StreamEvent(type="token", content="error: no provider available")
            trace.record("error", "no provider available")
            self._trace_store.write(trace)
            yield StreamEvent(type="done")
            return

        self._apply_reason(difficulty.label)

        route_summary = f"route -> {chosen_role} ({provider.model_id})"
        route_detail = (
            f"tier: {provider.tier}\nmodel: {provider.model_id}\nrole: {chosen_role}\n"
            f"difficulty: {difficulty.label} ({difficulty.source})"
        )
        yield StreamEvent(type="iteration", content=route_summary, detail=route_detail)
        trace.record("iteration", route_summary, detail=route_detail)

        tokens = TokenAccounting()
        all_tool_calls: list[ToolCall] = []
        first_iteration = True

        for iteration_idx in range(self._max_iterations):
            ctx = self._assemble_context(session, user_msg)

            # Emit the rendered prompt — the "LLM factory" input — as a trace
            # event. content stays empty; the full prompt rides in detail.
            rendered_prompt = self._render_prompt(ctx)
            yield StreamEvent(
                type="prompt",
                content=f"prompt assembled ({len(ctx)} messages)",
                detail=rendered_prompt,
                iteration=iteration_idx,
            )
            trace.record(
                "prompt",
                f"prompt assembled ({len(ctx)} messages)",
                detail=rendered_prompt,
            )

            # Emit recall event once on the first iteration if we have hits
            if first_iteration:
                recall_hits = self._recall_state.get("recall_hits", [])
                if recall_hits:
                    yield StreamEvent(type="recall", recall_hits=list(recall_hits))
                    trace.record(
                        "recall",
                        f"recall · {len(recall_hits)} memories",
                        detail="\n".join(
                            str(h.get("text", h)) for h in recall_hits
                        ),
                    )
                first_iteration = False

            reasoning_accum = ""

            # --- stream the provider response ---
            accumulated = ""
            suppress_mode = False
            emit_mode = False
            buffered_head = ""
            splitter = ReasoningSplitter()

            try:
                async for delta in provider.stream(
                    GenerateRequest(messages=ctx, max_tokens=_MAX_TOKENS)
                ):
                    accumulated += delta

                    # Feed delta through the reasoning splitter first.
                    segments = splitter.feed(delta)
                    for seg_kind, seg_text in segments:
                        if seg_kind == "reasoning":
                            if seg_text:
                                reasoning_accum += seg_text
                                yield StreamEvent(type="reasoning", content=seg_text)
                        else:
                            # seg_kind == "answer": pass through the
                            # tool-call-suppression gate exactly as before.
                            answer_delta = seg_text
                            if not answer_delta:
                                continue

                            if not emit_mode and not suppress_mode:
                                buffered_head += answer_delta
                                stripped = buffered_head.lstrip()

                                candidate = stripped
                                if candidate.startswith("```"):
                                    newline_pos = candidate.find("\n")
                                    if newline_pos != -1:
                                        candidate = candidate[newline_pos + 1:].lstrip()
                                    else:
                                        continue

                                if candidate:
                                    if candidate[0] in ("{", "["):
                                        suppress_mode = True
                                    else:
                                        emit_mode = True
                                        if buffered_head:
                                            yield StreamEvent(type="token", content=buffered_head)
                                        buffered_head = ""
                            elif emit_mode:
                                yield StreamEvent(type="token", content=answer_delta)

                # Flush any residual reasoning/answer at end-of-stream.
                for seg_kind, seg_text in splitter.flush():
                    if seg_kind == "reasoning":
                        if seg_text:
                            reasoning_accum += seg_text
                            yield StreamEvent(type="reasoning", content=seg_text)
                    else:
                        answer_delta = seg_text
                        if not answer_delta:
                            continue
                        if not emit_mode and not suppress_mode:
                            buffered_head += answer_delta
                            stripped = buffered_head.lstrip()
                            candidate = stripped
                            if candidate.startswith("```"):
                                newline_pos = candidate.find("\n")
                                if newline_pos != -1:
                                    candidate = candidate[newline_pos + 1:].lstrip()
                            if candidate:
                                if candidate[0] in ("{", "["):
                                    suppress_mode = True
                                else:
                                    emit_mode = True
                                    if buffered_head:
                                        yield StreamEvent(type="token", content=buffered_head)
                                    buffered_head = ""
                        elif emit_mode:
                            yield StreamEvent(type="token", content=answer_delta)

            except Exception:
                # Stream failed — fall back to a single generate call
                try:
                    result = await provider.generate(
                        GenerateRequest(messages=ctx, max_tokens=_MAX_TOKENS)
                    )
                    accumulated = result.text
                    tokens.input_tokens += result.input_tokens
                    tokens.output_tokens += result.output_tokens
                    suppress_mode = False
                    emit_mode = True
                    _, fallback_answer = split_reasoning(accumulated)
                    yield StreamEvent(type="token", content=fallback_answer)
                except Exception:
                    accumulated = ""

            # Accumulate token estimates for the streamed response
            tokens.input_tokens += estimate_tokens(" ".join(m.content for m in ctx))
            tokens.output_tokens += estimate_tokens(accumulated)

            # Build a GenerateResult-like object for tool-call extraction
            pseudo_result = GenerateResult(
                text=accumulated,
                input_tokens=estimate_tokens(" ".join(m.content for m in ctx)),
                output_tokens=estimate_tokens(accumulated),
                latency_ms=0,
                model_id=provider.model_id,
                tier=provider.tier,
            )

            tool_calls = self._extract_tool_calls(pseudo_result)

            # Record + stream this iteration's internals: the raw intermediate
            # model output (incl. any suppressed tool JSON / think blocks) and
            # how many tool calls were parsed out of it. This is the glass box —
            # the unedited model output behind the cleaned answer tokens.
            iter_summary = f"iteration {iteration_idx} · {len(tool_calls)} tool call(s)"
            iter_detail = accumulated or "(no output)"
            yield StreamEvent(
                type="iteration",
                content=iter_summary,
                detail=iter_detail,
                iteration=iteration_idx,
            )
            trace.record("iteration", iter_summary, detail=iter_detail)
            if reasoning_accum:
                trace.record("reasoning", "reasoning", detail=reasoning_accum)

            if tool_calls and self._dispatch is not None:
                all_tool_calls.extend(tool_calls)

                # Partition: networked tools are blocked under a Local-only dial
                # rather than silently egressing. Blocked calls are not dispatched;
                # June is told (via an observation) to surface the choice to the user.
                runnable: list[ToolCall] = []
                observations: list[Message] = []
                for tc in tool_calls:
                    args_detail = json.dumps(tc.args, ensure_ascii=False, indent=2)
                    networked = _is_network_tool(tc.name)
                    if networked and block_egress:
                        yield StreamEvent(
                            type="tool_blocked",
                            tool_name=tc.name,
                            tool_args=tc.args,
                            detail=args_detail,
                            network=True,
                        )
                        trace.record(
                            "tool_blocked",
                            f"blocked · {tc.name} (local-only)",
                            detail=args_detail,
                        )
                        observations.append(
                            Message(
                                role="tool",
                                content=(
                                    f"BLOCKED: '{tc.name}' needs the internet, which "
                                    "Local-only mode does not allow. Tell the user you "
                                    "could not do this in Local-only mode and that they "
                                    "can switch to Private-by-default to enable it."
                                ),
                            )
                        )
                        continue
                    if networked:
                        self._network_calls.append(tc.name)
                    yield StreamEvent(
                        type="tool_call",
                        tool_name=tc.name,
                        tool_args=tc.args,
                        detail=args_detail,
                        network=networked,
                    )
                    egress_note = " [egress: leaves your machine]" if networked else ""
                    trace.record(
                        "tool_call", f"tool · {tc.name}{egress_note}", detail=args_detail
                    )
                    runnable.append(tc)

                if runnable:
                    run_obs = await self._dispatch(runnable, session)
                    for i, obs_msg in enumerate(run_obs):
                        tc_name = runnable[i].name if i < len(runnable) else ""
                        yield StreamEvent(
                            type="tool_result",
                            tool_name=tc_name,
                            tool_result=obs_msg.content,
                            detail=obs_msg.content,
                        )
                        trace.record(
                            "tool_result", f"result · {tc_name}", detail=obs_msg.content
                        )
                    observations.extend(run_obs)

                tool_turn = Message(role="assistant", content=accumulated)
                session.messages.append(tool_turn)
                session.messages.extend(observations)
                # Reset classification state for next iteration
                suppress_mode = False
                emit_mode = False
                buffered_head = ""
            else:
                # No tool calls — if we suppressed (looked like JSON but wasn't a real
                # tool call), emit the accumulated text now so the user sees something
                if suppress_mode and accumulated:
                    _, suppressed_answer = split_reasoning(accumulated)
                    yield StreamEvent(type="token", content=suppressed_answer)
                break

        compacted = await self._maybe_compact(session)
        if compacted:
            yield StreamEvent(
                type="compaction",
                content="conversation compacted",
                detail="The conversation history was compacted into the pinned-state anchor.",
            )
            trace.record(
                "compaction",
                "conversation compacted",
                detail="History compacted into the pinned-state anchor.",
            )

        provenance = self._build_provenance(
            provider, chosen_role, all_tool_calls, _start, tokens, difficulty
        )
        yield StreamEvent(type="provenance", provenance=provenance)
        trace.record(
            "provenance",
            provenance.rationale,
            detail=(
                f"tiers: {', '.join(provenance.tiers_used)}\n"
                f"models: {', '.join(provenance.model_ids)}\n"
                f"cloud_call: {provenance.cloud_call}\n"
                f"memories_recalled: {provenance.memories_recalled}\n"
                f"skills_called: {', '.join(provenance.skills_called) or '(none)'}\n"
                f"egress: {', '.join(provenance.egress) or '(none)'}\n"
                f"latency_ms: {provenance.latency_ms}\n"
                f"tokens: in={tokens.input_tokens} out={tokens.output_tokens}"
                + (
                    f"\ncloud_payload: {provenance.cloud_payload_summary}"
                    if provenance.cloud_payload_summary
                    else ""
                )
            ),
        )

        trace.record("done", "done")
        self._trace_store.write(trace)
        yield StreamEvent(type="done")


# Satisfy the Protocol at import time (runtime_checkable check in tests).
def _assert_protocol() -> None:
    assert isinstance(HandwrittenLoop(), HarnessLoop)
