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

from ..failure import degrade_quietly
from .interface import (
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


def _local_model_ready(model_id: str, base_url: str) -> bool:
    """True when Ollama is up AND the given tag is pulled.

    Only returns False for a *specifically missing* model on a *reachable*
    Ollama. When Ollama is unreachable we return True so we do NOT spuriously
    'degrade' (the fast tier would be just as unreachable) — that failure is a
    different concern handled elsewhere.
    """
    from june_brain.ollama_manager import is_model_available, is_ollama_running
    try:
        if not is_ollama_running(base_url):
            return True
        return is_model_available(model_id, base_url)
    except Exception:
        return True


def _fmt_tok(n: int | None) -> str | None:
    """Format a token count for Glass Box labels: k-suffix for >=1000, else str."""
    if n is None:
        return None
    return f"{n // 1000}k" if n >= 1000 else str(n)


def _format_recall_detail(recall_hits: list) -> str:
    # Format mirrors the live recall row in chat.svelte.ts — keep in sync.
    lines = []
    for h in recall_hits:
        if not isinstance(h, dict):
            lines.append(str(h))
            continue
        score = h.get("score")
        recency = h.get("recency")
        frequency = h.get("frequency")
        relevance = h.get("relevance")
        text = str(h.get("text", ""))[:80]
        score_part = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
        rec_part = f" rec {recency:.2f}" if isinstance(recency, (int, float)) else ""
        freq_part = f" freq {frequency:.2f}" if isinstance(frequency, (int, float)) else ""
        rel_part = f" rel {relevance:.2f}" if isinstance(relevance, (int, float)) else ""
        lines.append(f"{score_part}{rec_part}{freq_part}{rel_part} · {text}")
    return "\n".join(lines)


class _AnswerGate:
    """Withholds the head of an answer until it is clear it is prose, not tool JSON.

    A model may answer in prose or emit a tool call as bare JSON, and the first
    token does not say which. So the head is buffered until one character
    decides: ``{`` or ``[`` means tool JSON and the answer is suppressed (the
    loop parses it as a tool call instead), anything else means prose and the
    buffer is released. A leading ``` fence is stripped before deciding, since a
    model that wraps its JSON in a code block is still calling a tool.

    This lived twice — once in the streaming loop, once in the end-of-stream
    flush — and the copies had already drifted on one branch. They are not
    identical by nature, though: mid-stream an unterminated fence means "wait
    for more deltas", and at end-of-stream there are no more, so the text has to
    be released rather than dropped. That is the ``final`` flag, which makes the
    difference deliberate instead of accidental.
    """

    def __init__(self) -> None:
        self.emit = False
        self.suppress = False
        self._head = ""

    def feed(self, delta: str, *, final: bool = False) -> str:
        """Return the text to emit for this delta (empty string for none)."""
        if not delta:
            return ""
        if self.emit:
            return delta
        if self.suppress:
            return ""

        self._head += delta
        candidate = self._head.lstrip()

        if candidate.startswith("```"):
            newline_pos = candidate.find("\n")
            if newline_pos != -1:
                candidate = candidate[newline_pos + 1:].lstrip()
            elif not final:
                # Fence not terminated yet; more deltas are coming.
                return ""
            # At end-of-stream an unterminated fence is all we will ever get,
            # so fall through and let it be judged as-is rather than dropped.

        if not candidate:
            return ""
        if candidate[0] in ("{", "["):
            self.suppress = True
            return ""

        self.emit = True
        out, self._head = self._head, ""
        return out

    def reset(self) -> None:
        self.emit = False
        self.suppress = False
        self._head = ""


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
        model_available: Callable[[str, str], bool] | None = None,
        max_iterations: int = 6,
    ) -> None:
        self._registry = registry if registry is not None else get_registry()
        self._role = role
        self._model_available = model_available or _local_model_ready
        self._degrade_note = ""
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
        # Provider-native tool specs (S5.2). Empty unless the default wiring
        # builds them; the prose-JSON path stays the fallback (invariant 6).
        self._tool_specs: list[Any] = []
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
                degrade_quietly("local model availability check")

            from .wiring import make_recall_fn, make_tool_specs, make_tools_block

            recall_fn, self._recall_state = make_recall_fn()
            self._tool_specs = make_tool_specs()
            from datetime import datetime

            _assembler = ContextAssembler(
                character_block=character_block,
                recall=recall_fn,
                tools_block=make_tools_block(),
                reason=False,  # gated per turn by difficulty (S4.3)
                # Read-time local clock so June knows the actual wall-clock time
                # each turn (D.1). No timer/process — evaluated only at assembly.
                clock=datetime.now,
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
        # Tool calls the guard blocked pending user approval (ADR 0021, S6.2).
        self._blocked_names: list[str] = []
        # Structured records for those blocks, so the loop can surface a
        # first-class approval event rather than leaking the block as trace text.
        self._blocked_details: list[dict[str, Any]] = []
        if dispatch is not None:
            self._dispatch: Callable[[list[ToolCall], SessionState], Awaitable[list[Message]]] | None = dispatch
        else:
            from .wiring import make_dispatch_fn

            self._dispatched_names = []
            self._dispatch = make_dispatch_fn(
                self._dispatched_names, self._blocked_names, self._blocked_details
            )

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
        """True when networked tools are blocked — the loop offers to switch
        instead of silently egressing.

        Delegates to ``june_brain.privacy``, which owns the predicate and fails
        closed when the dial cannot be read.
        """
        from june_brain.privacy import local_only

        return local_only()

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
        self._blocked_names.clear()
        self._blocked_details.clear()
        self._recall_state["memories_recalled"] = 0
        self._recall_state["recall_hits"] = []
        self._degrade_note = ""

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

        # Graceful degradation: if we routed to a LOCAL tier other than the
        # baseline role and that tier's model isn't pulled, fall back to the
        # baseline provider so the turn still completes (invariant: graceful
        # degradation ships with model-judgment features).
        if (
            chosen_role != self._role
            and str(getattr(provider, "tier", "")).startswith("local")
            and not self._model_available(
                provider.model_id, getattr(provider, "base_url", "")
            )
        ):
            try:
                fallback = self._registry.get(self._role)
            except Exception:
                fallback = None
            if fallback is not None:
                self._degrade_note = (
                    f" Deep tier {provider.model_id} is not pulled; handled on "
                    f"{fallback.model_id} ({self._role}) — pull {provider.model_id} "
                    f"for deeper reasoning."
                )
                provider = fallback
                chosen_role = self._role

        return provider, chosen_role, difficulty

    def _resolve_tool_calls(self, result: Any) -> list[ToolCall]:
        """Prefer provider-native tool calls; fall back to prose-JSON extraction.

        When the provider returned structured ``tool_calls`` we use them directly
        (the reliability win of S5); otherwise the prompt-advertised prose-JSON
        path parses the model text (invariant 6 — graceful degradation).
        """
        native = getattr(result, "tool_calls", None) or []
        if native:
            return [ToolCall(name=c.name, args=dict(c.arguments)) for c in native]
        return self._extract_tool_calls(result)

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
        compacted: bool = False,
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

        degrade_note = getattr(self, "_degrade_note", "")
        if degrade_note:
            rationale = f"{rationale}{degrade_note}"

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
            input_tokens=tokens.input_tokens,
            output_tokens=tokens.output_tokens,
            compacted=compacted,
        )

    # ------------------------------------------------------------------
    # run_turn — the non-streaming convenience wrapper.
    #
    # ADR 0018 says there is one engine. Before D.4c this method WAS a second
    # one: it duplicated routing, dispatch, compaction and token accounting, and
    # the two copies had drifted — run_turn resolved provider-native tool calls
    # while stream_turn dropped them. Every production caller uses stream_turn,
    # so the drift lived in the path nobody ran, and the reliability harness
    # measured that path.
    #
    # Now it drains stream_turn and reassembles the result. Non-streaming
    # callers keep their ergonomics; there is exactly one implementation of a
    # turn, so nothing can drift again.
    # ------------------------------------------------------------------

    async def run_turn(self, session: SessionState, user_msg: Message) -> TurnResult:
        answer_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        provenance: TurnProvenance | None = None
        compacted = False

        async for ev in self.stream_turn(session, user_msg):
            if ev.type == "prompt":
                # A prompt event opens each iteration. Text from an earlier
                # iteration was a preamble to a tool call, not the reply: a
                # streaming client shows it as it arrives, but a non-streaming
                # caller wants the answer the turn settled on. Keeping only the
                # final iteration's text matches what run_turn returned when it
                # was a separate engine (`last_result.text`), including when the
                # loop exhausts max_iterations still calling tools.
                answer_parts.clear()
            elif ev.type == "token":
                answer_parts.append(ev.content)
            elif ev.type == "tool_call":
                tool_calls.append(ToolCall(name=ev.tool_name, args=dict(ev.tool_args)))
            elif ev.type == "compaction":
                compacted = True
            elif ev.type == "provenance":
                provenance = ev.provenance

        if provenance is None:
            # stream_turn returns early without provenance only when no provider
            # could be resolved. Keep the shape valid so callers do not have to
            # special-case a failure they can already see in the empty answer.
            provenance = TurnProvenance(
                tiers_used=[], cloud_call=False, model_ids=[],
                rationale="no provider available",
            )

        return TurnResult(
            assistant_msg=Message(role="assistant", content="".join(answer_parts)),
            tool_calls=tool_calls,
            provenance=provenance,
            tokens=TokenAccounting(
                input_tokens=provenance.input_tokens,
                output_tokens=provenance.output_tokens,
            ),
            compacted=compacted or provenance.compacted,
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

        def _emit(ev: StreamEvent) -> StreamEvent:
            """Stamp the current turn's id onto every outbound event."""
            ev.turn_id = trace.turn_id
            return ev

        block_egress = self._egress_blocked()
        try:
            provider, chosen_role, difficulty = await self._route(user_msg)
        except Exception:
            # Graceful degradation: no provider found
            yield _emit(StreamEvent(type="token", content="error: no provider available"))
            trace.record("error", "no provider available")
            self._trace_store.write(trace)
            yield _emit(StreamEvent(type="done"))
            return

        self._apply_reason(difficulty.label)

        route_summary = f"route -> {chosen_role} ({provider.model_id})"
        route_detail = (
            f"tier: {provider.tier}\nmodel: {provider.model_id}\nrole: {chosen_role}\n"
            f"difficulty: {difficulty.label} ({difficulty.source})"
        )
        yield _emit(StreamEvent(type="iteration", content=route_summary, detail=route_detail))
        trace.record("iteration", route_summary, detail=route_detail)

        # Emit a model_call event only when the classifier actually called a model.
        # Cache hits and heuristic fallbacks made no LLM call, so emitting one
        # would be dishonest.
        if difficulty.source == "model" and difficulty.model_id is not None:
            in_part = _fmt_tok(difficulty.input_tokens)
            out_part = _fmt_tok(difficulty.output_tokens)
            tok_part = f" · {in_part}/{out_part} tok" if in_part and out_part else ""
            lat_part = (
                f" · {difficulty.latency_ms}ms"
                if difficulty.latency_ms is not None
                else ""
            )
            mc_label = (
                f'classifier · {difficulty.model_id} · "{difficulty.label}"'
                f"{tok_part}{lat_part}"
            )
            mc_detail = f"difficulty classification → {difficulty.label} (source: model)"
            yield _emit(StreamEvent(type="model_call", content=mc_label, detail=mc_detail))
            trace.record("model_call", mc_label, detail=mc_detail)

        tokens = TokenAccounting()
        all_tool_calls: list[ToolCall] = []
        first_iteration = True

        for iteration_idx in range(self._max_iterations):
            ctx = self._assemble_context(session, user_msg)

            # Emit the rendered prompt — the "LLM factory" input — as a trace
            # event. content stays empty; the full prompt rides in detail.
            rendered_prompt = self._render_prompt(ctx)
            yield _emit(StreamEvent(
                type="prompt",
                content=f"prompt assembled ({len(ctx)} messages)",
                detail=rendered_prompt,
                iteration=iteration_idx,
            ))
            trace.record(
                "prompt",
                f"prompt assembled ({len(ctx)} messages)",
                detail=rendered_prompt,
            )

            # Emit recall event once on the first iteration if we have hits
            if first_iteration:
                recall_hits = self._recall_state.get("recall_hits", [])
                if recall_hits:
                    yield _emit(StreamEvent(type="recall", recall_hits=list(recall_hits)))
                    trace.record(
                        "recall",
                        f"recall · {len(recall_hits)} memories",
                        detail=_format_recall_detail(recall_hits),
                    )
                first_iteration = False

            reasoning_observed = False

            # --- stream the provider response ---
            usage_reported = False
            answer_accum = ""
            reasoning_accum = ""
            gate = _AnswerGate()
            splitter = ReasoningSplitter()

            # Native tool calls arrive on their own channel of the delta and are
            # collected here, then preferred over prose-JSON extraction below
            # (D.4b). Before the seam was typed, they had nowhere to arrive and
            # the turn ended blank.
            native_tool_calls: list[ToolCall] = []

            try:
                async for raw_delta in provider.stream(
                    GenerateRequest(
                        messages=ctx,
                        max_tokens=_MAX_TOKENS,
                        tools=self._tool_specs or None,
                    )
                ):
                    # A provider may yield a bare str (text delta) or a typed
                    # StreamDelta. Normalise to the latter.
                    if isinstance(raw_delta, str):
                        text_part, reasoning_part = raw_delta, ""
                    else:
                        text_part = getattr(raw_delta, "text", "") or ""
                        reasoning_part = getattr(raw_delta, "reasoning", "") or ""
                        for call in getattr(raw_delta, "tool_calls", None) or []:
                            native_tool_calls.append(
                                ToolCall(name=call.name, args=dict(call.arguments))
                            )

                    # Reasoning arrives on its own channel now; feed it through
                    # the splitter as a tagged block so downstream handling and
                    # the trace stay identical to the inline <think> form.
                    if reasoning_part:
                        for seg_kind, seg_text in splitter.feed(
                            f"<think>{reasoning_part}</think>"
                        ):
                            if seg_kind == "reasoning" and seg_text:
                                reasoning_observed = True
                                reasoning_accum += seg_text
                                yield _emit(
                                    StreamEvent(type="reasoning", content=seg_text)
                                )
                    if not text_part:
                        continue

                    # Feed delta through the reasoning splitter first. Reasoning
                    # is observed for activity traces, but never streamed raw.
                    segments = splitter.feed(text_part)
                    for seg_kind, seg_text in segments:
                        if seg_kind == "reasoning":
                            if seg_text:
                                reasoning_observed = True
                                reasoning_accum += seg_text
                                yield _emit(StreamEvent(type="reasoning", content=seg_text))
                        else:
                            # seg_kind == "answer": through the suppression gate.
                            answer_delta = seg_text
                            if not answer_delta:
                                continue
                            answer_accum += answer_delta
                            out = gate.feed(answer_delta)
                            if out:
                                yield _emit(StreamEvent(type="token", content=out))

                # Flush any residual reasoning/answer at end-of-stream.
                for seg_kind, seg_text in splitter.flush():
                    if seg_kind == "reasoning":
                        if seg_text:
                            reasoning_observed = True
                            reasoning_accum += seg_text
                            yield _emit(StreamEvent(type="reasoning", content=seg_text))
                    else:
                        answer_delta = seg_text
                        if not answer_delta:
                            continue
                        answer_accum += answer_delta
                        out = gate.feed(answer_delta, final=True)
                        if out:
                            yield _emit(StreamEvent(type="token", content=out))

            except Exception:
                # Stream failed — fall back to a single generate call
                try:
                    result = await provider.generate(
                        GenerateRequest(
                            messages=ctx,
                            max_tokens=_MAX_TOKENS,
                            tools=self._tool_specs or None,
                        )
                    )
                    # generate() carries tool_calls on the result; collect them
                    # so a stream failure does not also lose the tool (D.4b).
                    for call in getattr(result, "tool_calls", None) or []:
                        native_tool_calls.append(
                            ToolCall(name=call.name, args=dict(call.arguments))
                        )
                    # The provider reported real usage; prefer it over the
                    # estimate below rather than adding both (which is what this
                    # path used to do, roughly doubling every fallback turn's
                    # count in the Glass Box frame).
                    tokens.input_tokens += result.input_tokens
                    tokens.output_tokens += result.output_tokens
                    usage_reported = True
                    gate.emit = True
                    gate.suppress = False
                    fallback_reasoning, fallback_answer = split_reasoning(result.text)
                    reasoning_observed = reasoning_observed or bool(fallback_reasoning)
                    answer_accum = fallback_answer
                    yield _emit(StreamEvent(type="token", content=fallback_answer))
                    if fallback_reasoning:
                        reasoning_accum += fallback_reasoning
                        yield _emit(StreamEvent(type="reasoning", content=fallback_reasoning))
                except Exception:
                    answer_accum = ""

            # Estimate the streamed response's tokens. A streamed reply does not
            # reliably carry a usage block, so estimation is the normal path —
            # but when the generate() fallback ran it already contributed the
            # provider's real numbers, and adding these on top would count the
            # same turn twice.
            est_in = estimate_tokens(" ".join(m.content for m in ctx))
            est_out = estimate_tokens(answer_accum)
            if not usage_reported:
                tokens.input_tokens += est_in
                tokens.output_tokens += est_out

            # Build a GenerateResult-like object for tool-call extraction
            pseudo_result = GenerateResult(
                text=answer_accum,
                input_tokens=est_in,
                output_tokens=est_out,
                latency_ms=0,
                model_id=provider.model_id,
                tier=provider.tier,
            )

            # Native calls win when the provider produced them; the prose-JSON
            # extractor stays as the fallback (invariant 6), which is what a
            # model without function calling still uses.
            tool_calls = native_tool_calls or self._extract_tool_calls(pseudo_result)

            # Record + stream this iteration's internals: cleaned model output
            # (including suppressed tool JSON, but excluding raw reasoning) and
            # how many tool calls were parsed out of it.
            iter_summary = f"iteration {iteration_idx} · {len(tool_calls)} tool call(s)"
            iter_detail = answer_accum or "(no output)"
            yield _emit(StreamEvent(
                type="iteration",
                content=iter_summary,
                detail=iter_detail,
                iteration=iteration_idx,
            ))
            trace.record("iteration", iter_summary, detail=iter_detail)
            if reasoning_observed:
                trace.record(
                    "reasoning",
                    "model reasoning",
                    detail=reasoning_accum,
                )

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
                        yield _emit(StreamEvent(
                            type="tool_blocked",
                            tool_name=tc.name,
                            tool_args=tc.args,
                            detail=args_detail,
                            network=True,
                        ))
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
                    yield _emit(StreamEvent(
                        type="tool_call",
                        tool_name=tc.name,
                        tool_args=tc.args,
                        detail=args_detail,
                        network=networked,
                    ))
                    egress_note = " [egress: leaves your machine]" if networked else ""
                    trace.record(
                        "tool_call", f"tool · {tc.name}{egress_note}", detail=args_detail
                    )
                    runnable.append(tc)

                if runnable:
                    blocked_before = len(self._blocked_details)
                    run_obs = await self._dispatch(runnable, session)
                    # Calls the guard withheld this batch, keyed by their position
                    # in `runnable` so we can pair each back to its observation.
                    blocked_by_index = {
                        d["index"]: d
                        for d in self._blocked_details[blocked_before:]
                    }
                    for i, obs_msg in enumerate(run_obs):
                        tc_name = runnable[i].name if i < len(runnable) else ""
                        blk = blocked_by_index.get(i)
                        if blk is not None:
                            # The tool never ran — surface a first-class approval
                            # request instead of a fake result. The model still
                            # gets the [ACTION BLOCKED] observation so it relays
                            # the ask to the user in prose.
                            reason = str(blk.get("reason") or "")
                            yield _emit(StreamEvent(
                                type="tool_blocked",
                                tool_name=tc_name,
                                tool_args=runnable[i].args if i < len(runnable) else {},
                                detail=reason,
                                needs_approval=True,
                                action_class=str(blk.get("action_class") or ""),
                            ))
                            trace.record(
                                "tool_blocked",
                                f"needs approval · {tc_name}",
                                detail=reason,
                            )
                            continue
                        yield _emit(StreamEvent(
                            type="tool_result",
                            tool_name=tc_name,
                            tool_result=obs_msg.content,
                            detail=obs_msg.content,
                        ))
                        trace.record(
                            "tool_result", f"result · {tc_name}", detail=obs_msg.content
                        )
                    observations.extend(run_obs)

                tool_turn = Message(role="assistant", content=answer_accum)
                session.messages.append(tool_turn)
                session.messages.extend(observations)
                # Reset classification state for next iteration
                gate.reset()
            else:
                # No tool calls — if we suppressed (looked like JSON but wasn't a real
                # tool call), emit the accumulated text now so the user sees something
                if gate.suppress and answer_accum:
                    yield _emit(StreamEvent(type="token", content=answer_accum))
                break

        _compact_outcome = await self._maybe_compact(session)
        compacted = bool(_compact_outcome)
        if compacted:
            yield _emit(StreamEvent(
                type="compaction",
                content="conversation compacted",
                detail="The conversation history was compacted into the pinned-state anchor.",
            ))
            trace.record(
                "compaction",
                "conversation compacted",
                detail="History compacted into the pinned-state anchor.",
            )
            # Emit a model_call event only when the compactor actually called a
            # model. Truncation/fallback compaction made no LLM call, so emitting
            # one would be dishonest (mirrors GB-4a classifier visibility).
            mc_model_id = getattr(_compact_outcome, "model_id", None)
            if mc_model_id is not None:
                in_part = _fmt_tok(getattr(_compact_outcome, "input_tokens", None))
                out_part = _fmt_tok(getattr(_compact_outcome, "output_tokens", None))
                tok_part = f" · {in_part}/{out_part} tok" if in_part and out_part else ""
                raw_latency = getattr(_compact_outcome, "latency_ms", None)
                lat_part = f" · {raw_latency}ms" if raw_latency is not None else ""
                mc_label = f"compactor · {mc_model_id}{tok_part}{lat_part}"
                mc_detail = "compacted conversation → pinned-state anchor"
                yield _emit(StreamEvent(type="model_call", content=mc_label, detail=mc_detail))
                trace.record("model_call", mc_label, detail=mc_detail)

        provenance = self._build_provenance(
            provider, chosen_role, all_tool_calls, _start, tokens, difficulty,
            compacted=compacted,
        )
        yield _emit(StreamEvent(type="provenance", provenance=provenance))
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
        yield _emit(StreamEvent(type="done"))

