"""Tests for HandwrittenLoop.stream_turn (Slice 2b).

All tests are synchronous (asyncio.run). No Ollama or network required.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from june_brain.loop.handwritten import HandwrittenLoop
from june_brain.loop.interface import SessionState, StreamEvent, ToolCall
from june_brain.providers.base import (
    GenerateRequest,
    GenerateResult,
    Message,
    ProviderHealth,
    ToolSpec,
)
from june_brain.providers.registry import ProviderRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registry_with(role: str, provider: Any) -> ProviderRegistry:
    registry = ProviderRegistry(toml_data={"roles": {}, "providers": {}})
    registry.register(role, provider)
    return registry


async def _noop_compact(session: SessionState) -> bool:
    return False


def _collect_stream(gen: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    """Drive an async generator to completion and return all events."""

    async def _drain() -> list[StreamEvent]:
        events: list[StreamEvent] = []
        async for ev in gen:
            events.append(ev)
        return events

    return asyncio.run(_drain())


# ---------------------------------------------------------------------------
# Multi-chunk streaming provider
# ---------------------------------------------------------------------------


class MultiChunkProvider:
    """Streams multiple text deltas; single generate fallback."""

    def __init__(
        self,
        chunks: list[str],
        model_id: str = "mock-model",
        tier: str = "local-fast",
    ) -> None:
        self.model_id = model_id
        self.tier = tier
        self._chunks = chunks
        self.generate_requests: list[GenerateRequest] = []
        self.stream_requests: list[GenerateRequest] = []

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        self.generate_requests.append(req)
        text = "".join(self._chunks)
        return GenerateResult(
            text=text,
            input_tokens=10,
            output_tokens=5,
            latency_ms=1,
            model_id=self.model_id,
            tier=self.tier,
        )

    async def stream(self, req: GenerateRequest) -> AsyncIterator[str]:
        self.stream_requests.append(req)
        for chunk in self._chunks:
            yield chunk

    async def health(self) -> ProviderHealth:
        return ProviderHealth(reachable=True, loaded=True)


# ---------------------------------------------------------------------------
# Test 1: multi-chunk streaming yields ordered token events
# ---------------------------------------------------------------------------


def test_stream_turn_yields_ordered_token_events() -> None:
    """Provider streaming ['Hel', 'lo ', 'world'] → token events reconstruct the text."""
    provider = MultiChunkProvider(chunks=["Hel", "lo ", "world"])
    reg = _registry_with("local-fast", provider)

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
    )

    session = SessionState(user_id="u1", messages=[])
    events = _collect_stream(loop.stream_turn(session, Message(role="user", content="hi")))

    token_events = [e for e in events if e.type == "token"]
    assert len(token_events) >= 1

    full_text = "".join(e.content for e in token_events)
    assert full_text == "Hello world"

    # done must be last
    assert events[-1].type == "done"

    # provenance must be second to last
    assert events[-2].type == "provenance"


def test_stream_turn_passes_tool_specs_to_provider_stream() -> None:
    """Streaming requests advertise the same native tool specs as run_turn."""
    provider = MultiChunkProvider(chunks=["Hello"])
    reg = _registry_with("local-fast", provider)
    spec = ToolSpec(
        name="get_weather",
        description="Look up weather.",
        parameters={"type": "object", "properties": {}},
    )

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
    )
    loop._tool_specs = [spec]

    session = SessionState(user_id="u-tools", messages=[])
    _collect_stream(loop.stream_turn(session, Message(role="user", content="hi")))

    assert provider.stream_requests
    assert provider.stream_requests[0].tools == [spec]


# ---------------------------------------------------------------------------
# Test 2: tool-call suppression — JSON stream suppressed, prose streams as tokens
# ---------------------------------------------------------------------------


def test_stream_turn_tool_call_suppression() -> None:
    """First stream is a JSON tool call (suppressed); second stream is prose (emitted)."""
    call_count = [0]
    tool_json = '{"tool_calls": [{"name": "x", "args": {}}]}'
    prose_chunks = ["The ", "answer ", "is 42."]

    class SequentialStreamProvider:
        model_id = "mock"
        tier = "local-fast"

        async def generate(self, req: GenerateRequest) -> GenerateResult:
            # Not used by stream_turn directly but needed for Protocol
            return GenerateResult(
                text="",
                input_tokens=1,
                output_tokens=1,
                latency_ms=1,
                model_id=self.model_id,
                tier=self.tier,
            )

        async def stream(self, req: GenerateRequest) -> AsyncIterator[str]:
            call_count[0] += 1
            if call_count[0] == 1:
                yield tool_json
            else:
                for chunk in prose_chunks:
                    yield chunk

        async def health(self) -> ProviderHealth:
            return ProviderHealth(reachable=True)

    # Extractor: parse the JSON tool call on first call, return [] on second
    extract_calls = [0]

    def extractor(result: GenerateResult) -> list[ToolCall]:
        extract_calls[0] += 1
        if extract_calls[0] == 1:
            return [ToolCall(name="x", args={})]
        return []

    async def dispatch(
        tool_calls: list[ToolCall], session: SessionState
    ) -> list[Message]:
        return [Message(role="tool", content="tool x result")]

    reg = _registry_with("local-fast", SequentialStreamProvider())
    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=extractor,
        dispatch=dispatch,
        maybe_compact=_noop_compact,
    )

    session = SessionState(user_id="u2", messages=[])
    events = _collect_stream(loop.stream_turn(session, Message(role="user", content="do x")))

    token_events = [e for e in events if e.type == "token"]
    tool_call_events = [e for e in events if e.type == "tool_call"]
    tool_result_events = [e for e in events if e.type == "tool_result"]

    # No token event should contain the raw JSON
    for ev in token_events:
        assert tool_json not in ev.content, f"Raw JSON leaked into token: {ev.content!r}"

    # A tool_call event for "x" must be present
    assert len(tool_call_events) == 1
    assert tool_call_events[0].tool_name == "x"

    # A tool_result event must follow
    assert len(tool_result_events) == 1
    assert tool_result_events[0].tool_result == "tool x result"

    # The prose from the second stream should appear as token events
    prose_text = "".join(e.content for e in token_events)
    assert "The " in prose_text or "answer" in prose_text or "42" in prose_text

    # done is last
    assert events[-1].type == "done"


# ---------------------------------------------------------------------------
# Test 3: recall event emitted before provenance
# ---------------------------------------------------------------------------


def test_stream_turn_recall_event_before_provenance() -> None:
    """With a recall function that returns hits, a recall event is yielded first."""
    from june_brain.context.assembler import ContextAssembler
    from june_brain.loop.wiring import make_recall_fn

    fake_hits = [{"content": "mem1"}, {"content": "mem2"}]
    fake_block = "## Recalled\n- mem1\n- mem2"

    def fake_recall_with_hits(user_id: str, query: str, k: int = 5):
        return fake_block, fake_hits

    recall_fn, recall_state = make_recall_fn(recall_with_hits_fn=fake_recall_with_hits)
    assembler = ContextAssembler(recall=recall_fn)

    provider = MultiChunkProvider(chunks=["Hello!"])
    reg = _registry_with("local-fast", provider)

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=assembler.assemble,
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
    )
    loop._recall_state = recall_state

    session = SessionState(user_id="recall-user", messages=[])
    events = _collect_stream(
        loop.stream_turn(session, Message(role="user", content="what do I like?"))
    )

    event_types = [e.type for e in events]

    # recall event must appear
    assert "recall" in event_types

    recall_idx = event_types.index("recall")
    provenance_idx = event_types.index("provenance")

    # recall must come before provenance
    assert recall_idx < provenance_idx

    # The recall event must carry the hits
    recall_event = events[recall_idx]
    assert len(recall_event.recall_hits) == 2


# ---------------------------------------------------------------------------
# Test 4: provenance event carries memories_recalled and skills_called
# ---------------------------------------------------------------------------


def test_stream_turn_provenance_fields() -> None:
    """provenance event has correct memories_recalled and cloud_call."""
    from june_brain.context.assembler import ContextAssembler
    from june_brain.loop.wiring import make_recall_fn

    fake_hits = [{"content": "m1"}]
    fake_block = "## Recalled\n- m1"

    def fake_recall_with_hits(user_id: str, query: str, k: int = 5):
        return fake_block, fake_hits

    recall_fn, recall_state = make_recall_fn(recall_with_hits_fn=fake_recall_with_hits)
    assembler = ContextAssembler(recall=recall_fn)

    provider = MultiChunkProvider(chunks=["ok"], tier="cloud-capable", model_id="gemini-flash")
    reg = _registry_with("cloud-capable", provider)

    loop = HandwrittenLoop(
        registry=reg,
        role="cloud-capable",
        assemble_context=assembler.assemble,
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
    )
    loop._recall_state = recall_state

    session = SessionState(user_id="prov-user", messages=[])
    events = _collect_stream(
        loop.stream_turn(session, Message(role="user", content="help"))
    )

    prov_events = [e for e in events if e.type == "provenance"]
    assert len(prov_events) == 1
    prov = prov_events[0].provenance
    assert prov is not None
    assert prov.memories_recalled == 1
    assert prov.cloud_call is True


# ---------------------------------------------------------------------------
# Test 5: provenance event carries token counts from the turn accumulator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test: model_call event emitted when classifier uses model, not when heuristic
# ---------------------------------------------------------------------------


def test_stream_turn_emits_model_call_when_classifier_uses_model() -> None:
    """A model_call event is emitted exactly once when the difficulty classifier called a model."""
    from june_brain.router.difficulty import DifficultyResult

    provider = MultiChunkProvider(chunks=["Hello!"])
    reg = _registry_with("local-fast", provider)

    async def _classify_model(text: str) -> DifficultyResult:
        return DifficultyResult(
            label="standard",
            source="model",
            model_id="mock-classifier",
            input_tokens=5,
            output_tokens=2,
            latency_ms=10,
        )

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
        classify=_classify_model,
    )

    session = SessionState(user_id="mc-user", messages=[])
    events = _collect_stream(
        loop.stream_turn(session, Message(role="user", content="hello"))
    )

    mc_events = [e for e in events if e.type == "model_call"]
    assert len(mc_events) == 1
    assert "classifier" in mc_events[0].content
    assert "mock-classifier" in mc_events[0].content
    assert "standard" in mc_events[0].content
    assert mc_events[0].detail == "difficulty classification → standard (source: model)"


def test_stream_turn_no_model_call_when_heuristic() -> None:
    """No model_call event is emitted when the classifier used the heuristic path."""
    from june_brain.router.difficulty import DifficultyResult

    provider = MultiChunkProvider(chunks=["Hello!"])
    reg = _registry_with("local-fast", provider)

    async def _classify_heuristic(text: str) -> DifficultyResult:
        return DifficultyResult(label="trivial", source="heuristic")

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
        classify=_classify_heuristic,
    )

    session = SessionState(user_id="heur-user", messages=[])
    events = _collect_stream(
        loop.stream_turn(session, Message(role="user", content="hi"))
    )

    mc_events = [e for e in events if e.type == "model_call"]
    assert len(mc_events) == 0


def test_stream_turn_no_model_call_when_cache() -> None:
    """No model_call event is emitted when the classifier returned a cache hit."""
    from june_brain.router.difficulty import DifficultyResult

    provider = MultiChunkProvider(chunks=["Hello!"])
    reg = _registry_with("local-fast", provider)

    async def _classify_cache(text: str) -> DifficultyResult:
        return DifficultyResult(label="standard", source="cache")

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
        classify=_classify_cache,
    )

    session = SessionState(user_id="cache-user", messages=[])
    events = _collect_stream(
        loop.stream_turn(session, Message(role="user", content="hello again"))
    )

    mc_events = [e for e in events if e.type == "model_call"]
    assert len(mc_events) == 0


def test_stream_turn_provenance_carries_token_counts() -> None:
    """provenance event carries input_tokens and output_tokens estimated from the turn."""
    provider = MultiChunkProvider(chunks=["Hello world"])
    reg = _registry_with("local-fast", provider)

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
    )

    session = SessionState(user_id="tok-user", messages=[])
    events = _collect_stream(loop.stream_turn(session, Message(role="user", content="hi")))

    prov_events = [e for e in events if e.type == "provenance"]
    assert len(prov_events) == 1
    prov = prov_events[0].provenance
    assert prov is not None
    assert hasattr(prov, "input_tokens")
    assert hasattr(prov, "output_tokens")
    assert hasattr(prov, "compacted")
    assert prov.input_tokens >= 0
    assert prov.output_tokens >= 0
    assert prov.compacted is False


# ---------------------------------------------------------------------------
# Compactor model_call visibility (GB-4b)
# ---------------------------------------------------------------------------


def test_stream_turn_emits_model_call_when_compactor_uses_model() -> None:
    """When compaction runs via the model path, a model_call event is emitted."""
    from june_brain.context.compactor import CompactionOutcome

    provider = MultiChunkProvider(chunks=["Hello!"])
    reg = _registry_with("local-fast", provider)

    async def _compact_model_path(session: SessionState) -> CompactionOutcome:
        return CompactionOutcome(
            compacted=True,
            model_id="mock-compactor",
            input_tokens=50,
            output_tokens=20,
            latency_ms=42,
        )

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_compact_model_path,
    )

    session = SessionState(user_id="cmp-user", messages=[])
    events = _collect_stream(
        loop.stream_turn(session, Message(role="user", content="compact me"))
    )

    compaction_events = [e for e in events if e.type == "compaction"]
    assert len(compaction_events) == 1

    mc_events = [e for e in events if e.type == "model_call"]
    compactor_mc = [e for e in mc_events if "compactor" in e.content]
    assert len(compactor_mc) == 1
    assert "mock-compactor" in compactor_mc[0].content
    assert "50" in compactor_mc[0].content or "20" in compactor_mc[0].content
    assert compactor_mc[0].detail == "compacted conversation → pinned-state anchor"


def test_stream_turn_no_model_call_when_compactor_fallback() -> None:
    """When compaction runs via the truncation fallback (no model call), no model_call event."""
    from june_brain.context.compactor import CompactionOutcome

    provider = MultiChunkProvider(chunks=["Hello!"])
    reg = _registry_with("local-fast", provider)

    async def _compact_fallback(session: SessionState) -> CompactionOutcome:
        # Fallback path: compacted=True but model_id is None (no LLM call)
        return CompactionOutcome(compacted=True)

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_compact_fallback,
    )

    session = SessionState(user_id="cmp-fallback-user", messages=[])
    events = _collect_stream(
        loop.stream_turn(session, Message(role="user", content="compact me"))
    )

    compaction_events = [e for e in events if e.type == "compaction"]
    assert len(compaction_events) == 1

    compactor_mc = [e for e in events if e.type == "model_call" and "compactor" in e.content]
    assert len(compactor_mc) == 0


def test_stream_turn_no_model_call_when_no_compaction() -> None:
    """When compaction does not run (below threshold), no compactor model_call event."""
    from june_brain.context.compactor import CompactionOutcome

    provider = MultiChunkProvider(chunks=["Hello!"])
    reg = _registry_with("local-fast", provider)

    async def _compact_noop(session: SessionState) -> CompactionOutcome:
        return CompactionOutcome(compacted=False)

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_compact_noop,
    )

    session = SessionState(user_id="cmp-noop-user", messages=[])
    events = _collect_stream(
        loop.stream_turn(session, Message(role="user", content="no compact"))
    )

    compactor_mc = [e for e in events if e.type == "model_call" and "compactor" in e.content]
    assert len(compactor_mc) == 0
