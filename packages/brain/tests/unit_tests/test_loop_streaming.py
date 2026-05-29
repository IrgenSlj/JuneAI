"""Tests for HandwrittenLoop.stream_turn and LangGraphLoop.stream_turn (Slice 2b).

All tests are synchronous (asyncio.run). No Ollama or network required.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from june_brain.loop.handwritten import HandwrittenLoop
from june_brain.loop.interface import HarnessLoop, SessionState, StreamEvent, ToolCall
from june_brain.loop.langgraph_loop import LangGraphLoop
from june_brain.providers.base import (
    GenerateRequest,
    GenerateResult,
    Message,
    ProviderHealth,
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

    async def generate(self, req: GenerateRequest) -> GenerateResult:
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
# Test 5: LangGraphLoop satisfies HarnessLoop and stream_turn yields token+done
# ---------------------------------------------------------------------------


def test_langgraph_satisfies_harness_loop_with_stream_turn() -> None:
    """LangGraphLoop still satisfies isinstance(loop, HarnessLoop)."""
    fake_agent = SimpleNamespace(
        invoke=lambda state: {"messages": [SimpleNamespace(content="lg reply")]}
    )
    loop = LangGraphLoop(agent=fake_agent)
    assert isinstance(loop, HarnessLoop)


def test_langgraph_stream_turn_yields_token_then_done() -> None:
    """LangGraphLoop.stream_turn yields a token event then done."""
    fake_agent = SimpleNamespace(
        invoke=lambda state: {"messages": [SimpleNamespace(content="langgraph answer")]}
    )
    loop = LangGraphLoop(agent=fake_agent)

    session = SessionState(user_id="lg-user", messages=[])
    events = _collect_stream(
        loop.stream_turn(session, Message(role="user", content="hi"))
    )

    event_types = [e.type for e in events]
    assert "token" in event_types
    assert event_types[-1] == "done"

    token_events = [e for e in events if e.type == "token"]
    assert token_events[0].content == "langgraph answer"


