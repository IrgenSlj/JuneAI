"""Tests for june_brain.loop.reasoning — splitter helper and stream_turn integration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from june_brain.loop.handwritten import HandwrittenLoop
from june_brain.loop.interface import SessionState, StreamEvent
from june_brain.loop.reasoning import ReasoningSplitter, split_reasoning
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


def _feed_all(chunks: list[str]) -> list[tuple[str, str]]:
    """Feed all chunks through a fresh ReasoningSplitter and flush."""
    s = ReasoningSplitter()
    out: list[tuple[str, str]] = []
    for chunk in chunks:
        out.extend(s.feed(chunk))
    out.extend(s.flush())
    return out


def _text_for(kind: str, segs: list[tuple[str, str]]) -> str:
    return "".join(t for k, t in segs if k == kind)


def _registry_with(role: str, provider: Any) -> ProviderRegistry:
    registry = ProviderRegistry(toml_data={"roles": {}, "providers": {}})
    registry.register(role, provider)
    return registry


async def _noop_compact(session: SessionState) -> bool:
    return False


def _collect_stream(gen: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    async def _drain() -> list[StreamEvent]:
        events: list[StreamEvent] = []
        async for ev in gen:
            events.append(ev)
        return events
    return asyncio.run(_drain())


class MultiChunkProvider:
    def __init__(self, chunks: list[str], model_id: str = "mock", tier: str = "local-fast") -> None:
        self.model_id = model_id
        self.tier = tier
        self._chunks = chunks

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        text = "".join(self._chunks)
        return GenerateResult(
            text=text, input_tokens=10, output_tokens=5, latency_ms=1,
            model_id=self.model_id, tier=self.tier,
        )

    async def stream(self, req: GenerateRequest) -> AsyncIterator[str]:
        for chunk in self._chunks:
            yield chunk

    async def health(self) -> ProviderHealth:
        return ProviderHealth(reachable=True, loaded=True)


# ---------------------------------------------------------------------------
# split_reasoning — pure helper
# ---------------------------------------------------------------------------


def test_split_reasoning_no_tags() -> None:
    """Text with no think-tags: all answer, no reasoning."""
    reasoning, answer = split_reasoning("Hello world")
    assert reasoning == ""
    assert answer == "Hello world"


def test_split_reasoning_full_think_block() -> None:
    """Full <think>...</think> block followed by answer text."""
    text = "<think>I should consider options A and B.</think>The answer is A."
    reasoning, answer = split_reasoning(text)
    assert "I should consider options A and B." in reasoning
    assert "The answer is A." in answer
    assert "<think>" not in reasoning
    assert "<think>" not in answer


def test_split_reasoning_thinking_variant() -> None:
    """<thinking>...</thinking> is also recognized."""
    text = "<thinking>step by step</thinking>Done."
    reasoning, answer = split_reasoning(text)
    assert "step by step" in reasoning
    assert "Done." in answer


def test_split_reasoning_case_insensitive() -> None:
    """Tags are case-insensitive."""
    text = "<THINK>upper case</THINK>result"
    reasoning, answer = split_reasoning(text)
    assert "upper case" in reasoning
    assert "result" in answer


def test_split_reasoning_no_crash_on_malformed() -> None:
    """Malformed / unclosed tags must not raise."""
    text = "<think>unclosed block and more text"
    reasoning, answer = split_reasoning(text)
    # Just must not raise; accept either interpretation.
    assert isinstance(reasoning, str)
    assert isinstance(answer, str)


# ---------------------------------------------------------------------------
# ReasoningSplitter — streaming interface
# ---------------------------------------------------------------------------


def test_splitter_no_tags_all_answer() -> None:
    """No think-tags: all segments are answer."""
    segs = _feed_all(["Hello ", "world"])
    assert _text_for("reasoning", segs) == ""
    assert _text_for("answer", segs) == "Hello world"


def test_splitter_complete_think_block_then_answer() -> None:
    """Full think block in one chunk, then answer."""
    segs = _feed_all(["<think>some reasoning</think>The answer"])
    assert "some reasoning" in _text_for("reasoning", segs)
    assert "The answer" in _text_for("answer", segs)
    assert "<think>" not in _text_for("reasoning", segs)
    assert "<think>" not in _text_for("answer", segs)


def test_splitter_tags_split_across_chunks() -> None:
    """Tag boundaries crossing chunk boundaries are handled correctly."""
    segs = _feed_all(["<thi", "nk>hi</thi", "nk>done"])
    assert "hi" in _text_for("reasoning", segs)
    assert "done" in _text_for("answer", segs)
    assert "<thi" not in _text_for("reasoning", segs)
    assert "<thi" not in _text_for("answer", segs)


def test_splitter_thinking_variant() -> None:
    """<thinking>...</thinking> recognized in streaming mode."""
    segs = _feed_all(["<thinking>chain</thinking>answer"])
    assert "chain" in _text_for("reasoning", segs)
    assert "answer" in _text_for("answer", segs)


def test_splitter_no_crash_on_garbage() -> None:
    """Arbitrary garbage input must not raise."""
    garbage = ["<<<", ">>>", "<think", "dangling", "no close"]
    try:
        segs = _feed_all(garbage)
        assert isinstance(segs, list)
    except Exception as exc:
        pytest.fail(f"ReasoningSplitter raised on garbage input: {exc}")


def test_splitter_multiple_think_blocks() -> None:
    """Multiple think blocks each produce reasoning; interleaved answer."""
    text = "<think>r1</think>a1<think>r2</think>a2"
    segs = _feed_all([text])
    assert "r1" in _text_for("reasoning", segs)
    assert "r2" in _text_for("reasoning", segs)
    assert "a1" in _text_for("answer", segs)
    assert "a2" in _text_for("answer", segs)


# ---------------------------------------------------------------------------
# stream_turn integration — reasoning events appear, token events clean
# ---------------------------------------------------------------------------


def test_stream_turn_emits_reasoning_event() -> None:
    """stream_turn emits reasoning events for <think> content."""
    chunks = ["<think>my chain of thought</think>", "The final answer."]
    provider = MultiChunkProvider(chunks=chunks)
    reg = _registry_with("local-fast", provider)

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
    )

    session = SessionState(user_id="u-reasoning", messages=[])
    events = _collect_stream(loop.stream_turn(session, Message(role="user", content="think")))

    reasoning_events = [e for e in events if e.type == "reasoning"]
    token_events = [e for e in events if e.type == "token"]

    # reasoning content was captured
    reasoning_text = "".join(e.content for e in reasoning_events)
    assert "my chain of thought" in reasoning_text

    # think-tags do NOT appear in token events
    token_text = "".join(e.content for e in token_events)
    assert "<think>" not in token_text
    assert "</think>" not in token_text
    assert "my chain of thought" not in token_text

    # The answer does appear
    assert "The final answer." in token_text


def test_stream_turn_no_think_tags_unchanged() -> None:
    """Models with no think-tags: no reasoning events, tokens unchanged."""
    chunks = ["Hel", "lo ", "world"]
    provider = MultiChunkProvider(chunks=chunks)
    reg = _registry_with("local-fast", provider)

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
    )

    session = SessionState(user_id="u-nothink", messages=[])
    events = _collect_stream(loop.stream_turn(session, Message(role="user", content="hi")))

    reasoning_events = [e for e in events if e.type == "reasoning"]
    token_events = [e for e in events if e.type == "token"]

    assert reasoning_events == [], "No reasoning events expected for plain-text model"
    full_text = "".join(e.content for e in token_events)
    assert full_text == "Hello world"
