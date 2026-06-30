"""Tests for the per-turn harness trace (loop/trace.py) and its capture in
HandwrittenLoop.stream_turn.

All synchronous (asyncio.run). No Ollama or network required.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import june_brain.config
import pytest
from june_brain.datadir import layout
from june_brain.loop.handwritten import HandwrittenLoop, _format_recall_detail
from june_brain.loop.interface import SessionState, StreamEvent
from june_brain.loop.trace import TraceStore, TurnTrace
from june_brain.providers.base import (
    GenerateRequest,
    GenerateResult,
    Message,
    ProviderHealth,
)
from june_brain.providers.registry import ProviderRegistry


def _point_datadir(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(june_brain.config, "MEMORY_DIR", str(path))


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


class _Provider:
    def __init__(self, chunks: list[str], model_id: str = "mock", tier: str = "local-fast") -> None:
        self.model_id = model_id
        self.tier = tier
        self._chunks = chunks

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        return GenerateResult(
            text="".join(self._chunks),
            input_tokens=10,
            output_tokens=5,
            latency_ms=1,
            model_id=self.model_id,
            tier=self.tier,
        )

    async def stream(self, req: GenerateRequest) -> AsyncIterator[str]:
        for c in self._chunks:
            yield c

    async def health(self) -> ProviderHealth:
        return ProviderHealth(reachable=True, loaded=True)


# ---------------------------------------------------------------------------
# TraceStore round-trip
# ---------------------------------------------------------------------------


def test_trace_store_write_read_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _point_datadir(monkeypatch, tmp_path)

    trace = TurnTrace(turn_id="abc123", user_id="u1")
    trace.record("prompt", "prompt assembled (2 messages)", detail="[system]\nhi\n\n[user]\nyo")
    trace.record("iteration", "iteration 0 · 0 tool call(s)", detail="some output")
    trace.record("done", "done")

    store = TraceStore()
    assert store.write(trace) is True
    assert (layout.traces_dir() / "abc123.json").exists()

    loaded = store.read("abc123")
    assert loaded is not None
    assert loaded.turn_id == "abc123"
    assert loaded.user_id == "u1"
    assert [e.kind for e in loaded.events] == ["prompt", "iteration", "done"]
    assert loaded.events[0].detail.startswith("[system]")

    recent = store.list_recent()
    assert any(r["turn_id"] == "abc123" and r["event_count"] == 3 for r in recent)


def test_trace_store_read_missing_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_datadir(monkeypatch, tmp_path)
    assert TraceStore().read("nope") is None


def test_trace_store_caps_retention(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _point_datadir(monkeypatch, tmp_path)
    import june_brain.loop.trace as trace_mod

    monkeypatch.setattr(trace_mod, "TRACE_MAX", 2)
    store = TraceStore()
    for i in range(5):
        t = TurnTrace(turn_id=f"t{i}", user_id="u")
        t.record("done", "done")
        store.write(t)
    # Only the 2 most recent survive the cap.
    assert len(list((layout.traces_dir()).glob("*.json"))) == 2


def test_trace_store_clear(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _point_datadir(monkeypatch, tmp_path)
    store = TraceStore()
    for i in range(3):
        t = TurnTrace(turn_id=f"c{i}", user_id="u")
        store.write(t)
    removed = store.clear()
    assert removed == 3
    assert store.list_recent() == []


def test_trace_capture_disabled_when_max_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_datadir(monkeypatch, tmp_path)
    import june_brain.loop.trace as trace_mod

    monkeypatch.setattr(trace_mod, "TRACE_MAX", 0)
    assert TraceStore().write(TurnTrace(turn_id="x", user_id="u")) is False
    assert TraceStore().list_recent() == []


# ---------------------------------------------------------------------------
# stream_turn captures prompt + iteration trace events and persists a file
# ---------------------------------------------------------------------------


def test_stream_turn_emits_prompt_and_iteration_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_datadir(monkeypatch, tmp_path)

    provider = _Provider(chunks=["Hello ", "world"])
    reg = _registry_with("local-fast", provider)
    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [Message(role="system", content="ctx"), m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
    )

    session = SessionState(user_id="u9", messages=[])
    events = _collect_stream(loop.stream_turn(session, Message(role="user", content="hi")))

    kinds = [e.type for e in events]
    assert "prompt" in kinds
    # The prompt event carries the rendered context in detail, not content.
    prompt_ev = next(e for e in events if e.type == "prompt")
    # Full prompt rides in detail; content is just the collapsed-line summary.
    assert "[system]" in prompt_ev.detail
    assert "[user]" in prompt_ev.detail
    assert "prompt assembled" in prompt_ev.content
    # No token event ever carries the raw prompt body.
    assert all(prompt_ev.detail not in e.content for e in events if e.type == "token")

    # done last, provenance before it
    assert kinds[-1] == "done"
    assert kinds[-2] == "provenance"


def test_stream_turn_persists_trace_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_datadir(monkeypatch, tmp_path)

    provider = _Provider(chunks=["ok"])
    reg = _registry_with("local-fast", provider)
    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
    )

    session = SessionState(user_id="persist-user", messages=[])
    _collect_stream(loop.stream_turn(session, Message(role="user", content="hi")))

    recent = TraceStore().list_recent()
    assert len(recent) == 1
    loaded = TraceStore().read(recent[0]["turn_id"])
    assert loaded is not None
    assert loaded.user_id == "persist-user"
    kinds = [e.kind for e in loaded.events]
    assert "prompt" in kinds
    assert "iteration" in kinds
    assert kinds[-1] == "done"


def test_stream_turn_trace_records_raw_reasoning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_datadir(monkeypatch, tmp_path)

    provider = _Provider(chunks=["<think>private chain</think>", "final"])
    reg = _registry_with("local-fast", provider)
    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
    )

    session = SessionState(user_id="trace-reasoning-user", messages=[])
    _collect_stream(loop.stream_turn(session, Message(role="user", content="hi")))

    recent = TraceStore().list_recent()
    assert len(recent) == 1
    loaded = TraceStore().read(recent[0]["turn_id"])
    assert loaded is not None

    # The trace now records reasoning with the actual content (not suppressed).
    assert any(e.kind == "reasoning" and "private chain" in (e.detail or "") for e in loaded.events)
    # Tags are stripped by the splitter — the raw <think> tag must not appear in the trace.
    reasoning_detail = next(
        (e.detail or "" for e in loaded.events if e.kind == "reasoning"), ""
    )
    assert "<think>" not in reasoning_detail
    # Reasoning events must NOT say "suppressed".
    assert not any(e.kind == "reasoning" and "suppressed" in (e.summary or "") for e in loaded.events)
    # The answer still appears in the iteration detail.
    assert any(e.kind == "iteration" and e.detail == "final" for e in loaded.events)


# ---------------------------------------------------------------------------
# _format_recall_detail unit tests
# ---------------------------------------------------------------------------


def test_format_recall_detail_vector_hit_all_components() -> None:
    """A vector hit with all salience components formats score + rec/freq/rel."""
    hits = [{"score": 0.85, "recency": 0.5, "frequency": 0.3, "relevance": 0.2, "text": "I like coffee"}]
    line = _format_recall_detail(hits)
    assert "0.85" in line
    assert "rec " in line
    assert "freq " in line
    assert "rel " in line
    assert "I like coffee" in line


def test_format_recall_detail_score_only_omits_components() -> None:
    """A hit with only score and text produces no rec/freq/rel group."""
    hits = [{"score": 0.72, "text": "user prefers dark mode"}]
    line = _format_recall_detail(hits)
    assert "0.72" in line
    assert "rec " not in line
    assert "freq " not in line
    assert "rel " not in line
    assert "user prefers dark mode" in line


def test_format_recall_detail_text_truncated_to_80_chars() -> None:
    """Text longer than 80 characters is truncated."""
    long_text = "a" * 100
    hits = [{"score": 0.5, "text": long_text}]
    line = _format_recall_detail(hits)
    # The text portion after ' · ' must be at most 80 chars.
    after_sep = line.split(" · ", 1)[-1]
    assert len(after_sep) == 80
