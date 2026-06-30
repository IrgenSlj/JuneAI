"""Test that every StreamEvent in a stream_turn carries the same turn_id,
and that the persisted trace file uses that same turn_id.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from june_brain.loop.handwritten import HandwrittenLoop
from june_brain.loop.interface import SessionState, StreamEvent
from june_brain.providers.base import GenerateRequest, GenerateResult, Message, ProviderHealth
from june_brain.providers.registry import ProviderRegistry


def _registry_with(role: str, provider: Any) -> ProviderRegistry:
    registry = ProviderRegistry(toml_data={"roles": {}, "providers": {}})
    registry.register(role, provider)
    return registry


async def _noop_compact(session: SessionState) -> bool:
    return False


def _collect(gen: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    async def _drain() -> list[StreamEvent]:
        return [ev async for ev in gen]

    return asyncio.run(_drain())


class _SimpleProvider:
    """Yields a single plain-text token so stream_turn runs to completion."""

    model_id = "mock"
    tier = "local-fast"

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        return GenerateResult(
            text="hello",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            model_id=self.model_id,
            tier=self.tier,
        )

    async def stream(self, req: GenerateRequest) -> AsyncIterator[str]:
        yield "hello"

    async def health(self) -> ProviderHealth:
        return ProviderHealth(reachable=True)


def test_stream_turn_id_consistent_and_nonempty(tmp_path, monkeypatch) -> None:
    """All events share one non-empty turn_id and the persisted trace matches it."""
    import june_brain.config as cfg

    monkeypatch.setattr(cfg, "MEMORY_DIR", str(tmp_path))

    loop = HandwrittenLoop(
        registry=_registry_with("local-fast", _SimpleProvider()),
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        maybe_compact=_noop_compact,
    )

    events = _collect(
        loop.stream_turn(
            SessionState(user_id="u1", messages=[]),
            Message(role="user", content="hi"),
        )
    )

    # Every event must carry a non-empty turn_id.
    assert events, "stream_turn emitted no events"
    turn_ids = {ev.turn_id for ev in events}
    assert None not in turn_ids, "some events are missing turn_id"
    assert len(turn_ids) == 1, f"events carry mixed turn_ids: {turn_ids}"
    (turn_id,) = turn_ids
    assert turn_id, "turn_id is empty"

    # The persisted trace file must exist with the same turn_id.
    traces_dir = tmp_path / "traces"
    trace_file = traces_dir / f"{turn_id}.json"
    assert trace_file.exists(), f"trace file not found: {trace_file}"
    data = json.loads(trace_file.read_text())
    assert data["turn_id"] == turn_id, (
        f"persisted turn_id {data['turn_id']!r} != stream turn_id {turn_id!r}"
    )
