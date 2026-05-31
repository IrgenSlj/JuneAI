"""Tests for tool egress surfacing and user_id injection in the handwritten loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from june_brain.loop.handwritten import HandwrittenLoop
from june_brain.loop.interface import SessionState, StreamEvent, ToolCall
from june_brain.loop.wiring import is_network_tool, make_dispatch_fn
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


def test_is_network_tool() -> None:
    assert is_network_tool("web_search")
    assert is_network_tool("fetch_url")
    assert not is_network_tool("save_calendar_item")


def test_stream_turn_flags_egress_for_network_tool() -> None:
    """A web_search tool call is flagged network=True and lands in provenance.egress."""
    call_count = [0]

    class SeqProvider:
        model_id = "mock"
        tier = "local-fast"

        async def generate(self, req: GenerateRequest) -> GenerateResult:
            return GenerateResult(text="", input_tokens=1, output_tokens=1, latency_ms=1,
                                  model_id=self.model_id, tier=self.tier)

        async def stream(self, req: GenerateRequest) -> AsyncIterator[str]:
            call_count[0] += 1
            if call_count[0] == 1:
                yield '{"tool_calls": [{"name": "web_search", "args": {"query": "x"}}]}'
            else:
                yield "Here is what I found."

        async def health(self) -> ProviderHealth:
            return ProviderHealth(reachable=True)

    extract_calls = [0]

    def extractor(result: GenerateResult) -> list[ToolCall]:
        extract_calls[0] += 1
        if extract_calls[0] == 1:
            return [ToolCall(name="web_search", args={"query": "x"})]
        return []

    async def dispatch(tool_calls: list[ToolCall], session: SessionState) -> list[Message]:
        return [Message(role="tool", content="search results")]

    loop = HandwrittenLoop(
        registry=_registry_with("local-fast", SeqProvider()),
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=extractor,
        dispatch=dispatch,
        maybe_compact=_noop_compact,
    )

    events = _collect(loop.stream_turn(SessionState(user_id="u1", messages=[]),
                                       Message(role="user", content="search x")))

    tool_calls = [e for e in events if e.type == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0].network is True

    prov = next(e for e in events if e.type == "provenance").provenance
    assert prov is not None
    assert "web_search" in prov.egress


def test_local_only_blocks_networked_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under a Local-only dial, web_search is blocked (not dispatched) and emits
    a tool_blocked event; nothing egresses."""
    import june_brain.config_store as cfg
    from june_brain.routing import UserPrivacyDial

    monkeypatch.setattr(cfg, "get_privacy_dial", lambda: UserPrivacyDial.LOCAL_ONLY)

    class P:
        model_id = "mock"
        tier = "local-fast"

        async def generate(self, req: GenerateRequest) -> GenerateResult:
            return GenerateResult(text="", input_tokens=1, output_tokens=1, latency_ms=1,
                                  model_id=self.model_id, tier=self.tier)

        async def stream(self, req: GenerateRequest) -> AsyncIterator[str]:
            yield '{"tool_calls": [{"name": "web_search", "args": {"query": "x"}}]}'

        async def health(self) -> ProviderHealth:
            return ProviderHealth(reachable=True)

    dispatched: list[str] = []

    async def dispatch(tool_calls: list[ToolCall], session: SessionState) -> list[Message]:
        dispatched.extend(tc.name for tc in tool_calls)
        return [Message(role="tool", content="should not run")]

    loop = HandwrittenLoop(
        registry=_registry_with("local-fast", P()),
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [ToolCall(name="web_search", args={"query": "x"})],
        dispatch=dispatch,
        maybe_compact=_noop_compact,
    )

    events = _collect(loop.stream_turn(SessionState(user_id="u1", messages=[]),
                                       Message(role="user", content="search x")))

    assert any(e.type == "tool_blocked" and e.tool_name == "web_search" for e in events)
    assert "web_search" not in dispatched  # never ran
    prov = next(e for e in events if e.type == "provenance").provenance
    assert prov is not None and prov.egress == []  # nothing egressed


def test_dispatch_injects_user_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool whose schema requires user_id gets the session user_id injected."""
    captured: dict[str, Any] = {}

    class FakeTool:
        name = "save_thing"
        args = {"user_id": {}, "title": {}}

        def invoke(self, a: dict) -> str:
            captured.update(a)
            return "ok"

    import june_brain.graph as graph_mod

    monkeypatch.setattr(graph_mod, "_select_tools_for_runtime", lambda rt: [FakeTool()])

    dispatch = make_dispatch_fn([])
    obs = asyncio.run(
        dispatch(
            [ToolCall(name="save_thing", args={"title": "x"})],
            SessionState(user_id="alice", messages=[]),
        )
    )
    assert captured.get("user_id") == "alice"
    assert obs[0].content == "ok"
