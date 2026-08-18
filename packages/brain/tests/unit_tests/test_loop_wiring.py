"""Tests for HandwrittenLoop default-seam wiring (Slice 2a).

Covers:
  (a) default extractor parses a JSON tool-call from model text into a ToolCall
  (b) default extractor returns [] for plain prose
  (c) default dispatcher executes a fake tool and degrades on unknown tool name
  (d) recall wiring: monkeypatched _recall_with_hits injects a system block
      and provenance.memories_recalled reflects the count
  (e) role routing: long/multi-question routes to local-deep when present;
      falls back to self._role when routed role is absent
  (f) provenance fields: skills_called, rationale non-empty, cloud_payload_summary
      None for local
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from june_brain.loop.handwritten import HandwrittenLoop
from june_brain.loop.interface import SessionState, ToolCall, TurnResult
from june_brain.loop.wiring import make_dispatch_fn, make_extract_tool_calls_fn
from june_brain.providers.base import (
    GenerateRequest,
    GenerateResult,
    Message,
    ProviderHealth,
    StreamDelta,
)
from june_brain.providers.registry import ProviderRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockProvider:
    def __init__(
        self,
        text: str = "mock reply",
        model_id: str = "mock-model",
        tier: str = "local-fast",
        input_tokens: int = 10,
        output_tokens: int = 5,
    ) -> None:
        self.model_id = model_id
        self.tier = tier
        self._text = text
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        return GenerateResult(
            text=self._text,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            latency_ms=1,
            model_id=self.model_id,
            tier=self.tier,
        )

    async def stream(self, req: GenerateRequest) -> AsyncIterator[str]:
        yield self._text

    async def health(self) -> ProviderHealth:
        return ProviderHealth(reachable=True, loaded=True)


def _registry_with(role: str, provider: Any) -> ProviderRegistry:
    registry = ProviderRegistry(toml_data={"roles": {}, "providers": {}})
    registry.register(role, provider)
    return registry


def _registry_with_two(
    role1: str, provider1: Any, role2: str, provider2: Any
) -> ProviderRegistry:
    registry = ProviderRegistry(toml_data={"roles": {}, "providers": {}})
    registry.register(role1, provider1)
    registry.register(role2, provider2)
    return registry


def _fake_result(text: str) -> GenerateResult:
    return GenerateResult(
        text=text,
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        model_id="test",
        tier="local-fast",
    )


# ---------------------------------------------------------------------------
# (a) default extractor parses JSON tool-call text into ToolCall
# ---------------------------------------------------------------------------


def test_default_extractor_parses_json_tool_call() -> None:
    extractor = make_extract_tool_calls_fn()
    json_text = '{"name": "weather", "args": {"city": "London"}}'
    result = extractor(_fake_result(json_text))
    assert len(result) == 1
    assert result[0].name == "weather"
    assert result[0].args == {"city": "London"}


def test_default_extractor_parses_tool_calls_list() -> None:
    extractor = make_extract_tool_calls_fn()
    json_text = '{"tool_calls": [{"name": "search", "args": {"q": "test"}}]}'
    result = extractor(_fake_result(json_text))
    assert len(result) == 1
    assert result[0].name == "search"


# ---------------------------------------------------------------------------
# (b) default extractor returns [] for plain prose
# ---------------------------------------------------------------------------


def test_default_extractor_returns_empty_for_plain_prose() -> None:
    extractor = make_extract_tool_calls_fn()
    result = extractor(_fake_result("Hello! How can I help you today?"))
    assert result == []


def test_default_extractor_returns_empty_for_empty_text() -> None:
    extractor = make_extract_tool_calls_fn()
    result = extractor(_fake_result(""))
    assert result == []


# ---------------------------------------------------------------------------
# (c) default dispatcher executes fake tool and degrades gracefully
# ---------------------------------------------------------------------------


def test_default_dispatcher_executes_tool() -> None:
    dispatched_names: list[str] = []
    dispatch_fn = make_dispatch_fn(dispatched_names)

    class FakeTool:
        name = "greet"

        def invoke(self, args: dict) -> str:
            return f"Hello, {args.get('name', 'world')}!"

    # Monkeypatch the tool map by replacing _get_tool_map via closure trick:
    # instead of that, we inject via the wiring path by patching agent_helpers imports.
    # Use a simpler approach: directly test via a loop with injected dispatch.
    # For make_dispatch_fn, we need to intercept _select_tools_for_runtime.
    # Let's do it via monkeypatching in a separate test using the full loop path.
    # Here we test the dispatch function directly with a custom resolver.

    # We'll verify the unknown-tool path still works.
    session = SessionState(user_id="u1", messages=[])
    tc_unknown = ToolCall(name="nonexistent_tool_xyz", args={})
    observations = asyncio.run(dispatch_fn([tc_unknown], session))
    assert len(observations) == 1
    assert "unknown tool" in observations[0].content.lower() or "nonexistent_tool_xyz" in observations[0].content
    assert dispatched_names == []


def test_default_dispatcher_unknown_tool_gives_error_observation_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatched_names: list[str] = []
    dispatch_fn = make_dispatch_fn(dispatched_names)

    session = SessionState(user_id="u1", messages=[])
    tc = ToolCall(name="definitely_not_a_real_tool", args={})
    # Must NOT raise; must return a Message with error text
    observations = asyncio.run(dispatch_fn([tc], session))
    assert len(observations) == 1
    assert isinstance(observations[0], Message)
    assert observations[0].role == "tool"


def test_default_dispatcher_with_mock_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatcher calls tool.invoke and records name in dispatched_names."""

    class FakeTool:
        name = "add_numbers"

        def invoke(self, args: dict) -> str:
            return str(args.get("a", 0) + args.get("b", 0))

    def fake_select_tools(_runtime: Any) -> list[Any]:
        return [FakeTool()]

    def fake_resolve_runtime() -> Any:
        return SimpleNamespace(preset_key="gemma")

    monkeypatch.setattr("june_brain.loop.agent_helpers._select_tools_for_runtime", fake_select_tools, raising=False)
    monkeypatch.setattr("june_brain.config.resolve_runtime_config", fake_resolve_runtime, raising=False)

    dispatched_names: list[str] = []
    dispatch_fn = make_dispatch_fn(dispatched_names)

    session = SessionState(user_id="u1", messages=[])
    tc = ToolCall(name="add_numbers", args={"a": 3, "b": 4})
    observations = asyncio.run(dispatch_fn([tc], session))
    assert len(observations) == 1
    assert "7" in observations[0].content
    assert "add_numbers" in dispatched_names


# ---------------------------------------------------------------------------
# (d) recall wiring: monkeypatched _recall_with_hits
# ---------------------------------------------------------------------------


def test_recall_wiring_injects_system_block_and_count() -> None:
    """Recall wiring via make_recall_fn: system block injected, memories_recalled set in loop."""
    from june_brain.context.assembler import ContextAssembler
    from june_brain.loop.wiring import make_recall_fn

    fake_hits = [{"content": "mem1"}, {"content": "mem2"}]
    fake_block = "## Recalled Memories\n- mem1\n- mem2"

    def fake_recall_with_hits(user_id: str, query: str, k: int = 5):
        return fake_block, fake_hits

    recall_fn, recall_state = make_recall_fn(recall_with_hits_fn=fake_recall_with_hits)
    assembler = ContextAssembler(recall=recall_fn)

    mock_provider = MockProvider(text="answer", tier="local-fast")
    reg = _registry_with("local-fast", mock_provider)

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=assembler.assemble,
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
    )
    # Wire recall_state into the loop so provenance reads from it
    loop._recall_state = recall_state

    session = SessionState(user_id="recall-user", messages=[])
    result: TurnResult = asyncio.run(
        loop.run_turn(session, Message(role="user", content="what do I like?"))
    )
    assert result.provenance.memories_recalled == 2


def test_recall_wiring_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test recall via make_recall_fn with injected recall_with_hits."""
    from june_brain.loop.wiring import make_recall_fn
    from june_brain.providers.base import Message

    fake_hits = [{"content": "mem1"}, {"content": "mem2"}, {"content": "mem3"}]
    fake_block = "## Recalled\n- mem1\n- mem2\n- mem3"

    def fake_recall_with_hits(user_id: str, query: str, k: int = 5):
        return fake_block, fake_hits

    recall_fn, state = make_recall_fn(recall_with_hits_fn=fake_recall_with_hits)
    session = SimpleNamespace(user_id="u99")
    msgs = recall_fn(session, Message(role="user", content="hello"))

    assert state["memories_recalled"] == 3
    assert len(msgs) == 1
    assert msgs[0].role == "system"
    assert "Recalled" in msgs[0].content


def test_recall_empty_block_returns_no_message() -> None:
    """Empty recall block → no injected message."""
    from june_brain.loop.wiring import make_recall_fn

    def fake_recall_with_hits(user_id: str, query: str, k: int = 5):
        return "", []

    recall_fn, state = make_recall_fn(recall_with_hits_fn=fake_recall_with_hits)
    session = SimpleNamespace(user_id="u99")
    msgs = recall_fn(session, Message(role="user", content="hello"))
    assert msgs == []
    assert state["memories_recalled"] == 0


# ---------------------------------------------------------------------------
# (e) role routing: difficulty-based selection with fallback
# ---------------------------------------------------------------------------


def test_role_routing_hard_uses_local_deep() -> None:
    """A long/multi-question prompt should route to local-deep when available."""
    local_fast = MockProvider(text="fast", tier="local-fast", model_id="fast-model")
    local_deep = MockProvider(text="deep", tier="local-deep", model_id="deep-model")
    reg = _registry_with_two("local-fast", local_fast, "local-deep", local_deep)

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
    )
    # Multi-question prompt → "hard" → local-deep
    long_prompt = "What is the weather? Then what should I eat? After that what should I do?"
    session = SessionState(user_id="route-user", messages=[])
    result: TurnResult = asyncio.run(
        loop.run_turn(session, Message(role="user", content=long_prompt))
    )
    assert result.provenance.tiers_used == ["local-deep"]


def test_role_routing_falls_back_when_role_absent() -> None:
    """When routed role is not in registry, falls back to self._role."""
    local_fast = MockProvider(text="fast answer", tier="local-fast", model_id="fast-model")
    # Only local-fast is registered, not local-deep
    reg = _registry_with("local-fast", local_fast)

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
    )
    # Multi-question prompt would route to local-deep, but it's absent → fallback
    long_prompt = "What is the weather? Then what should I eat? After that what should I do?"
    session = SessionState(user_id="fallback-user", messages=[])
    result: TurnResult = asyncio.run(
        loop.run_turn(session, Message(role="user", content=long_prompt))
    )
    assert result.provenance.tiers_used == ["local-fast"]
    assert result.assistant_msg.content == "fast answer"


def test_role_routing_trivial_stays_local_fast() -> None:
    """Trivial prompt stays on local-fast."""
    local_fast = MockProvider(text="hi!", tier="local-fast", model_id="fast-model")
    local_deep = MockProvider(text="deep reply", tier="local-deep", model_id="deep-model")
    reg = _registry_with_two("local-fast", local_fast, "local-deep", local_deep)

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
    )
    session = SessionState(user_id="trivial-user", messages=[])
    result: TurnResult = asyncio.run(
        loop.run_turn(session, Message(role="user", content="hi"))
    )
    assert result.provenance.tiers_used == ["local-fast"]


# ---------------------------------------------------------------------------
# (f) provenance fields: skills_called, rationale non-empty, cloud_payload_summary
# ---------------------------------------------------------------------------


def test_provenance_local_rationale_non_empty() -> None:
    mock = MockProvider(text="ans", tier="local-fast", model_id="gemma4:e2b")
    reg = _registry_with("local-fast", mock)

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
    )
    session = SessionState(user_id="prov-user", messages=[])
    result: TurnResult = asyncio.run(
        loop.run_turn(session, Message(role="user", content="hi"))
    )
    prov = result.provenance
    assert prov.rationale != ""
    assert prov.cloud_payload_summary is None
    assert isinstance(prov.skills_called, list)


def test_provenance_cloud_has_payload_summary() -> None:
    mock = MockProvider(text="cloud ans", tier="cloud-capable", model_id="gemini-flash")
    reg = _registry_with("cloud-capable", mock)

    loop = HandwrittenLoop(
        registry=reg,
        role="cloud-capable",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
    )
    session = SessionState(user_id="cloud-user", messages=[])
    result: TurnResult = asyncio.run(
        loop.run_turn(session, Message(role="user", content="help me now"))
    )
    prov = result.provenance
    assert prov.cloud_call is True
    assert prov.rationale != ""
    assert prov.cloud_payload_summary is not None
    assert "gemini-flash" in prov.cloud_payload_summary


def test_provenance_skills_called_populated(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a tool is dispatched, skills_called reflects its name."""

    class FakeTool:
        name = "get_weather"

        def invoke(self, args: dict) -> str:
            return "sunny"

    def fake_select_tools(_runtime: Any) -> list[Any]:
        return [FakeTool()]

    def fake_resolve_runtime() -> Any:
        return SimpleNamespace(preset_key="gemma")

    monkeypatch.setattr("june_brain.loop.agent_helpers._select_tools_for_runtime", fake_select_tools, raising=False)
    monkeypatch.setattr("june_brain.config.resolve_runtime_config", fake_resolve_runtime, raising=False)

    call_count = [0]

    class TwoPassProvider:
        model_id = "mock"
        tier = "local-fast"

        async def generate(self, req: GenerateRequest) -> GenerateResult:
            call_count[0] += 1
            if call_count[0] == 1:
                return GenerateResult(
                    text='{"name": "get_weather", "args": {"city": "NYC"}}',
                    input_tokens=5,
                    output_tokens=5,
                    latency_ms=1,
                    model_id="mock",
                    tier="local-fast",
                )
            return GenerateResult(
                text="It is sunny in NYC.",
                input_tokens=5,
                output_tokens=5,
                latency_ms=1,
                model_id="mock",
                tier="local-fast",
            )

        async def stream(self, req: GenerateRequest) -> AsyncIterator[StreamDelta]:
            # Delegates to generate() so the double has one source of truth for
            # what it says. run_turn drains stream_turn since D.4c, so a stubbed
            # stream() would make the fake contradict itself.
            result = await self.generate(req)
            if result.tool_calls:
                yield StreamDelta(tool_calls=list(result.tool_calls))
            if result.text:
                yield StreamDelta(text=result.text)

        async def health(self) -> ProviderHealth:
            return ProviderHealth(reachable=True)

    reg = _registry_with("local-fast", TwoPassProvider())

    async def _fixed_classify(_text: str):
        from june_brain.router.difficulty import DifficultyResult

        return DifficultyResult("standard", "heuristic")

    # Use the default extract + dispatch so the wiring is real. Inject a fixed
    # classifier so the routing classification doesn't consume a scripted
    # provider response.
    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        maybe_compact=_noop_compact,
        classify=_fixed_classify,
    )
    session = SessionState(user_id="skills-user", messages=[])
    result: TurnResult = asyncio.run(
        loop.run_turn(session, Message(role="user", content="what's the weather?"))
    )
    assert "get_weather" in result.provenance.skills_called


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _noop_compact(session: SessionState) -> bool:
    return False
