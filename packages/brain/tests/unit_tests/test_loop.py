"""Tests for the C.2 loop interface, both engines, engine selection, and experiment.

All tests are synchronous (asyncio.run).  No Ollama or network required:
mock providers are injected via ProviderRegistry.register().
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from june_brain.loop.engine import get_loop
from june_brain.loop.experiment import CLEAR_TASKS, run_clear, write_report
from june_brain.loop.handwritten import HandwrittenLoop
from june_brain.loop.interface import (
    HarnessLoop,
    SessionState,
    TokenAccounting,
    ToolCall,
    TurnProvenance,
    TurnResult,
)
from june_brain.providers.base import (
    GenerateRequest,
    GenerateResult,
    Message,
    ProviderHealth,
    StreamDelta,
)
from june_brain.providers.registry import ProviderRegistry

# ---------------------------------------------------------------------------
# Helpers — mock provider
# ---------------------------------------------------------------------------


class MockProvider:
    """Synchronous stand-in for a real provider; satisfies the Provider Protocol."""

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
        self.call_count = 0

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        self.call_count += 1
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
    """Build an isolated ProviderRegistry with one mock provider registered."""
    registry = ProviderRegistry(toml_data={"roles": {}, "providers": {}})
    registry.register(role, provider)
    return registry


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_handwritten_satisfies_protocol() -> None:
    mock = MockProvider()
    reg = _registry_with("local-fast", mock)
    loop = HandwrittenLoop(registry=reg)
    assert isinstance(loop, HarnessLoop)


# ---------------------------------------------------------------------------
# HandwrittenLoop — basic happy path
# ---------------------------------------------------------------------------


def test_handwritten_local_reply() -> None:
    mock = MockProvider(text="hello from mock", input_tokens=12, output_tokens=7)
    reg = _registry_with("local-fast", mock)
    loop = HandwrittenLoop(registry=reg, role="local-fast")

    session = SessionState(user_id="u1", messages=[])
    user_msg = Message(role="user", content="hi")
    result: TurnResult = asyncio.run(loop.run_turn(session, user_msg))

    assert result.assistant_msg.content == "hello from mock"
    assert result.provenance.cloud_call is False
    assert result.provenance.tiers_used == ["local-fast"]

    # Token counts are *estimates* on this path, not the provider's reported
    # usage. run_turn drains stream_turn since D.4c, and a streamed response
    # does not reliably carry a usage block, so the loop estimates from the
    # rendered prompt and the accumulated answer. This test used to assert the
    # provider's exact numbers (12/7) because run_turn called generate()
    # directly — but nothing in production ever took that path, so those were
    # never the numbers a user's Glass Box frame showed. Assert the property
    # that matters instead: counts are populated and plausible.
    assert result.tokens.input_tokens > 0
    assert result.tokens.output_tokens > 0
    assert result.tokens == TokenAccounting(
        input_tokens=result.provenance.input_tokens,
        output_tokens=result.provenance.output_tokens,
    )


def test_handwritten_cloud_provider_marks_cloud() -> None:
    mock = MockProvider(text="cloud reply", tier="cloud-capable", model_id="gemini-flash")
    reg = _registry_with("cloud-capable", mock)
    loop = HandwrittenLoop(registry=reg, role="cloud-capable")

    session = SessionState(user_id="u2", messages=[])
    user_msg = Message(role="user", content="help")
    result: TurnResult = asyncio.run(loop.run_turn(session, user_msg))

    assert result.provenance.cloud_call is True
    assert result.provenance.tiers_used == ["cloud-capable"]


# ---------------------------------------------------------------------------
# HandwrittenLoop — tool dispatch path
# ---------------------------------------------------------------------------


def test_tool_dispatch_called_once() -> None:
    """extract_tool_calls returns one ToolCall on pass 1, then [] on pass 2."""
    call_counter = [0]
    dispatch_log: list[list[ToolCall]] = []

    first_result = GenerateResult(
        text="calling tool",
        input_tokens=5,
        output_tokens=3,
        latency_ms=1,
        model_id="mock",
        tier="local-fast",
    )
    second_result = GenerateResult(
        text="final answer",
        input_tokens=5,
        output_tokens=3,
        latency_ms=1,
        model_id="mock",
        tier="local-fast",
    )

    results_sequence = [first_result, second_result]

    class SequentialProvider:
        model_id = "mock"
        tier = "local-fast"

        async def generate(self, req: GenerateRequest) -> GenerateResult:
            idx = min(call_counter[0], len(results_sequence) - 1)
            call_counter[0] += 1
            return results_sequence[idx]

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

    reg = _registry_with("local-fast", SequentialProvider())

    extract_calls = [0]

    def extractor(result: GenerateResult) -> list[ToolCall]:
        extract_calls[0] += 1
        if extract_calls[0] == 1:
            return [ToolCall(name="search", args={"q": "test"})]
        return []

    async def dispatch(
        tool_calls: list[ToolCall], session: SessionState
    ) -> list[Message]:
        dispatch_log.append(list(tool_calls))
        return [Message(role="tool", content="search result")]

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        extract_tool_calls=extractor,
        dispatch=dispatch,
    )
    session = SessionState(user_id="u3", messages=[])
    result = asyncio.run(loop.run_turn(session, Message(role="user", content="search for me")))

    assert len(dispatch_log) == 1
    assert dispatch_log[0][0].name == "search"
    assert result.assistant_msg.content == "final answer"


def test_max_iterations_terminates() -> None:
    """An extractor that always returns a tool call must stop at max_iterations."""
    mock = MockProvider(text="looping")
    reg = _registry_with("local-fast", mock)
    dispatch_count = [0]

    def always_tool(result: GenerateResult) -> list[ToolCall]:
        return [ToolCall(name="always", args={})]

    async def dispatch(
        tool_calls: list[ToolCall], session: SessionState
    ) -> list[Message]:
        dispatch_count[0] += 1
        return [Message(role="tool", content="obs")]

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        extract_tool_calls=always_tool,
        dispatch=dispatch,
        max_iterations=3,
    )
    session = SessionState(user_id="u4", messages=[])
    result = asyncio.run(loop.run_turn(session, Message(role="user", content="go")))

    assert dispatch_count[0] < 10
    assert result.assistant_msg.content == "looping"


# ---------------------------------------------------------------------------
# Fixed-shape order: assemble_context before provider, maybe_compact after loop
# ---------------------------------------------------------------------------


def test_fixed_shape_order() -> None:
    order_log: list[str] = []
    mock = MockProvider(text="ok")
    reg = _registry_with("local-fast", mock)

    def spy_assemble(session: SessionState, user_msg: Message) -> list[Message]:
        order_log.append("assemble")
        return [Message(role="system", content="sys"), user_msg]

    async def spy_compact(session: SessionState) -> bool:
        order_log.append("compact")
        return False

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=spy_assemble,
        maybe_compact=spy_compact,
    )
    session = SessionState(user_id="u5", messages=[])
    asyncio.run(loop.run_turn(session, Message(role="user", content="hey")))

    assert order_log[0] == "assemble"
    assert order_log[-1] == "compact"
    assert order_log.count("compact") == 1


# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------


def test_get_loop_handwritten() -> None:
    loop = get_loop("handwritten")
    assert isinstance(loop, HandwrittenLoop)


def test_get_loop_unknown_raises() -> None:
    with pytest.raises(ValueError, match="nope"):
        get_loop("nope")


# ---------------------------------------------------------------------------
# CLEAR experiment — lightweight mock engines
# ---------------------------------------------------------------------------


class _FastEngine:
    """Trivial mock engine that always succeeds quickly."""

    async def run_turn(self, session: SessionState, user_msg: Message) -> TurnResult:
        return TurnResult(
            assistant_msg=Message(role="assistant", content="fast answer"),
            tool_calls=[],
            provenance=TurnProvenance(
                tiers_used=["local-fast"],
                cloud_call=False,
                model_ids=["mock-fast"],
                rationale="mock",
            ),
            tokens=TokenAccounting(input_tokens=5, output_tokens=3),
            compacted=False,
        )


class _SlowEngine:
    """Trivial mock engine that always succeeds slightly slower."""

    async def run_turn(self, session: SessionState, user_msg: Message) -> TurnResult:
        return TurnResult(
            assistant_msg=Message(role="assistant", content="slow answer"),
            tool_calls=[],
            provenance=TurnProvenance(
                tiers_used=["local-fast"],
                cloud_call=False,
                model_ids=["mock-slow"],
                rationale="mock",
            ),
            tokens=TokenAccounting(input_tokens=20, output_tokens=10),
            compacted=False,
        )


def test_run_clear_and_write_report(tmp_path: Path) -> None:
    engines: dict[str, Any] = {
        "fast_engine": _FastEngine(),
        "slow_engine": _SlowEngine(),
    }
    results = asyncio.run(run_clear(engines, CLEAR_TASKS, runs=2))

    assert "fast_engine" in results
    assert "slow_engine" in results
    for task in CLEAR_TASKS:
        assert task.name in results["fast_engine"]
        assert len(results["fast_engine"][task.name]) == 2

    out = tmp_path / "loop-clear.md"
    write_report(results, str(out))

    assert out.exists()
    content = out.read_text()
    assert "fast_engine" in content
    assert "slow_engine" in content
    for col in ("Cost", "Latency", "Efficacy", "Assurance", "Reliability"):
        assert col in content
