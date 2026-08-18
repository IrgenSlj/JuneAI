"""run_turn prefers provider-native tool calls, prose JSON as fallback (S5.2)."""

from __future__ import annotations

import asyncio
from typing import Any

from june_brain.loop.handwritten import HandwrittenLoop
from june_brain.loop.interface import SessionState
from june_brain.loop.wiring import _tool_to_spec
from june_brain.providers.base import GenerateResult, Message
from june_brain.providers.base import ToolCall as ProviderToolCall
from june_brain.providers.registry import ProviderRegistry


class _Tool:
    """Minimal Tool-shaped object for _tool_to_spec."""

    name = "get_weather"
    description = "Look up the weather."
    args = {"city": {"type": "str", "required": True}}


def test_tool_to_spec_builds_json_schema():
    spec = _tool_to_spec(_Tool())
    assert spec.name == "get_weather"
    assert spec.parameters == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }


def _registry_with(provider) -> ProviderRegistry:
    reg = ProviderRegistry(toml_data={"roles": {}, "providers": {}})
    reg.register("local-fast", provider)
    return reg


async def _fixed_classify(_text: str):
    from june_brain.router.difficulty import DifficultyResult

    return DifficultyResult("standard", "heuristic")


def _make_loop(provider, **kw) -> HandwrittenLoop:
    captured: dict[str, Any] = {}

    async def dispatch(tool_calls, session):
        captured["dispatched"] = [tc.name for tc in tool_calls]
        return [Message(role="tool", content="sunny")]

    async def _noop_compact(_session):
        return False

    loop = HandwrittenLoop(
        registry=_registry_with(provider),
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],  # prose path returns nothing
        dispatch=dispatch,
        maybe_compact=_noop_compact,
        classify=_fixed_classify,
        **kw,
    )
    loop._captured = captured  # type: ignore[attr-defined]
    return loop


class _NativeProvider:
    model_id = "mock"
    tier = "local-fast"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, req):
        self.calls += 1
        if self.calls == 1:
            return GenerateResult(
                text="",
                input_tokens=1,
                output_tokens=1,
                latency_ms=1,
                model_id="mock",
                tier="local-fast",
                tool_calls=[ProviderToolCall(name="get_weather", arguments={"city": "NYC"})],
            )
        return GenerateResult(
            text="It is sunny in NYC.",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            model_id="mock",
            tier="local-fast",
        )

    async def stream(self, req):
        # Mirrors generate(): run_turn drains stream_turn since D.4c, so the
        # native tool call has to arrive on the stream to be dispatched.
        from june_brain.providers.base import StreamDelta

        result = await self.generate(req)
        if result.tool_calls:
            yield StreamDelta(tool_calls=list(result.tool_calls))
        if result.text:
            yield StreamDelta(text=result.text)

    async def health(self):  # pragma: no cover - unused
        from june_brain.providers.base import ProviderHealth

        return ProviderHealth(reachable=True)


def test_run_turn_prefers_native_tool_calls():
    provider = _NativeProvider()
    loop = _make_loop(provider)
    session = SessionState(user_id="u", messages=[])
    result = asyncio.run(loop.run_turn(session, Message(role="user", content="weather?")))

    # The native tool call was dispatched even though the prose extractor returns [].
    assert loop._captured["dispatched"] == ["get_weather"]  # type: ignore[attr-defined]
    assert any(tc.name == "get_weather" for tc in result.tool_calls)


def test_run_turn_falls_back_to_prose_when_no_native():
    # Provider returns no native tool_calls; the injected prose extractor drives it.
    prose_calls = {"n": 0}

    class _ProseProvider(_NativeProvider):
        async def generate(self, req):
            return GenerateResult(
                text='{"tool_calls":[{"name":"get_weather","args":{}}]}',
                input_tokens=1, output_tokens=1, latency_ms=1,
                model_id="mock", tier="local-fast",
            )

    from june_brain.loop.interface import ToolCall as IfaceToolCall

    def prose_extract(result):
        prose_calls["n"] += 1
        if prose_calls["n"] == 1:
            return [IfaceToolCall(name="get_weather", args={})]
        return []

    provider = _ProseProvider()
    loop = _make_loop(provider)
    loop._extract_tool_calls = prose_extract  # type: ignore[attr-defined]
    session = SessionState(user_id="u", messages=[])
    result = asyncio.run(loop.run_turn(session, Message(role="user", content="weather?")))
    # Native tool_calls were empty, so the prose extractor drove the dispatch.
    assert loop._captured["dispatched"] == ["get_weather"]  # type: ignore[attr-defined]
    assert any(tc.name == "get_weather" for tc in result.tool_calls)
