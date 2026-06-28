"""GemmaProvider suppresses a thinking model's separate reasoning field."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from june_brain.providers import GemmaProvider
from june_brain.providers.base import GenerateRequest, Message, ToolSpec


def _req() -> GenerateRequest:
    return GenerateRequest(messages=[Message(role="user", content="hi")], max_tokens=64)


def _chunk(content: str | None = None, reasoning_content: str | None = None) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, reasoning_content=reasoning_content, model_extra={})
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


async def _astream(chunks: list[SimpleNamespace]):
    for c in chunks:
        yield c


def test_reasoning_of_reads_known_fields() -> None:
    assert GemmaProvider._reasoning_of(SimpleNamespace(reasoning_content="r")) == "r"
    assert GemmaProvider._reasoning_of(SimpleNamespace(reasoning="r2")) == "r2"
    assert GemmaProvider._reasoning_of(SimpleNamespace(model_extra={"reasoning_content": "r3"})) == "r3"
    assert GemmaProvider._reasoning_of(SimpleNamespace()) == ""


def test_stream_suppresses_native_reasoning() -> None:
    provider = GemmaProvider(model_id="qwen3:8b", base_url="http://x/v1", tier="local-deep")
    chunks = [
        _chunk(reasoning_content="think a "),
        _chunk(reasoning_content="think b"),
        _chunk(content="the answer"),
    ]
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(return_value=_astream(chunks))

    async def drain() -> str:
        return "".join([c async for c in provider.stream(_req())])

    with patch.object(GemmaProvider, "_client", return_value=fake):
        out = asyncio.run(drain())

    assert out == "the answer"
    assert "think a" not in out
    assert "<think>" not in out


def test_generate_suppresses_native_reasoning() -> None:
    provider = GemmaProvider(model_id="qwen3:8b", base_url="http://x/v1", tier="local-deep")
    message = SimpleNamespace(content="the answer", reasoning_content="my reasoning", model_extra={})
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
    )
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(return_value=resp)

    with patch.object(GemmaProvider, "_client", return_value=fake):
        result = asyncio.run(provider.generate(_req()))

    assert result.text == "the answer"
    assert "my reasoning" not in result.text


def test_generate_without_reasoning_unchanged() -> None:
    provider = GemmaProvider(model_id="gemma4:e2b", base_url="http://x/v1", tier="local-fast")
    message = SimpleNamespace(content="plain answer", reasoning_content=None, model_extra={})
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
    )
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(return_value=resp)

    with patch.object(GemmaProvider, "_client", return_value=fake):
        result = asyncio.run(provider.generate(_req()))

    assert result.text == "plain answer"


def test_stream_forwards_tool_specs() -> None:
    provider = GemmaProvider(model_id="gemma4:e2b", base_url="http://x/v1", tier="local-fast")
    req = GenerateRequest(
        messages=[Message(role="user", content="hi")],
        max_tokens=64,
        tools=[
            ToolSpec(
                name="get_weather",
                description="Look up weather.",
                parameters={"type": "object", "properties": {}},
            )
        ],
    )
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(return_value=_astream([_chunk(content="ok")]))

    async def drain() -> str:
        return "".join([c async for c in provider.stream(req)])

    with patch.object(GemmaProvider, "_client", return_value=fake):
        assert asyncio.run(drain()) == "ok"

    kwargs = fake.chat.completions.create.call_args.kwargs
    assert kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up weather.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
