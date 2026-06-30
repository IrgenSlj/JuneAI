"""GemmaProvider surfaces a thinking model's separate reasoning field as inline think tags."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from june_brain.loop.reasoning import ReasoningSplitter, split_reasoning
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


def test_stream_surfaces_native_reasoning() -> None:
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

    # Reasoning is wrapped in think tags and appears before the answer.
    assert "<think>" in out
    assert "</think>" in out
    assert "think a" in out
    assert "think b" in out
    assert "the answer" in out
    # Answer must appear outside the think block.
    think_end = out.index("</think>")
    assert out.index("the answer") > think_end


def test_generate_surfaces_native_reasoning() -> None:
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

    # Reasoning is prefixed as an inline think block so split_reasoning can separate it.
    assert "<think>my reasoning</think>" in result.text
    assert "the answer" in result.text
    assert result.text.index("<think>") < result.text.index("the answer")


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


def test_stream_reasoning_splits_correctly_via_splitter() -> None:
    """End-to-end: stream output feeds through ReasoningSplitter and separates correctly."""
    provider = GemmaProvider(model_id="qwen3:8b", base_url="http://x/v1", tier="local-deep")
    chunks = [
        _chunk(reasoning_content="chain of thought"),
        _chunk(content="final answer"),
    ]
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(return_value=_astream(chunks))

    async def drain() -> str:
        return "".join([c async for c in provider.stream(_req())])

    with patch.object(GemmaProvider, "_client", return_value=fake):
        raw = asyncio.run(drain())

    splitter = ReasoningSplitter()
    segments = splitter.feed(raw) + splitter.flush()
    reasoning_text = "".join(t for kind, t in segments if kind == "reasoning")
    answer_text = "".join(t for kind, t in segments if kind == "answer")

    assert "chain of thought" in reasoning_text
    assert "final answer" in answer_text
    assert "chain of thought" not in answer_text


def test_generate_reasoning_splits_correctly_via_split_reasoning() -> None:
    """End-to-end: generate output feeds through split_reasoning and separates correctly."""
    provider = GemmaProvider(model_id="qwen3:8b", base_url="http://x/v1", tier="local-deep")
    message = SimpleNamespace(content="final answer", reasoning_content="chain of thought", model_extra={})
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
    )
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(return_value=resp)

    with patch.object(GemmaProvider, "_client", return_value=fake):
        result = asyncio.run(provider.generate(_req()))

    reasoning, answer = split_reasoning(result.text)
    assert "chain of thought" in reasoning
    assert "final answer" in answer
    assert "chain of thought" not in answer
