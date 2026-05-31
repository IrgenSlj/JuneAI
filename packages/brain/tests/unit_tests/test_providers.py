"""Tests for the C.1 provider layer.

All tests are synchronous and drive async methods via asyncio.run() so no
pytest-asyncio config is required.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch  # noqa: F401

import pytest
from june_brain.providers import (
    CloudCallEvent,
    GeminiProvider,
    GemmaProvider,
    GenerateRequest,
    GenerateResult,
    Message,
    Provider,
    ProviderHealth,
    get_registry,
    reset_cloud_call_recorder,
    reset_registry,
    set_cloud_call_recorder,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _req(n_messages: int = 1) -> GenerateRequest:
    msgs = [Message(role="user", content=f"msg {i}") for i in range(n_messages)]
    return GenerateRequest(messages=msgs, max_tokens=64)


def _fake_completion(text: str = "hello", prompt_tokens: int = 5, completion_tokens: int = 3):
    """Build a mock openai ChatCompletion response."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    choice = MagicMock()
    choice.message.content = text

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _fake_async_client(text: str = "hello") -> MagicMock:
    """Return a mock AsyncOpenAI whose chat.completions.create returns a fake response."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_fake_completion(text))
    return client


def _fake_chunk(content: str | None):
    """Build a mock streaming chunk exposing choices[0].delta.content."""
    delta = MagicMock()
    delta.content = content
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


async def _achunks(contents: list[str | None]):
    for c in contents:
        yield _fake_chunk(c)


def _fake_stream_client(contents: list[str | None]) -> MagicMock:
    """Mock AsyncOpenAI whose create(stream=True) awaits to an async chunk iterator."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_achunks(contents))
    return client


def _collect(provider, req: GenerateRequest) -> list[str]:
    async def run() -> list[str]:
        return [chunk async for chunk in provider.stream(req)]

    return asyncio.run(run())


# ---------------------------------------------------------------------------
# 1. Role → model resolution table
# ---------------------------------------------------------------------------

def test_registry_roles_match_toml() -> None:
    reset_registry()
    reg = get_registry()
    roles = reg.roles()
    assert roles["local-fast"] == "gemma4:e2b"
    assert roles["local-deep"] == "gemma4:e4b"
    assert roles["cloud-capable"] == "gemini-2.0-flash"
    reset_registry()


# ---------------------------------------------------------------------------
# 2. Mock-provider swap: register() swaps in with zero brain changes
# ---------------------------------------------------------------------------

class FakeProvider:
    """Trivial stand-in that satisfies the Provider protocol."""

    model_id = "fake-model"
    tier = "local-fast"

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        return GenerateResult(
            text="fake", input_tokens=0, output_tokens=0,
            latency_ms=0, model_id=self.model_id, tier="local-fast",
        )

    async def stream(self, req: GenerateRequest) -> AsyncIterator[str]:
        yield "fake"

    async def health(self) -> ProviderHealth:
        return ProviderHealth(reachable=True)


def test_register_swaps_provider() -> None:
    reset_registry()
    reg = get_registry()
    fake = FakeProvider()
    reg.register("local-fast", fake)
    assert reg.get("local-fast") is fake
    reset_registry()


def test_get_unknown_role_raises() -> None:
    reset_registry()
    reg = get_registry()
    with pytest.raises(KeyError, match="Unknown role"):
        reg.get("nonexistent-role")
    reset_registry()


# ---------------------------------------------------------------------------
# 3. Cloud-call provenance invariant
# ---------------------------------------------------------------------------

def test_gemini_generate_records_start_and_end() -> None:
    """GeminiProvider.generate must emit start then end CloudCallEvent."""
    events: list[CloudCallEvent] = []
    set_cloud_call_recorder(events.append)

    provider = GeminiProvider(model_id="gemini-2.0-flash", base_url="https://example.com/v1/")
    fake_client = _fake_async_client("world")

    with patch("june_brain.providers.gemini.GeminiProvider._client", return_value=fake_client):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            asyncio.run(provider.generate(_req()))

    reset_cloud_call_recorder()

    phases = [e.phase for e in events]
    assert phases == ["start", "end"], f"Expected [start, end], got {phases}"
    assert all(e.model_id == "gemini-2.0-flash" for e in events)


def test_gemma_generate_records_nothing() -> None:
    """GemmaProvider.generate must NEVER emit a cloud-call event."""
    events: list[CloudCallEvent] = []
    set_cloud_call_recorder(events.append)

    provider = GemmaProvider(
        model_id="gemma4:e2b",
        base_url="http://localhost:11434/v1",
        tier="local-fast",
    )
    fake_client = _fake_async_client("local response")

    with patch("june_brain.providers.gemma.GemmaProvider._client", return_value=fake_client):
        asyncio.run(provider.generate(_req()))

    reset_cloud_call_recorder()

    assert events == [], f"Gemma must never emit cloud events but got: {events}"


# ---------------------------------------------------------------------------
# 4. GeminiProvider raises ValueError when no API key is present
# ---------------------------------------------------------------------------

def test_gemini_no_api_key_raises() -> None:
    provider = GeminiProvider(model_id="gemini-2.0-flash", base_url="https://example.com/v1/")

    # Clear all key sources for this test.
    env_patch = {k: "" for k in ("GEMINI_API_KEY", "LLM_API_KEY")}
    with patch.dict(os.environ, env_patch):
        # Also patch load_secret to return None so keychain doesn't interfere.
        with patch("june_brain.providers.gemini._resolve_api_key", return_value=None):
            with pytest.raises(ValueError, match="Gemini API key is not set"):
                asyncio.run(provider.generate(_req()))


def test_gemini_caches_client_across_calls() -> None:
    """A fresh AsyncOpenAI per call would leak an httpx connection (FD) each turn."""
    provider = GeminiProvider(model_id="gemini-2.0-flash", base_url="https://example.com/v1/")
    c1 = provider._client("key-a")
    c2 = provider._client("key-a")
    assert c1 is c2  # reused across calls
    c3 = provider._client("key-b")
    assert c3 is not c1  # rebuilt only when the API key changes


# ---------------------------------------------------------------------------
# 5. GenerateResult.tier is correct per provider
# ---------------------------------------------------------------------------

def test_gemma_result_tier() -> None:
    provider = GemmaProvider(
        model_id="gemma4:e2b",
        base_url="http://localhost:11434/v1",
        tier="local-fast",
    )
    fake_client = _fake_async_client("hi")
    with patch("june_brain.providers.gemma.GemmaProvider._client", return_value=fake_client):
        result = asyncio.run(provider.generate(_req()))
    assert result.tier == "local-fast"


def test_gemma_deep_result_tier() -> None:
    provider = GemmaProvider(
        model_id="gemma4:e4b",
        base_url="http://localhost:11434/v1",
        tier="local-deep",
    )
    fake_client = _fake_async_client("hi")
    with patch("june_brain.providers.gemma.GemmaProvider._client", return_value=fake_client):
        result = asyncio.run(provider.generate(_req()))
    assert result.tier == "local-deep"


def test_gemini_result_tier() -> None:
    provider = GeminiProvider(model_id="gemini-2.0-flash", base_url="https://example.com/v1/")
    fake_client = _fake_async_client("cloud response")
    with patch("june_brain.providers.gemini.GeminiProvider._client", return_value=fake_client):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            result = asyncio.run(provider.generate(_req()))
    assert result.tier == "cloud-capable"


# ---------------------------------------------------------------------------
# 6. Streaming — call shape, yielded deltas, and the cloud-boundary invariant
# ---------------------------------------------------------------------------

def test_gemma_stream_yields_and_records_nothing() -> None:
    events: list[CloudCallEvent] = []
    set_cloud_call_recorder(events.append)

    provider = GemmaProvider(
        model_id="gemma4:e2b", base_url="http://localhost:11434/v1", tier="local-fast"
    )
    fake = _fake_stream_client(["he", None, "llo"])
    with patch("june_brain.providers.gemma.GemmaProvider._client", return_value=fake):
        chunks = _collect(provider, _req())

    reset_cloud_call_recorder()
    assert "".join(chunks) == "hello"
    assert events == [], f"Gemma streaming must never emit cloud events but got: {events}"


def test_gemini_stream_records_start_and_end() -> None:
    events: list[CloudCallEvent] = []
    set_cloud_call_recorder(events.append)

    provider = GeminiProvider(model_id="gemini-2.0-flash", base_url="https://example.com/v1/")
    fake = _fake_stream_client(["a", "b"])
    with patch("june_brain.providers.gemini.GeminiProvider._client", return_value=fake):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            chunks = _collect(provider, _req())

    reset_cloud_call_recorder()
    assert "".join(chunks) == "ab"
    assert [e.phase for e in events] == ["start", "end"]


# ---------------------------------------------------------------------------
# 7. health() never raises and skips the network when no key is configured
# ---------------------------------------------------------------------------

def test_gemini_health_no_key_is_unreachable() -> None:
    provider = GeminiProvider(model_id="gemini-2.0-flash", base_url="https://example.com/v1/")
    with patch("june_brain.providers.gemini._resolve_api_key", return_value=None):
        health = asyncio.run(provider.health())
    assert health.reachable is False


# ---------------------------------------------------------------------------
# 8. response_format=json is forwarded to the create call
# ---------------------------------------------------------------------------

def test_json_response_format_forwarded() -> None:
    provider = GemmaProvider(
        model_id="gemma4:e2b", base_url="http://localhost:11434/v1", tier="local-fast"
    )
    fake = _fake_async_client("{}")
    req = GenerateRequest(
        messages=[Message(role="user", content="hi")], max_tokens=16, response_format="json"
    )
    with patch("june_brain.providers.gemma.GemmaProvider._client", return_value=fake):
        asyncio.run(provider.generate(req))

    kwargs = fake.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# 9. FakeProvider satisfies the Provider protocol
# ---------------------------------------------------------------------------

def test_fake_provider_satisfies_protocol() -> None:
    assert isinstance(FakeProvider(), Provider)
