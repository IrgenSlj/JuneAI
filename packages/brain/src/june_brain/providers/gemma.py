"""Gemma 4 provider — local inference via Ollama's OpenAI-compatible endpoint.

Local provider: NEVER emits cloud-call provenance events. All inference stays
on this machine. keep_alive=-1 keeps the model resident between turns so
subsequent calls have no cold-start latency.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from .base import GenerateRequest, GenerateResult, Message, ProviderHealth


class GemmaProvider:
    """Gemma 4 via Ollama — local-fast or local-deep tier."""

    def __init__(
        self,
        model_id: str,
        base_url: str,
        tier: str,
        api_key: str = "ollama",
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url
        self.tier = tier
        self._api_key = api_key
        self._cached_client: AsyncOpenAI | None = None

    def _client(self) -> AsyncOpenAI:
        # Build once and reuse: a new client per call would drop connection pooling.
        if self._cached_client is None:
            self._cached_client = AsyncOpenAI(base_url=self.base_url, api_key=self._api_key)
        return self._cached_client

    def _build_messages(self, messages: list[Message]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        client = self._client()
        kwargs: dict = {
            "model": self.model_id,
            "messages": self._build_messages(req.messages),
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            # Keep the model resident in memory between turns.
            "extra_body": {"keep_alive": -1},
        }
        if req.stop:
            kwargs["stop"] = req.stop
        if req.response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.monotonic()
        response = await client.chat.completions.create(**kwargs)
        latency_ms = int((time.monotonic() - t0) * 1000)

        text = response.choices[0].message.content or ""
        usage = response.usage
        return GenerateResult(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            model_id=self.model_id,
            tier=self.tier,  # type: ignore[arg-type]
        )

    async def stream(self, req: GenerateRequest) -> AsyncIterator[str]:
        client = self._client()
        kwargs: dict = {
            "model": self.model_id,
            "messages": self._build_messages(req.messages),
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": True,
            "extra_body": {"keep_alive": -1},
        }
        if req.stop:
            kwargs["stop"] = req.stop

        response = await client.chat.completions.create(**kwargs)
        async for chunk in response:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    async def health(self) -> ProviderHealth:
        """Probe Ollama reachability with a short timeout. Never raises."""
        import httpx

        base = self.base_url.rstrip("/")
        # Remove the /v1 suffix to reach the Ollama root health endpoint.
        root = base.removesuffix("/v1")
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{root}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models: list[dict] = data.get("models", [])
                loaded = any(
                    m.get("name", "").startswith(self.model_id.split(":")[0])
                    for m in models
                )
                return ProviderHealth(reachable=True, loaded=loaded)
            return ProviderHealth(reachable=False, detail=f"HTTP {resp.status_code}")
        except Exception as exc:  # noqa: BLE001
            return ProviderHealth(reachable=False, detail=str(exc))
