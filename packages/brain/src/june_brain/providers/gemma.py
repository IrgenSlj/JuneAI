"""Gemma 4 provider — local inference via Ollama's OpenAI-compatible endpoint.

Local provider: NEVER emits cloud-call provenance events. All inference stays
on this machine. keep_alive=-1 keeps the model resident between turns so
subsequent calls have no cold-start latency.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from .base import (
    GenerateRequest,
    GenerateResult,
    Message,
    ProviderHealth,
    parse_openai_tool_calls,
    tool_specs_to_openai,
)


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

    @staticmethod
    def _reasoning_of(obj: object) -> str:
        """Pull a thinking model's separate reasoning field, if present.

        Ollama's OpenAI-compatible endpoint returns qwen3's chain-of-thought in
        a separate ``reasoning_content`` (or ``reasoning``) field rather than
        inline ``<think>`` tags. Return it so callers can re-wrap it as
        ``<think>...</think>`` for the reasoning splitter. Returns "" if absent.
        """
        for attr in ("reasoning_content", "reasoning"):
            val = getattr(obj, attr, None)
            if isinstance(val, str) and val:
                return val
        extra = getattr(obj, "model_extra", None) or {}
        for key in ("reasoning_content", "reasoning"):
            val = extra.get(key)
            if isinstance(val, str) and val:
                return val
        return ""

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
        if req.tools:
            kwargs["tools"] = tool_specs_to_openai(req.tools)

        t0 = time.monotonic()
        response = await client.chat.completions.create(**kwargs)
        latency_ms = int((time.monotonic() - t0) * 1000)

        message = response.choices[0].message
        text = message.content or ""
        tool_calls = parse_openai_tool_calls(message)
        # Thinking models (qwen3) return reasoning in a separate field; re-wrap
        # it as <think>...</think> so split_reasoning / ReasoningSplitter handle
        # it uniformly with models that emit inline think tags.
        reasoning = self._reasoning_of(message)
        if reasoning:
            text = f"<think>{reasoning}</think>{text}"
        usage = response.usage
        return GenerateResult(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            model_id=self.model_id,
            tier=self.tier,  # type: ignore[arg-type]
            tool_calls=tool_calls,
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
        # Thinking models stream reasoning in a separate field before the answer.
        # Bridge it into inline <think>...</think> so the loop's ReasoningSplitter
        # routes it to the reasoning channel rather than the answer.
        in_think = False
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = self._reasoning_of(delta)
            if reasoning:
                if not in_think:
                    yield "<think>"
                    in_think = True
                yield reasoning
            content = delta.content
            if content:
                if in_think:
                    yield "</think>"
                    in_think = False
                yield content
        if in_think:
            yield "</think>"

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        """Embed texts via Ollama's OpenAI-compatible embeddings endpoint.

        Local provider: stays on this machine, never emits cloud provenance.
        ``model`` is the embedding model (e.g. ``nomic-embed-text``), distinct
        from the chat ``model_id``. Raises on transport/model error so the
        caller can degrade to keyword recall (invariant 6).
        """
        client = self._client()
        response = await client.embeddings.create(model=model, input=list(texts))
        # response.data preserves input order.
        return [list(item.embedding) for item in response.data]

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
