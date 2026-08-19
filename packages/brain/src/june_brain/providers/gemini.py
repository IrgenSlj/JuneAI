"""Gemini provider — cloud inference via Google's OpenAI-compatible endpoint.

Privacy invariant: every network call is bracketed by record_cloud_call(start)
and record_cloud_call(end). This makes cloud egress visible in code (Part B,
invariant 1) and gives C.6 its hook.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ..failure import degrade_quietly
from .base import (
    GenerateRequest,
    GenerateResult,
    Message,
    ProviderHealth,
    StreamDelta,
    _assemble_tool_calls,
    parse_openai_tool_calls,
    tool_specs_to_openai,
)
from .provenance import CloudCallEvent, record_cloud_call

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def _resolve_api_key() -> str | None:
    """Read the Gemini key from env or the OS keychain. Returns None if absent."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
    if key and key.strip():
        return key.strip()
    # Fall back to keychain (may return None if no backend is present).
    try:
        from june_brain.config_store import GEMINI_KEY_NAME
        from june_brain.secret_store import load_secret

        stored = load_secret(GEMINI_KEY_NAME)
        if stored and stored.strip():
            return stored.strip()
    except Exception:  # noqa: BLE001
        degrade_quietly("Gemini key lookup")
    return None


class GeminiProvider:
    """Gemini 2 via Google's OpenAI-compatible REST endpoint — cloud-capable tier."""

    def __init__(
        self,
        model_id: str,
        base_url: str,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url
        self.tier = "cloud-capable"
        self._cached_client: AsyncOpenAI | None = None
        self._cached_key: str | None = None

    def _client(self, api_key: str) -> AsyncOpenAI:
        # Build once and reuse: a fresh AsyncOpenAI per call leaks an httpx
        # connection (a file descriptor) every turn. Rebuild only if the key
        # changes (e.g. the user rotates their API key in settings).
        # Import openai lazily here (not at module load) so merely importing the
        # provider layer — which happens on every API boot — doesn't drag in the
        # large openai SDK tree. It's only needed once an actual cloud call runs.
        if self._cached_client is None or self._cached_key != api_key:
            from openai import AsyncOpenAI

            self._cached_client = AsyncOpenAI(base_url=self.base_url, api_key=api_key)
            self._cached_key = api_key
        return self._cached_client

    def _build_messages(self, messages: list[Message]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _payload_summary(self, req: GenerateRequest) -> str:
        return f"{len(req.messages)} messages, {self.model_id}"

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        api_key = _resolve_api_key()
        if not api_key:
            raise ValueError(
                "Gemini API key is not set. "
                "Set GEMINI_API_KEY (or LLM_API_KEY) in your environment, "
                "or store it in the OS keychain via June's setup screen."
            )

        client = self._client(api_key)
        kwargs: dict = {
            "model": self.model_id,
            "messages": self._build_messages(req.messages),
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        }
        if req.stop:
            kwargs["stop"] = req.stop
        if req.response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        # Advertised again as of D.4b — the seam yields StreamDelta.
        if req.tools:
            kwargs["tools"] = tool_specs_to_openai(req.tools)

        summary = self._payload_summary(req)
        record_cloud_call(CloudCallEvent(model_id=self.model_id, phase="start", payload_summary=summary))
        t0 = time.monotonic()
        try:
            response = await client.chat.completions.create(**kwargs)
        finally:
            latency_ms = int((time.monotonic() - t0) * 1000)
            record_cloud_call(CloudCallEvent(model_id=self.model_id, phase="end", payload_summary=summary))

        message = response.choices[0].message
        text = message.content or ""
        usage = response.usage
        return GenerateResult(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            model_id=self.model_id,
            tier="cloud-capable",
            tool_calls=parse_openai_tool_calls(message),
        )

    async def stream(self, req: GenerateRequest) -> AsyncIterator[StreamDelta]:
        api_key = _resolve_api_key()
        if not api_key:
            raise ValueError(
                "Gemini API key is not set. "
                "Set GEMINI_API_KEY (or LLM_API_KEY) in your environment, "
                "or store it in the OS keychain via June's setup screen."
            )

        client = self._client(api_key)
        kwargs: dict = {
            "model": self.model_id,
            "messages": self._build_messages(req.messages),
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": True,
        }
        if req.stop:
            kwargs["stop"] = req.stop
        if req.response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        if req.tools:
            kwargs["tools"] = tool_specs_to_openai(req.tools)

        summary = self._payload_summary(req)
        record_cloud_call(CloudCallEvent(model_id=self.model_id, phase="start", payload_summary=summary))
        try:
            response = await client.chat.completions.create(**kwargs)
            pending: dict[int, dict[str, str]] = {}
            async for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield StreamDelta(text=delta.content)
                for frag in getattr(delta, "tool_calls", None) or []:
                    slot = pending.setdefault(
                        getattr(frag, "index", 0) or 0, {"name": "", "args": ""}
                    )
                    fn = getattr(frag, "function", None)
                    if fn is not None:
                        slot["name"] += getattr(fn, "name", "") or ""
                        slot["args"] += getattr(fn, "arguments", "") or ""
            calls = _assemble_tool_calls(pending)
            if calls:
                yield StreamDelta(tool_calls=calls)
        finally:
            record_cloud_call(CloudCallEvent(model_id=self.model_id, phase="end", payload_summary=summary))

    async def health(self) -> ProviderHealth:
        """Check that an API key exists and the endpoint is reachable. Never raises."""
        api_key = _resolve_api_key()
        if not api_key:
            return ProviderHealth(reachable=False, detail="No API key configured.")
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    self.base_url.rstrip("/") + "/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            if resp.status_code == 200:
                return ProviderHealth(reachable=True, loaded=True)
            return ProviderHealth(reachable=False, detail=f"HTTP {resp.status_code}")
        except Exception as exc:  # noqa: BLE001
            return ProviderHealth(reachable=False, detail=str(exc))
