"""Model client construction for June.

Per ADR 0002, both supported providers (Gemma via Ollama, Gemini via Google's
OpenAI-compatible endpoint) use LangChain's ChatOpenAI under the hood.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.outputs.chat_generation import ChatGenerationChunk
from pydantic import SecretStr

from .config import RuntimeConfig


class GeminiChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that captures Gemini thought_signature from extra_content.

    Gemini 3.x sends thought_signature in the OpenAI-compatible endpoint's
    ``extra_content`` field rather than on the ``function_call`` dict itself.
    The standard ChatOpenAI delta parser discards ``extra_content``, so the
    signature is lost on the round-trip and the next request gets a 400.

    This override captures ``extra_content`` from the raw delta and injects
    ``thought_signature`` directly into the ``function_call`` dict inside
    ``additional_kwargs`` so that LangChain's serialisation includes it.
    """

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return None
        choices = chunk.get("choices", [])
        if not choices:
            return generation_chunk
        delta = choices[0].get("delta", {})
        extra_content = delta.get("extra_content")
        if isinstance(extra_content, dict):
            akw = generation_chunk.message.additional_kwargs
            akw.setdefault("extra_content", extra_content)
            google = extra_content.get("google")
            if isinstance(google, dict):
                ts = google.get("thought_signature")
                if isinstance(ts, str):
                    fc = akw.get("function_call")
                    if isinstance(fc, dict) and "thought_signature" not in fc:
                        fc["thought_signature"] = ts
        return generation_chunk


def build_chat_model(runtime: RuntimeConfig) -> Any:
    """Create a chat model for the resolved runtime.

    Validates the two failure modes ChatOpenAI would otherwise hit only at
    first invoke: an empty API key on a non-local preset, and a missing
    model name. base_url shape is already validated at config-resolve time.
    """
    if not runtime.model or not runtime.model.strip():
        raise ValueError(
            f"Model name is empty for preset {runtime.preset_key!r}. "
            "Set MODEL_NAME or the preset-specific env var."
        )
    if runtime.is_api and not runtime.api_key.strip():
        raise ValueError(
            f"API key is empty for preset {runtime.preset_key!r}. "
            "Set GEMINI_API_KEY (or LLM_API_KEY) before calling the provider."
        )
    # For local Ollama, send keep_alive=-1 on every request so the model stays
    # resident between turns (Ollama otherwise resets a ~5min unload timer each
    # request, reintroducing cold starts). Cloud providers must not receive it.
    extra: dict[str, Any] = {}
    if runtime.is_local:
        extra["extra_body"] = {"keep_alive": -1}

    cls = GeminiChatOpenAI if runtime.is_api else ChatOpenAI
    return cls(
        model=runtime.model,
        api_key=SecretStr(runtime.api_key or "unused"),
        base_url=runtime.base_url,
        temperature=runtime.temperature,
        max_completion_tokens=runtime.max_tokens,
        streaming=True,
        timeout=120,
        **extra,
    )
