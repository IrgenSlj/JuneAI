"""Model client construction for June.

Per ADR 0002, both supported providers (Gemma via Ollama, Gemini via Google's
OpenAI-compatible endpoint) use LangChain's ChatOpenAI under the hood.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .config import RuntimeConfig


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
    return ChatOpenAI(
        model=runtime.model,
        api_key=SecretStr(runtime.api_key or "unused"),
        base_url=runtime.base_url,
        temperature=runtime.temperature,
        max_completion_tokens=runtime.max_tokens,
        streaming=True,
        timeout=120,
    )
