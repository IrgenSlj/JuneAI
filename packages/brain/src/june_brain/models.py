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
    """Create a chat model for the resolved runtime."""
    return ChatOpenAI(
        model=runtime.model,
        api_key=SecretStr(runtime.api_key or "unused"),
        base_url=runtime.base_url,
        temperature=runtime.temperature,
        max_completion_tokens=runtime.max_tokens,
        streaming=True,
        timeout=120,
    )
