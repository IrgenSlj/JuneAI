"""Model client construction for JuneAI."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .config import RuntimeConfig


def build_chat_model(runtime: RuntimeConfig) -> Any:
    """Create a chat model for the resolved runtime."""
    if runtime.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - exercised by runtime, not tests
            raise RuntimeError(
                "Claude support requires 'langchain-anthropic'. Install project dependencies again."
            ) from exc

        if not runtime.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when MODEL_PROVIDER=anthropic.")

        return ChatAnthropic(
            model_name=runtime.model,
            api_key=SecretStr(runtime.api_key),
            temperature=runtime.temperature,
            max_tokens_to_sample=runtime.max_tokens,
            streaming=True,
            timeout=None,
            stop=None,
        )

    return ChatOpenAI(
        model=runtime.model,
        api_key=SecretStr(runtime.api_key),
        base_url=runtime.base_url,
        temperature=runtime.temperature,
        max_completion_tokens=runtime.max_tokens,
        streaming=True,
        timeout=120,  # prevent indefinite hang if Ollama goes down
    )
