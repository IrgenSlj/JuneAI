"""Model client construction for JuneAI."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import RuntimeConfig


def build_chat_model(runtime: RuntimeConfig):
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
            model=runtime.model,
            anthropic_api_key=runtime.api_key,
            temperature=runtime.temperature,
            max_tokens=runtime.max_tokens,
            streaming=True,
        )

    return ChatOpenAI(
        model=runtime.model,
        openai_api_key=runtime.api_key,
        openai_api_base=runtime.base_url,
        temperature=runtime.temperature,
        max_tokens=runtime.max_tokens,
        streaming=True,
    )
