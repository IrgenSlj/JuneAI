"""Tests for the difficulty classifier (C.6a).

All tests are synchronous; async methods are driven via asyncio.run().
No Ollama or network calls — mock provider only.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from june_brain.providers.base import GenerateResult
from june_brain.providers.registry import ProviderRegistry
from june_brain.router.difficulty import (
    classify_difficulty,
    heuristic_difficulty,
    tier_for_difficulty,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_result(text: str) -> GenerateResult:
    return GenerateResult(
        text=text,
        input_tokens=5,
        output_tokens=2,
        latency_ms=10,
        model_id="mock",
        tier="local-fast",
    )


def _make_provider(response: str):
    mock = AsyncMock()
    mock.model_id = "mock"
    mock.tier = "local-fast"
    mock.generate = AsyncMock(return_value=_make_result(response))
    return mock


def _make_raising_provider():
    mock = AsyncMock()
    mock.model_id = "mock"
    mock.tier = "local-fast"
    mock.generate = AsyncMock(side_effect=RuntimeError("provider error"))
    return mock


def _registry_with(provider) -> ProviderRegistry:
    reg = ProviderRegistry(toml_data={"roles": {}, "providers": {}})
    reg.register("local-fast", provider)
    return reg


# ---------------------------------------------------------------------------
# heuristic_difficulty
# ---------------------------------------------------------------------------


def test_heuristic_greeting_trivial():
    assert heuristic_difficulty("hi") == "trivial"


def test_heuristic_short_trivial():
    assert heuristic_difficulty("hey there") == "trivial"


def test_heuristic_creative():
    assert heuristic_difficulty("write me a poem about the sea") == "creative"


def test_heuristic_creative_draft():
    assert heuristic_difficulty("draft a short story for me") == "creative"


def test_heuristic_hard_multi_step():
    assert (
        heuristic_difficulty(
            "First summarise the article, then compare it with the other one, "
            "and after that write a plan for improvement step by step"
        )
        == "hard"
    )


def test_heuristic_hard_multi_question():
    assert (
        heuristic_difficulty("What is the weather today? What should I wear?") == "hard"
    )


def test_heuristic_standard():
    assert heuristic_difficulty("What is the capital of France?") == "standard"


# ---------------------------------------------------------------------------
# classify_difficulty
# ---------------------------------------------------------------------------


def test_classify_returns_label_from_provider():
    provider = _make_provider("hard")
    reg = _registry_with(provider)
    result = asyncio.run(classify_difficulty("some complex multi-step question", registry=reg))
    assert result == "hard"


def test_classify_garbage_falls_back_to_heuristic():
    text = "What is the capital of France?"
    provider = _make_provider("definitely_not_a_label!!!")
    reg = _registry_with(provider)
    result = asyncio.run(classify_difficulty(text, registry=reg))
    assert result == heuristic_difficulty(text)


def test_classify_empty_response_falls_back():
    text = "Explain quantum computing."
    provider = _make_provider("")
    reg = _registry_with(provider)
    result = asyncio.run(classify_difficulty(text, registry=reg))
    assert result == heuristic_difficulty(text)


def test_classify_raising_provider_no_exception():
    text = "hi"
    provider = _make_raising_provider()
    reg = _registry_with(provider)
    result = asyncio.run(classify_difficulty(text, registry=reg))
    assert result == heuristic_difficulty(text)
    assert result == "trivial"


def test_classify_case_insensitive():
    provider = _make_provider("  CREATIVE  ")
    reg = _registry_with(provider)
    result = asyncio.run(classify_difficulty("write me a story", registry=reg))
    assert result == "creative"


# ---------------------------------------------------------------------------
# tier_for_difficulty
# ---------------------------------------------------------------------------


def test_tier_trivial():
    assert tier_for_difficulty("trivial") == "local-fast"


def test_tier_standard():
    assert tier_for_difficulty("standard") == "local-fast"


def test_tier_hard():
    assert tier_for_difficulty("hard") == "local-deep"


def test_tier_creative_not_cloud():
    result = tier_for_difficulty("creative")
    assert result == "local-deep"
    assert result != "cloud-capable"
