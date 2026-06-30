"""Tests for the difficulty classifier (C.6a).

All tests are synchronous; async methods are driven via asyncio.run().
No Ollama or network calls — mock provider only.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from june_brain.providers.base import GenerateResult
from june_brain.providers.registry import ProviderRegistry
from june_brain.router.difficulty import (
    classify_difficulty,
    classify_difficulty_detailed,
    heuristic_difficulty,
    reset_cache,
    tier_for_difficulty,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_cache()
    yield
    reset_cache()

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


def test_classify_parses_json_output():
    provider = _make_provider('{"difficulty": "hard"}')
    reg = _registry_with(provider)
    result = asyncio.run(classify_difficulty("a multi-step task", registry=reg))
    assert result == "hard"


# ---------------------------------------------------------------------------
# classify_difficulty_detailed — source + cache
# ---------------------------------------------------------------------------


def test_detailed_reports_model_source():
    provider = _make_provider("hard")
    reg = _registry_with(provider)
    result = asyncio.run(classify_difficulty_detailed("complex thing", registry=reg))
    assert result.label == "hard"
    assert result.source == "model"


def test_detailed_caches_model_result():
    provider = _make_provider("creative")
    reg = _registry_with(provider)
    first = asyncio.run(classify_difficulty_detailed("paint a scene", registry=reg))
    second = asyncio.run(classify_difficulty_detailed("paint a scene", registry=reg))
    assert first.source == "model"
    assert second.source == "cache"
    assert second.label == "creative"
    # The model was only consulted once.
    assert provider.generate.await_count == 1


def test_detailed_reports_heuristic_source_on_failure():
    provider = _make_raising_provider()
    reg = _registry_with(provider)
    result = asyncio.run(classify_difficulty_detailed("hi", registry=reg))
    assert result.label == "trivial"
    assert result.source == "heuristic"


def test_heuristic_multilingual_greetings():
    # Dutch and Greek greetings classify as trivial via the fallback.
    assert heuristic_difficulty("dankjewel voor je hulp vandaag echt") == "standard"
    assert heuristic_difficulty("καλησπέρα") == "trivial"
    assert heuristic_difficulty("goedemorgen") == "trivial"


# ---------------------------------------------------------------------------
# classify_difficulty_detailed — model_call metadata fields
# ---------------------------------------------------------------------------


def test_detailed_model_path_populates_metadata():
    """When the model path succeeds, DifficultyResult carries model_id/tokens/latency."""
    provider = _make_provider("hard")
    reg = _registry_with(provider)
    result = asyncio.run(classify_difficulty_detailed("complex multi-step task", registry=reg))
    assert result.source == "model"
    assert result.label == "hard"
    assert result.model_id == "mock"
    assert result.input_tokens == 5
    assert result.output_tokens == 2
    assert result.latency_ms == 10


def test_detailed_heuristic_leaves_metadata_none():
    """When the heuristic path is taken, all metadata fields are None."""
    provider = _make_raising_provider()
    reg = _registry_with(provider)
    result = asyncio.run(classify_difficulty_detailed("hi", registry=reg))
    assert result.source == "heuristic"
    assert result.model_id is None
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.latency_ms is None


def test_detailed_cache_leaves_metadata_none():
    """A cache hit has source='cache' and all metadata fields are None (honest: no call happened)."""
    provider = _make_provider("standard")
    reg = _registry_with(provider)
    asyncio.run(classify_difficulty_detailed("what time is it", registry=reg))
    result = asyncio.run(classify_difficulty_detailed("what time is it", registry=reg))
    assert result.source == "cache"
    assert result.model_id is None
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.latency_ms is None


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
