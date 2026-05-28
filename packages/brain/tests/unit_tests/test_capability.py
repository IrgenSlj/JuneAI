"""Tests for the capability probe (C.6a / D17).

All tests are synchronous; async methods are driven via asyncio.run().
No Ollama or network calls — mock provider only.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

from june_brain.capability.probe import (
    CapabilityProfile,
    get_capability_profile,
    refresh_capability_profile,
    reset_capability_cache,
    run_probe,
)
from june_brain.providers.base import GenerateResult
from june_brain.providers.registry import ProviderRegistry

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_result(text: str) -> GenerateResult:
    return GenerateResult(
        text=text,
        input_tokens=5,
        output_tokens=4,
        latency_ms=10,
        model_id="mock",
        tier="local-fast",
    )


def _registry_with_responses(*responses: str) -> ProviderRegistry:
    """Return a registry whose local-fast provider yields *responses* in order."""
    results = [_make_result(r) for r in responses]
    mock = AsyncMock()
    mock.model_id = "mock"
    mock.tier = "local-fast"
    mock.generate = AsyncMock(side_effect=results)
    reg = ProviderRegistry(toml_data={"roles": {}, "providers": {}})
    reg.register("local-fast", mock)
    return reg


def _registry_raising() -> ProviderRegistry:
    mock = AsyncMock()
    mock.model_id = "mock"
    mock.tier = "local-fast"
    mock.generate = AsyncMock(side_effect=RuntimeError("provider down"))
    reg = ProviderRegistry(toml_data={"roles": {}, "providers": {}})
    reg.register("local-fast", mock)
    return reg


# ---------------------------------------------------------------------------
# CapabilityProfile default (optimistic)
# ---------------------------------------------------------------------------


def test_default_profile_is_optimistic():
    reset_capability_cache()
    profile = get_capability_profile()
    assert profile.summarization == "good"
    assert profile.structured_output == "good"
    assert profile.long_context == "good"
    assert profile.relevance_scoring == "good"
    assert profile.checked_at == ""


# ---------------------------------------------------------------------------
# run_probe with correct answers → "good"
# ---------------------------------------------------------------------------


def test_run_probe_correct_answers_good():
    correct_struct = json.dumps({"name": "Alice", "age": 30, "city": "Paris"})
    reg = _registry_with_responses(
        # summarization — contains curie + radioactiv + nobel
        "Marie Curie was awarded Nobel Prizes for her research on radioactivity.",
        # structured output — valid JSON with required keys
        correct_struct,
        # long context — contains needle
        "blue-42",
        # relevance — correct pick
        "quantum entanglement",
    )
    profile = asyncio.run(run_probe(registry=reg))
    assert profile.summarization == "good"
    assert profile.structured_output == "good"
    assert profile.long_context == "good"
    assert profile.relevance_scoring == "good"
    assert profile.checked_at != ""


# ---------------------------------------------------------------------------
# run_probe with wrong/garbage answers → weak or poor
# ---------------------------------------------------------------------------


def test_run_probe_garbage_answers_weak_or_poor():
    reg = _registry_with_responses(
        "The weather is nice today.",   # summarization — no key terms
        "not json at all",              # structured output — unparseable
        "I don't know.",               # long context — needle missing
        "the recipe for pasta",        # relevance — wrong item
    )
    profile = asyncio.run(run_probe(registry=reg))
    assert profile.summarization in ("weak", "poor")
    assert profile.structured_output in ("weak", "poor")
    assert profile.long_context in ("weak", "poor")
    assert profile.relevance_scoring in ("weak", "poor")


# ---------------------------------------------------------------------------
# run_probe with raising provider → "poor", no exception
# ---------------------------------------------------------------------------


def test_run_probe_raising_provider_no_exception():
    reg = _registry_raising()
    profile = asyncio.run(run_probe(registry=reg))
    assert profile.summarization == "poor"
    assert profile.structured_output == "poor"
    assert profile.long_context == "poor"
    assert profile.relevance_scoring == "poor"


# ---------------------------------------------------------------------------
# get_capability_profile / refresh / reset
# ---------------------------------------------------------------------------


def test_refresh_updates_cache():
    reset_capability_cache()
    correct_struct = json.dumps({"name": "Bob", "age": 25, "city": "Rome"})
    reg = _registry_with_responses(
        "Marie Curie won Nobel prizes for radioactivity research.",
        correct_struct,
        "blue-42",
        "quantum entanglement",
    )
    profile = asyncio.run(refresh_capability_profile(registry=reg))
    assert profile.checked_at != ""
    cached = get_capability_profile()
    assert cached.checked_at == profile.checked_at
    assert cached.summarization == profile.summarization


def test_reset_restores_optimistic_default():
    reset_capability_cache()
    profile = get_capability_profile()
    assert profile.checked_at == ""
    assert profile.summarization == "good"


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


def test_profile_round_trip():
    p = CapabilityProfile(
        summarization="weak",
        structured_output="poor",
        long_context="good",
        relevance_scoring="weak",
        checked_at="2026-05-28T00:00:00+00:00",
    )
    d = p.to_dict()
    p2 = CapabilityProfile.from_dict(d)
    assert p2.summarization == "weak"
    assert p2.structured_output == "poor"
    assert p2.long_context == "good"
    assert p2.relevance_scoring == "weak"
    assert p2.checked_at == "2026-05-28T00:00:00+00:00"
