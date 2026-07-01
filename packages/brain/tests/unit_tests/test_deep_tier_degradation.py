"""Tests for graceful degradation when the deep-tier model is not pulled.

Covers _route() falling back to the baseline provider when:
  (A) model_available returns False for the deep model
  (B) model_available returns True (no degradation)
"""

from __future__ import annotations

import asyncio
from typing import Any

from june_brain.loop.handwritten import HandwrittenLoop
from june_brain.providers.registry import ProviderRegistry

# ---------------------------------------------------------------------------
# Minimal fake provider — only the three attributes _route() inspects
# ---------------------------------------------------------------------------


class _FakeProvider:
    def __init__(self, model_id: str, tier: str, base_url: str) -> None:
        self.model_id = model_id
        self.tier = tier
        self.base_url = base_url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry(
    fast_provider: _FakeProvider, deep_provider: _FakeProvider
) -> ProviderRegistry:
    reg = ProviderRegistry(toml_data={"roles": {}, "providers": {}})
    reg.register("local-fast", fast_provider)  # type: ignore[arg-type]
    reg.register("local-deep", deep_provider)  # type: ignore[arg-type]
    return reg


async def _hard_classify(_text: str) -> Any:
    from june_brain.router.difficulty import DifficultyResult

    return DifficultyResult("hard", "heuristic")


async def _noop_compact(session: Any) -> bool:
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_route_degrades_when_deep_model_not_pulled() -> None:
    """Case A: model_available=False for e4b → _route returns fast provider."""
    fast = _FakeProvider("gemma4:e2b", "local-fast", "http://x/v1")
    deep = _FakeProvider("gemma4:e4b", "local-deep", "http://x/v1")
    reg = _make_registry(fast, deep)

    def model_available(model_id: str, base_url: str) -> bool:
        return model_id != "gemma4:e4b"

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
        classify=_hard_classify,
        model_available=model_available,
    )

    from june_brain.providers.base import Message

    async def _run() -> tuple[Any, str]:
        loop._reset_per_turn()
        return await loop._route(Message(role="user", content="explain something hard"))

    provider, chosen_role, _difficulty = asyncio.run(_run())

    assert provider.model_id == "gemma4:e2b", "should fall back to fast provider"
    assert chosen_role == "local-fast"
    assert loop._degrade_note != "", "degrade_note should be set"
    assert "gemma4:e4b" in loop._degrade_note


def test_route_does_not_degrade_when_deep_model_available() -> None:
    """Case B: model_available=True → _route returns deep provider, no degrade_note."""
    fast = _FakeProvider("gemma4:e2b", "local-fast", "http://x/v1")
    deep = _FakeProvider("gemma4:e4b", "local-deep", "http://x/v1")
    reg = _make_registry(fast, deep)

    def model_available(model_id: str, base_url: str) -> bool:
        return True

    loop = HandwrittenLoop(
        registry=reg,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=_noop_compact,
        classify=_hard_classify,
        model_available=model_available,
    )

    from june_brain.providers.base import Message

    async def _run() -> tuple[Any, str]:
        loop._reset_per_turn()
        return await loop._route(Message(role="user", content="explain something hard"))

    provider, chosen_role, _difficulty = asyncio.run(_run())

    assert provider.model_id == "gemma4:e4b", "should use deep provider when available"
    assert chosen_role == "local-deep"
    assert loop._degrade_note == "", "degrade_note should be empty when no degradation"
