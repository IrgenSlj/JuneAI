"""Tests for C.3: PinnedState, ContextAssembler, and Compactor.

All tests are sync or use asyncio.run().  No Ollama or network required —
mock providers are injected via ProviderRegistry.register().
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator  # noqa: TCH003

from june_brain.context.assembler import ContextAssembler, estimate_tokens
from june_brain.context.compactor import Compactor
from june_brain.context.pinned_state import PinnedState
from june_brain.loop.interface import SessionState
from june_brain.providers.base import (
    GenerateRequest,
    GenerateResult,
    Message,
    ProviderHealth,
)
from june_brain.providers.registry import ProviderRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockSummarizer:
    """Returns a fixed structured summary text."""

    def __init__(
        self,
        text: str = "GOAL: test goal\nCONFIRMED:\n- fact one\n- fact two",
        raises: bool = False,
        model_id: str = "mock-local",
        tier: str = "local-fast",
    ) -> None:
        self.model_id = model_id
        self.tier = tier
        self._text = text
        self._raises = raises

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        if self._raises:
            raise RuntimeError("provider unavailable")
        return GenerateResult(
            text=self._text,
            input_tokens=10,
            output_tokens=5,
            latency_ms=1,
            model_id=self.model_id,
            tier=self.tier,  # type: ignore[arg-type]
        )

    async def stream(self, req: GenerateRequest) -> AsyncIterator[str]:
        yield self._text

    async def health(self) -> ProviderHealth:
        return ProviderHealth(reachable=True, loaded=True)


def _registry_with(provider: object) -> ProviderRegistry:
    reg = ProviderRegistry(toml_data={"roles": {}, "providers": {}})
    reg.register("local-fast", provider)  # type: ignore[arg-type]
    return reg


def _make_messages(n: int, chars_each: int = 40) -> list[Message]:
    return [
        Message(role="user" if i % 2 == 0 else "assistant", content="x" * chars_each)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# PinnedState.merge — invariants
# ---------------------------------------------------------------------------


def test_merge_updates_goal_and_appends_constraints() -> None:
    ps = PinnedState(user_goal="old goal", constraints=["c1"])
    ps.merge(user_goal="new goal", constraints=["c2", "c3"])
    assert ps.user_goal == "new goal"
    assert "c1" in ps.constraints
    assert "c2" in ps.constraints
    assert "c3" in ps.constraints


def test_merge_does_not_blank_existing_goal_with_none() -> None:
    ps = PinnedState(user_goal="keep this")
    ps.merge(user_goal=None)
    assert ps.user_goal == "keep this"


def test_merge_does_not_blank_existing_goal_with_empty_string() -> None:
    ps = PinnedState(user_goal="keep this")
    ps.merge(user_goal="")
    assert ps.user_goal == "keep this"


def test_merge_all_none_leaves_state_unchanged() -> None:
    ps = PinnedState(
        user_goal="goal",
        constraints=["c1"],
        confirmed_facts=["f1"],
        open_questions=["q1"],
    )
    ps.merge()
    assert ps.user_goal == "goal"
    assert ps.constraints == ["c1"]
    assert ps.confirmed_facts == ["f1"]
    assert ps.open_questions == ["q1"]


def test_merge_deduplicates_list_items() -> None:
    ps = PinnedState(constraints=["c1", "c2"])
    ps.merge(constraints=["c2", "c3"])
    assert ps.constraints.count("c2") == 1
    assert "c3" in ps.constraints


def test_merge_caps_list_at_eight() -> None:
    ps = PinnedState()
    ps.merge(constraints=[f"item{i}" for i in range(10)])
    assert len(ps.constraints) <= 8


def test_merge_cap_keeps_most_recent() -> None:
    ps = PinnedState(constraints=[f"old{i}" for i in range(8)])
    ps.merge(constraints=["new_one"])
    assert "new_one" in ps.constraints
    assert len(ps.constraints) <= 8


def test_updated_at_set_after_merge() -> None:
    ps = PinnedState()
    assert ps.updated_at == ""
    ps.merge(user_goal="hello")
    assert ps.updated_at != ""


# ---------------------------------------------------------------------------
# PinnedState.to_block
# ---------------------------------------------------------------------------


def test_to_block_empty_returns_empty_string() -> None:
    ps = PinnedState()
    assert ps.to_block() == ""


def test_to_block_nonempty_contains_goal() -> None:
    ps = PinnedState(user_goal="ship Tier 1")
    block = ps.to_block()
    assert "ship Tier 1" in block


def test_is_empty() -> None:
    assert PinnedState().is_empty()
    ps = PinnedState(user_goal="x")
    assert not ps.is_empty()


# ---------------------------------------------------------------------------
# ContextAssembler — fixed order and trimming
# ---------------------------------------------------------------------------


def test_assembler_section1_is_system_and_first() -> None:
    assembler = ContextAssembler(system_prompt="sys prompt")
    session = SessionState(user_id="u1", messages=[])
    user_msg = Message(role="user", content="hello")
    ctx = assembler.assemble(session, user_msg)
    assert ctx[0].role == "system"
    # The system block leads with the configured prompt; the standing
    # untrusted-content rule (S6.1) is always appended after it.
    assert ctx[0].content.startswith("sys prompt")


def test_assembler_pinned_block_appears_as_system_message() -> None:
    assembler = ContextAssembler()
    session = SessionState(user_id="u1", messages=[])
    session.pinned.merge(user_goal="my goal")
    user_msg = Message(role="user", content="hi")
    ctx = assembler.assemble(session, user_msg)
    system_contents = [m.content for m in ctx if m.role == "system"]
    assert any("my goal" in c for c in system_contents)


def test_assembler_tools_block_appears_as_system_message() -> None:
    assembler = ContextAssembler(
        tools_block='You can call tools.\n{"tool_calls": [{"name": "web_search", "args": {}}]}'
    )
    session = SessionState(user_id="u1", messages=[])
    user_msg = Message(role="user", content="search the web")
    ctx = assembler.assemble(session, user_msg)
    system_contents = [m.content for m in ctx if m.role == "system"]
    assert any("web_search" in c for c in system_contents)


# ---------------------------------------------------------------------------
# Temporal context (D.1)
# ---------------------------------------------------------------------------


def test_temporal_block_pure_builder_formats_and_buckets() -> None:
    from datetime import datetime

    from june_brain.context.temporal import build_temporal_block

    block = build_temporal_block(datetime(2026, 7, 2, 20, 47))
    assert block.startswith("[Temporal context]")
    assert "Thursday" in block
    assert "2 July 2026" in block
    assert "8:47 PM" in block
    assert "evening" in block


def test_temporal_block_time_of_day_buckets() -> None:
    from datetime import datetime

    from june_brain.context.temporal import build_temporal_block

    assert "morning" in build_temporal_block(datetime(2026, 7, 2, 8, 0))
    assert "afternoon" in build_temporal_block(datetime(2026, 7, 2, 13, 0))
    assert "evening" in build_temporal_block(datetime(2026, 7, 2, 19, 0))
    assert "night" in build_temporal_block(datetime(2026, 7, 2, 23, 0))
    assert "night" in build_temporal_block(datetime(2026, 7, 2, 3, 0))
    # Midnight/noon 12-hour rendering is correct.
    assert "12:00 AM" in build_temporal_block(datetime(2026, 7, 2, 0, 0))
    assert "12:00 PM" in build_temporal_block(datetime(2026, 7, 2, 12, 0))


def test_assembler_temporal_block_present_when_clock_injected() -> None:
    from datetime import datetime

    fixed = datetime(2026, 7, 2, 9, 15)
    assembler = ContextAssembler(clock=lambda: fixed)
    session = SessionState(user_id="u1", messages=[])
    ctx = assembler.assemble(session, Message(role="user", content="hi"))
    system_contents = [m.content for m in ctx if m.role == "system"]
    assert any("[Temporal context]" in c and "morning" in c for c in system_contents)


def test_assembler_no_temporal_block_when_no_clock() -> None:
    assembler = ContextAssembler()
    session = SessionState(user_id="u1", messages=[])
    ctx = assembler.assemble(session, Message(role="user", content="hi"))
    assert not any("[Temporal context]" in m.content for m in ctx)


def test_assembler_temporal_clock_raising_degrades_gracefully() -> None:
    def _boom():  # type: ignore[no-untyped-def]
        raise RuntimeError("no clock")

    assembler = ContextAssembler(clock=_boom)
    session = SessionState(user_id="u1", messages=[])
    # The turn still assembles; the temporal block is simply omitted.
    ctx = assembler.assemble(session, Message(role="user", content="hi"))
    assert ctx[-1].content == "hi"
    assert not any("[Temporal context]" in m.content for m in ctx)


def test_assembler_no_tools_block_when_none() -> None:
    assembler = ContextAssembler()
    session = SessionState(user_id="u1", messages=[])
    user_msg = Message(role="user", content="hi")
    ctx = assembler.assemble(session, user_msg)
    # Only the section-1 system message — no tools block injected.
    assert len([m for m in ctx if m.role == "system"]) == 1


def test_assembler_reason_adds_think_instruction() -> None:
    assembler = ContextAssembler(system_prompt="base", reason=True)
    session = SessionState(user_id="u1", messages=[])
    ctx = assembler.assemble(session, Message(role="user", content="hi"))
    assert ctx[0].role == "system"
    assert "base" in ctx[0].content
    assert "<think>" in ctx[0].content


def test_assembler_reason_off_by_default() -> None:
    from june_brain.context.assembler import _REASONING_INSTRUCTION

    assembler = ContextAssembler(system_prompt="base")
    session = SessionState(user_id="u1", messages=[])
    ctx = assembler.assemble(session, Message(role="user", content="hi"))
    # Reason off: the <think> instruction is absent (the always-on guard rule
    # is appended, so the content is no longer exactly "base").
    assert ctx[0].content.startswith("base")
    assert _REASONING_INSTRUCTION not in ctx[0].content


def test_make_tools_block_never_raises() -> None:
    # Graceful degradation: returns a string (possibly empty) regardless of
    # whether skills/runtime are resolvable in the test environment.
    from june_brain.loop.wiring import make_tools_block

    assert isinstance(make_tools_block(), str)


def test_assembler_no_pinned_block_when_empty() -> None:
    assembler = ContextAssembler()
    session = SessionState(user_id="u1", messages=[])
    user_msg = Message(role="user", content="hi")
    ctx = assembler.assemble(session, user_msg)
    # Only one system message (section 1)
    system_msgs = [m for m in ctx if m.role == "system"]
    assert len(system_msgs) == 1


def test_assembler_user_msg_is_last() -> None:
    assembler = ContextAssembler()
    session = SessionState(user_id="u1", messages=_make_messages(3))
    user_msg = Message(role="user", content="final")
    ctx = assembler.assemble(session, user_msg)
    assert ctx[-1] is user_msg


def test_assembler_does_not_exceed_token_budget() -> None:
    budget = 500
    assembler = ContextAssembler(token_budget=budget)
    # Create a large session that would overflow without trimming
    session = SessionState(user_id="u1", messages=_make_messages(50, chars_each=100))
    session.pinned.merge(user_goal="goal")
    user_msg = Message(role="user", content="latest question")
    ctx = assembler.assemble(session, user_msg)
    total = sum(estimate_tokens(m.role + m.content) for m in ctx)
    assert total <= budget


def test_assembler_drops_oldest_raw_turns_keeps_newest() -> None:
    budget = 300
    assembler = ContextAssembler(token_budget=budget)
    msgs = [Message(role="user", content=f"msg{i}") for i in range(20)]
    session = SessionState(user_id="u1", messages=msgs)
    user_msg = Message(role="user", content="now")
    ctx = assembler.assemble(session, user_msg)
    raw = [m for m in ctx if m.role != "system" and m is not user_msg]
    if raw:
        # Newest message should survive
        assert raw[-1].content == "msg19"


def test_assembler_fixed_sections_always_present() -> None:
    budget = 200
    assembler = ContextAssembler(
        system_prompt="sys",
        character_block="char block",
        token_budget=budget,
    )
    session = SessionState(user_id="u1", messages=_make_messages(100, chars_each=80))
    session.pinned.merge(user_goal="goal")
    user_msg = Message(role="user", content="q")
    ctx = assembler.assemble(session, user_msg)
    contents = [m.content for m in ctx]
    # The system block leads with "sys" (the guard rule is appended after it).
    assert any(c.startswith("sys") for c in contents)
    assert "char block" in contents
    assert any("goal" in c for c in contents)


# ---------------------------------------------------------------------------
# Compactor — trigger and fallback
# ---------------------------------------------------------------------------


def test_compactor_below_threshold_returns_false() -> None:
    reg = _registry_with(MockSummarizer())
    compactor = Compactor(registry=reg, context_window=8192, threshold_ratio=0.70)
    session = SessionState(user_id="u1", messages=_make_messages(5, chars_each=10))
    result = asyncio.run(compactor.compact(session))
    assert not result
    assert result.compacted is False
    assert len(session.messages) == 5


def test_compactor_above_threshold_returns_true_and_drops_turns() -> None:
    reg = _registry_with(MockSummarizer())
    # Small window so 20 messages of 40 chars each trigger compaction
    compactor = Compactor(registry=reg, context_window=50, threshold_ratio=0.70)
    session = SessionState(user_id="u1", messages=_make_messages(20, chars_each=40))
    original_len = len(session.messages)
    result = asyncio.run(compactor.compact(session))
    assert result.compacted is True
    assert len(session.messages) < original_len


def test_compactor_merge_not_blank_invariant() -> None:
    reg = _registry_with(
        MockSummarizer(text="GOAL: new goal\nCONFIRMED:\n- new fact")
    )
    compactor = Compactor(registry=reg, context_window=50, threshold_ratio=0.70)
    session = SessionState(user_id="u1", messages=_make_messages(20, chars_each=40))
    session.pinned.merge(user_goal="original goal", constraints=["pre-existing constraint"])
    asyncio.run(compactor.compact(session))
    # user_goal updated (new value was given), pre-existing constraint preserved
    assert session.pinned.user_goal == "new goal"
    assert "pre-existing constraint" in session.pinned.constraints


def test_compactor_fallback_on_provider_raise() -> None:
    reg = _registry_with(MockSummarizer(raises=True))
    compactor = Compactor(registry=reg, context_window=50, threshold_ratio=0.70)
    session = SessionState(user_id="u1", messages=_make_messages(20, chars_each=40))
    session.pinned.merge(user_goal="must survive")
    # Should NOT raise, should return compacted=True (or drop oldest)
    result = asyncio.run(compactor.compact(session))
    assert result.compacted is True
    # Fallback path: no model call made, so metadata is None
    assert result.model_id is None
    # PinnedState must be intact
    assert session.pinned.user_goal == "must survive"


def test_compactor_fallback_capability_false() -> None:
    reg = _registry_with(MockSummarizer())
    compactor = Compactor(
        registry=reg,
        context_window=50,
        threshold_ratio=0.70,
        capability_ok=lambda: False,
    )
    session = SessionState(user_id="u1", messages=_make_messages(20, chars_each=40))
    session.pinned.merge(user_goal="intact goal")
    original_len = len(session.messages)
    result = asyncio.run(compactor.compact(session))
    assert result.compacted is True
    # Fallback path: no model call, metadata is None
    assert result.model_id is None
    # No summary produced, but turns dropped
    assert len(session.messages) < original_len
    # PinnedState unchanged
    assert session.pinned.user_goal == "intact goal"


def test_compactor_100_turn_invariant() -> None:
    """100-turn session: goal survives compaction and context comes back under threshold."""
    reg = _registry_with(
        MockSummarizer(text="GOAL: ship Tier 1\nCONFIRMED:\n- fact one")
    )
    context_window = 400
    compactor = Compactor(
        registry=reg,
        context_window=context_window,
        threshold_ratio=0.70,
    )
    # 100 messages × ~3 tokens each ≈ 300 tokens → above 280 threshold
    # after dropping oldest 50, ~150 tokens remain < 280
    session = SessionState(user_id="u1", messages=_make_messages(100, chars_each=8))
    session.pinned.merge(user_goal="ship Tier 1")

    asyncio.run(compactor.compact(session))

    # Goal must survive
    assert session.pinned.user_goal == "ship Tier 1"

    # Estimated context must be under threshold after compaction
    total_tokens = sum(
        estimate_tokens(m.role + m.content) for m in session.messages
    )
    assert total_tokens < context_window * 0.70


def test_compactor_model_path_outcome_has_metadata() -> None:
    """Model-path compaction returns CompactionOutcome with populated token/model fields."""
    reg = _registry_with(MockSummarizer(model_id="mock-local", text="GOAL: g\nCONFIRMED:\n- f"))
    compactor = Compactor(registry=reg, context_window=50, threshold_ratio=0.70)
    session = SessionState(user_id="u1", messages=_make_messages(20, chars_each=40))
    result = asyncio.run(compactor.compact(session))
    assert result.compacted is True
    assert result.model_id == "mock-local"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.latency_ms == 1


def test_compactor_below_threshold_outcome_has_no_metadata() -> None:
    """Below-threshold no-op: CompactionOutcome has compacted=False and no metadata."""
    reg = _registry_with(MockSummarizer())
    compactor = Compactor(registry=reg, context_window=8192, threshold_ratio=0.70)
    session = SessionState(user_id="u1", messages=_make_messages(5, chars_each=10))
    result = asyncio.run(compactor.compact(session))
    assert result.compacted is False
    assert result.model_id is None
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.latency_ms is None


def test_estimate_tokens_basic() -> None:
    assert estimate_tokens("hello") == 1
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("") == 1
