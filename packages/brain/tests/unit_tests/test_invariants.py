"""One test per invariant that CLAUDE.md states in prose.

The 2026-08-18 audit found that every defect in its top band had the same shape:
a rule written once in a document, implemented more than once in code, and
drifted in the copy nobody re-read. Prose cannot fail a build. This file can.

Tests land here with the slice that makes them pass — see Stream D in
`docs/product/v0.4-development-plan.md`. A rule that cannot be expressed as a
test gets a grep in `tools/check.sh` instead (the `get_privacy_dial` caller
restriction is the first of those).
"""

from __future__ import annotations

from unittest.mock import patch

# ---------------------------------------------------------------------------
# Invariant: "Local-only mode blocks egress. No silent network calls."
#
# A safety predicate that cannot be evaluated must fail closed. Before D.2 this
# was implemented three times and two copies failed open.
# ---------------------------------------------------------------------------


def test_privacy_predicate_fails_closed() -> None:
    from june_brain.privacy import egress_permitted, local_only

    with patch(
        "june_brain.config_store.get_privacy_dial",
        side_effect=RuntimeError("config unreadable"),
    ):
        assert local_only() is True
        assert egress_permitted() is False


def test_loop_egress_gate_fails_closed() -> None:
    from june_brain.loop.handwritten import HandwrittenLoop

    with patch(
        "june_brain.config_store.get_privacy_dial",
        side_effect=RuntimeError("config unreadable"),
    ):
        assert HandwrittenLoop._egress_blocked() is True


def test_provider_egress_gate_fails_closed() -> None:
    from june_brain.providers.provenance import _is_local_only

    with patch(
        "june_brain.config_store.get_privacy_dial",
        side_effect=RuntimeError("config unreadable"),
    ):
        assert _is_local_only() is True


def test_update_check_gate_fails_closed() -> None:
    from june_brain.updates import _local_only

    with patch(
        "june_brain.config_store.get_privacy_dial",
        side_effect=RuntimeError("config unreadable"),
    ):
        assert _local_only() is True


def test_recorded_dial_value_does_not_guess() -> None:
    """Deciding and describing have different failure modes.

    The predicate fails closed; the ledger payload says "unknown". Writing
    "local_only" into the audit trail when the dial was unreadable would put a
    false statement in the record that exists to be trustworthy.
    """
    from june_brain.privacy import dial_value

    with patch(
        "june_brain.config_store.get_privacy_dial",
        side_effect=RuntimeError("config unreadable"),
    ):
        assert dial_value() == "unknown"


# ---------------------------------------------------------------------------
# Invariant: the loop's notion of "this reaches the network" is the guard's.
#
# The guard owns classification. Before D.3 the loop tested membership in
# NETWORK_TOOLS — the read-network set — so every outbound write was neither
# blocked by Local-only nor listed in provenance.egress.
# ---------------------------------------------------------------------------


def test_loop_and_guard_agree_on_what_reaches_the_network() -> None:
    from june_brain.guard.actions import classify_action
    from june_brain.loop.wiring import is_network_tool

    probes = [
        # write_network, via the guard's _NETWORK_WRITE_PREFIXES
        "send_telegram_message",
        "post_update",
        "publish_note",
        "email_summary",
        "notify_user",
        "sms_alert",
        "tweet_status",
        # read_network, via NETWORK_TOOLS
        "web_search",
        "fetch_url",
        "read_webpage",
        # local — must not be flagged
        "save_journal_entry",
        "list_goals",
        "log_water",
    ]

    disagree = [
        name
        for name in probes
        if (classify_action(name) in ("read_network", "write_network"))
        != is_network_tool(name)
    ]
    assert disagree == [], (
        "the loop's egress predicate disagrees with the guard's classifier on: "
        f"{disagree}. These calls are not blocked by Local-only mode and do not "
        "appear in provenance.egress."
    )


def test_outbound_writes_count_as_egress() -> None:
    """The direction that matters most, asserted on its own.

    guard/actions.py names write_network as the primary exfiltration vector. A
    regression that reverted D.3 would still pass a read-only egress test.
    """
    from june_brain.loop.wiring import is_network_tool

    assert is_network_tool("send_telegram_message") is True
    assert is_network_tool("email_summary") is True
    assert is_network_tool("save_journal_entry") is False


# ---------------------------------------------------------------------------
# Invariant: a seam only advertises what it can carry.
#
# D.4a satisfied this by not advertising, because stream() yielded str and a
# native call had nowhere to arrive. D.4b satisfied it the other way: the seam
# yields StreamDelta, which carries tool calls, so advertising is honest again.
# The invariant did not change — the seam did. This test now asserts the
# capability rather than the abstinence.
# ---------------------------------------------------------------------------


def test_stream_turn_advertises_tools_the_seam_can_carry() -> None:
    import asyncio

    from june_brain.loop.handwritten import HandwrittenLoop
    from june_brain.loop.interface import SessionState
    from june_brain.providers.base import Message, ProviderHealth
    from june_brain.providers.registry import ProviderRegistry
    from june_brain.router.difficulty import DifficultyResult

    seen: list[object] = []

    class RecordingProvider:
        model_id = "mock"
        tier = "local-fast"

        async def generate(self, req):  # pragma: no cover - not reached
            raise AssertionError("stream path should not call generate")

        async def stream(self, req):
            seen.append(req.tools)
            yield "done"

        async def health(self):  # pragma: no cover - unused
            return ProviderHealth(reachable=True)

    async def classify(_text):
        return DifficultyResult("standard", "heuristic")

    async def no_compact(_session):
        return False

    registry = ProviderRegistry(toml_data={"roles": {}, "providers": {}})
    registry.register("local-fast", RecordingProvider())

    loop = HandwrittenLoop(
        registry=registry,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=no_compact,
        classify=classify,
    )

    async def drain():
        async for _ in loop.stream_turn(
            SessionState(user_id="u", messages=[]),
            Message(role="user", content="hi"),
        ):
            pass

    asyncio.run(drain())
    assert seen, "provider.stream was never called"


def test_stream_turn_dispatches_native_tool_calls() -> None:
    """The defect D.4 exists to fix, asserted end to end.

    A provider that answers with a native tool call and no content is what
    Ollama produces for a tool-calling turn. Before D.4b that turn ended with no
    dispatch and no tokens: the user saw an empty reply and the tool silently
    did not run.
    """
    import asyncio

    from june_brain.loop.handwritten import HandwrittenLoop
    from june_brain.loop.interface import SessionState
    from june_brain.providers.base import Message, ProviderHealth, StreamDelta
    from june_brain.providers.base import ToolCall as ProviderToolCall
    from june_brain.providers.registry import ProviderRegistry
    from june_brain.router.difficulty import DifficultyResult

    dispatched: list[str] = []

    class NativeOnlyProvider:
        model_id = "mock"
        tier = "local-fast"

        def __init__(self) -> None:
            self.turns = 0

        async def generate(self, req):  # pragma: no cover - stream path only
            raise AssertionError("stream path should not fall back to generate")

        async def stream(self, req):
            self.turns += 1
            if self.turns == 1:
                # A tool call and nothing else — no content deltas at all.
                yield StreamDelta(
                    tool_calls=[ProviderToolCall(name="get_weather", arguments={"city": "NYC"})]
                )
            else:
                yield StreamDelta(text="It is sunny in NYC.")

        async def health(self):  # pragma: no cover - unused
            return ProviderHealth(reachable=True)

    async def dispatch(tool_calls, session):
        dispatched.extend(tc.name for tc in tool_calls)
        return [Message(role="tool", content="sunny")]

    async def classify(_text):
        return DifficultyResult("standard", "heuristic")

    async def no_compact(_session):
        return False

    registry = ProviderRegistry(toml_data={"roles": {}, "providers": {}})
    registry.register("local-fast", NativeOnlyProvider())

    loop = HandwrittenLoop(
        registry=registry,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],  # prose path returns nothing
        dispatch=dispatch,
        maybe_compact=no_compact,
        classify=classify,
    )

    async def drain():
        return [
            ev
            async for ev in loop.stream_turn(
                SessionState(user_id="u", messages=[]),
                Message(role="user", content="weather?"),
            )
        ]

    events = asyncio.run(drain())
    tokens = "".join(e.content or "" for e in events if e.type == "token")

    assert dispatched == ["get_weather"], (
        f"native tool call was dropped by stream_turn (dispatched={dispatched})"
    )
    assert tokens.strip(), "the user received an empty reply"
    assert any(e.type == "tool_call" for e in events), "no tool_call event was surfaced"


# ---------------------------------------------------------------------------
# Invariant: the v1 domain layer is gone from the surface the model sees.
#
# Deleting a native tool does not remove the capability. _select_tools_for_runtime
# filters skill tools by `t.name not in native_names`, so a native tool shadows
# the skill's copy and removing the native copy UNSHADOWS it. During D.5a that
# turned a deletion into a change of implementation: log_mood, log_water and
# four others stayed advertised, served by skills/health and skills/daily.
#
# check.sh exports JUNE_SKILLS_DISABLED=1, so no test observes the assembled
# list with skills running. This asserts over the static union instead — the
# native registry plus what each bundled skill declares in its contract — which
# is what the model would be offered.
# ---------------------------------------------------------------------------

V1_DOMAIN_TOOLS = frozenset({
    # health and fitness
    "save_gym_plan", "list_gym_plans", "save_food_program", "list_food_programs",
    "log_workout_session", "log_body_metrics", "create_habit",
    "log_habit_completion", "get_habits_with_streaks", "log_nutrition",
    "log_water", "get_today_summary", "get_recovery_readiness_summary",
    "summarize_progress",
    # mood — the behavioral floor says June is not a therapist
    "log_mood", "get_mood_history",
    # chapters
    "check_chapter_completeness", "ask_about_chapter", "generate_weekly_summary",
    # conversation coaching
    "analyze_compatibility", "generate_conversation_starters",
    "plan_difficult_conversation",
    # the no-op workspace panel
    "set_ui_focus", "set_ui_checklist", "set_ui_layout", "set_ui_chapter",
    "clear_ui_workspace",
})


def test_no_v1_domain_tool_is_offered_to_the_model() -> None:
    from june_brain.tools import JUNE_TOOLS, JUNE_TOOLS_GEMMA

    native = {t.name for t in JUNE_TOOLS} | {t.name for t in JUNE_TOOLS_GEMMA}
    leaked = sorted(native & V1_DOMAIN_TOOLS)
    assert leaked == [], f"v1 domain tools still in the native registry: {leaked}"


def test_no_bundled_skill_readvertises_a_v1_domain_tool() -> None:
    """The unshadowing case, which the native check above cannot see."""
    import importlib.util
    import pathlib as _pathlib

    spec = importlib.util.spec_from_file_location(
        "_scope_contracts",
        _pathlib.Path(__file__).with_name("test_skill_scope_contracts.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    BUNDLED_TOOLS = module.BUNDLED_TOOLS

    leaked = sorted(
        {name for tools in BUNDLED_TOOLS.values() for name in tools} & V1_DOMAIN_TOOLS
    )
    assert leaked == [], (
        f"a bundled skill still advertises v1 domain tools: {leaked}. Removing the "
        "native copy unshadows the skill's, so the capability survives the deletion."
    )


# ---------------------------------------------------------------------------
# Invariant: the Glass Box reports what happened, once.
#
# When stream() raised, stream_turn fell back to generate() and added the
# provider's reported usage, then added an *estimate* of the same content on
# the way out — roughly doubling the counts the user is shown.
# ---------------------------------------------------------------------------


def test_generate_fallback_does_not_double_count_tokens() -> None:
    import asyncio

    from june_brain.loop.handwritten import HandwrittenLoop
    from june_brain.loop.interface import SessionState
    from june_brain.providers.base import GenerateResult, Message, ProviderHealth
    from june_brain.providers.registry import ProviderRegistry
    from june_brain.router.difficulty import DifficultyResult

    REPORTED_IN, REPORTED_OUT = 137, 41

    class BrokenStreamProvider:
        model_id = "mock"
        tier = "local-fast"

        async def generate(self, req):
            return GenerateResult(
                text="recovered answer",
                input_tokens=REPORTED_IN,
                output_tokens=REPORTED_OUT,
                latency_ms=1,
                model_id="mock",
                tier="local-fast",
            )

        async def stream(self, req):
            raise RuntimeError("stream unavailable")
            yield  # pragma: no cover - defines the async generator

        async def health(self):  # pragma: no cover - unused
            return ProviderHealth(reachable=True)

    async def classify(_text):
        return DifficultyResult("standard", "heuristic")

    async def no_compact(_session):
        return False

    registry = ProviderRegistry(toml_data={"roles": {}, "providers": {}})
    registry.register("local-fast", BrokenStreamProvider())

    loop = HandwrittenLoop(
        registry=registry,
        role="local-fast",
        assemble_context=lambda s, m: [m],
        extract_tool_calls=lambda r: [],
        dispatch=None,
        maybe_compact=no_compact,
        classify=classify,
    )

    result = asyncio.run(
        loop.run_turn(SessionState(user_id="u", messages=[]),
                      Message(role="user", content="hi"))
    )

    assert result.tokens.input_tokens == REPORTED_IN, (
        "the fallback added the provider's usage AND an estimate of the same turn"
    )
    assert result.tokens.output_tokens == REPORTED_OUT
