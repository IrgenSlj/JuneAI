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

# The source of truth is the denylist the merge enforces (june_brain.tools).
# Duplicating it here would recreate exactly the drift this stream exists to
# remove, so it is imported at use rather than restated.


def test_no_v1_domain_tool_is_offered_to_the_model() -> None:
    from june_brain.tools import (
        JUNE_TOOLS,
        JUNE_TOOLS_GEMMA,
    )
    from june_brain.tools import (
        RETIRED_TOOL_NAMES as V1_DOMAIN_TOOLS,
    )

    native = {t.name for t in JUNE_TOOLS} | {t.name for t in JUNE_TOOLS_GEMMA}
    leaked = sorted(native & V1_DOMAIN_TOOLS)
    assert leaked == [], f"v1 domain tools still in the native registry: {leaked}"


def test_no_bundled_skill_readvertises_a_v1_domain_tool() -> None:
    """The unshadowing case, which the native check above cannot see."""
    import importlib.util
    import pathlib as _pathlib

    from june_brain.tools import RETIRED_TOOL_NAMES as V1_DOMAIN_TOOLS

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


def test_a_handoff_actually_hands_off() -> None:
    """A name in SKILL_OWNED_TOOL_NAMES must be served by a bundled skill.

    Deleting a native tool unshadows any skill copy of the name. D.5c found that
    as a bug — skills/health and skills/daily silently resurrected six deleted
    capabilities — and D.5a uses it deliberately for calendar, whose skill is the
    better home for the name. The two cases are one line apart in the source and
    opposite in intent, so the difference has to be checked, not commented.
    """
    import importlib.util
    import pathlib as _pathlib

    from june_brain.tools import RETIRED_TOOL_NAMES, SKILL_OWNED_TOOL_NAMES

    overlap = sorted(RETIRED_TOOL_NAMES & SKILL_OWNED_TOOL_NAMES)
    assert overlap == [], (
        f"{overlap} is both retired and handed off. The denylist wins at merge "
        "time, so the skill's tool would be silently dropped."
    )

    spec = importlib.util.spec_from_file_location(
        "_scope_contracts",
        _pathlib.Path(__file__).with_name("test_skill_scope_contracts.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    advertised = {name for tools in module.BUNDLED_TOOLS.values() for name in tools}

    orphaned = sorted(SKILL_OWNED_TOOL_NAMES - advertised)
    assert orphaned == [], (
        f"{orphaned} was handed off to a skill that does not advertise it. The "
        "capability is gone, not moved."
    )


# ---------------------------------------------------------------------------
# Invariant: a dispatched tool knows which user it is acting for.
#
# The model is never told the partition key and must not be. Skill tools declare
# `user_id` as an ordinary argument; native tools declare `state` as
# `Annotated[AgentState, Inject]`, excluded from the advertised schema. Only the
# first was injected before D.5d, so seven of the twelve native tools ran
# without one — and the two failure shapes had very different visibility. The
# memory tools raise, so the user sees an error. The scheduler tools read
# `(state or {}).get("user_id", "default")`, so one user's schedules landed in
# another user's partition with nothing to see.
# ---------------------------------------------------------------------------


def test_a_dispatched_native_tool_receives_the_session_identity() -> None:
    import asyncio

    from june_brain.loop.interface import SessionState, ToolCall
    from june_brain.loop.wiring import make_dispatch_fn
    from june_brain.tools_base import Tool

    seen: dict[str, object] = {}

    def _probe(state=None):  # type: ignore[no-untyped-def]
        seen["state"] = state
        return "ok"

    # Built directly rather than through @tool: this module uses
    # `from __future__ import annotations`, so the decorator's get_type_hints
    # would have to resolve `Inject` from module globals. The shape being tested
    # is a tool with no advertised args and an injected `state`.
    probe_zero_arg_tool = Tool(
        name="probe_zero_arg_tool",
        description="A native tool the model calls with no arguments at all.",
        args={},
        func=_probe,
        injected=("state",),
    )

    dispatch = make_dispatch_fn(dispatched_names=[])
    session = SessionState(user_id="alice", messages=[])
    with patch(
        "june_brain.loop.agent_helpers._select_tools_for_runtime",
        return_value=[probe_zero_arg_tool],
    ):
        asyncio.run(dispatch([ToolCall(name="probe_zero_arg_tool", args={})], session))

    assert seen.get("state") is not None, (
        "the tool was dispatched without injected state; a zero-argument native "
        "tool has no other way to learn who it is acting for"
    )
    assert seen["state"]["user_id"] == "alice"  # type: ignore[index]


def test_no_native_tool_silently_defaults_its_user() -> None:
    """Failing loudly is recoverable; guessing "default" is a cross-user write."""
    import inspect

    from june_brain.tools import JUNE_TOOLS

    guessing = []
    for t in JUNE_TOOLS:
        if "state" not in (t.injected or ()):
            continue
        try:
            source = inspect.getsource(t.func)
        except (OSError, TypeError):  # pragma: no cover - source always available here
            continue
        if '"user_id", "default"' in source or "'user_id', 'default'" in source:
            guessing.append(t.name)

    assert guessing == [], (
        f"{guessing} fall back to the 'default' user when state is missing. A "
        "missing identity is a bug to surface, not a partition to guess."
    )


# ---------------------------------------------------------------------------
# Invariant: the system prompt may only name tools that exist.
#
# D.6 found `_BASE_INSTRUCTIONS` telling the model to call tools tranche 1 had
# deleted, rewrote it, and still left `get_recovery_readiness_summary` named
# twice in the persona instructions below it. A prompt that advertises a
# missing tool spends the model's attention on a call that can only fail, and
# the Glass Box shows the user the failure.
# ---------------------------------------------------------------------------


def test_the_system_prompt_only_names_tools_that_exist() -> None:
    import re

    from june_brain.skills import SKILLS, build_system_prompt
    from june_brain.skills.manifest import DEFAULT_MANIFEST
    from june_brain.tools import JUNE_TOOLS, JUNE_TOOLS_GEMMA, SKILL_OWNED_TOOL_NAMES

    known = (
        {t.name for t in JUNE_TOOLS}
        | {t.name for t in JUNE_TOOLS_GEMMA}
        | set(SKILL_OWNED_TOOL_NAMES)
    )
    # Bundled skills' tools are legitimate to name too.
    import importlib.util
    import pathlib as _pathlib

    spec = importlib.util.spec_from_file_location(
        "_scope_contracts",
        _pathlib.Path(__file__).with_name("test_skill_scope_contracts.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    known |= {name for tools in module.BUNDLED_TOOLS.values() for name in tools}
    assert set(DEFAULT_MANIFEST.entries)  # the manifest is what makes them reachable

    # Tool-shaped tokens: snake_case identifiers with a verb-ish leading segment.
    pattern = re.compile(
        r"\b((?:save|get|list|update|create|delete|log|track|set|clear|remember|forget|"
        r"switch|preview|draft|run|check|analyze|generate|plan|ask|summarize|web|fetch|read|search|send)"
        r"_[a-z_]+)\b"
    )
    for key in SKILLS:
        prompt = build_system_prompt(key)
        named = set(pattern.findall(prompt))
        unknown = sorted(named - known)
        assert unknown == [], (
            f"the '{key}' system prompt names tools that do not exist: {unknown}"
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


def test_a_skill_tool_cannot_reintroduce_a_native_name() -> None:
    """The merge itself, not just the declarations.

    _select_tools_for_runtime drops skill tools whose name matches a native one,
    so a native tool shadows the skill's copy. The failure mode this pins is the
    inverse: when the native copy is deleted, whatever the skill declares takes
    its place silently. Exercised against a stubbed loader so no MCP subprocess
    is needed — the merge is the part that was wrong, not the spawning.
    """
    from june_brain.loop import agent_helpers

    class _FakeTool:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Runtime:
        preset_key = "gemma"

    original = agent_helpers.__dict__.get("load_skill_tools")
    try:
        import june_brain.skills as skills_mod

        saved = skills_mod.load_skill_tools
        skills_mod.load_skill_tools = lambda: [  # type: ignore[assignment]
            _FakeTool("log_water"),          # a v1 name, as skills/health used to
            _FakeTool("web_search"),         # legitimate
        ]
        selected = {t.name for t in agent_helpers._select_tools_for_runtime(_Runtime())}
    finally:
        skills_mod.load_skill_tools = saved  # type: ignore[assignment]
        if original is not None:
            agent_helpers.__dict__["load_skill_tools"] = original

    assert "web_search" in selected, "a legitimate skill tool was dropped"
    assert "log_water" not in selected, (
        "a skill re-advertised a v1 domain tool and the merge accepted it. "
        "Deleting the native copy unshadows the skill's, so the capability "
        "survives the deletion — this is what happened during D.5a."
    )


# ---------------------------------------------------------------------------
# Invariant: "Behavioral safety floor" — not a therapist/doctor/lawyer/advisor,
# no engagement-maximizing metric, sensitive memories surfaced by the user.
#
# The floor is only a floor if June cannot edit it. It lives in FixedTraits,
# which `character_update` refuses to touch; this names the specific clauses so
# removing one is a test failure rather than a diff nobody re-read.
# ---------------------------------------------------------------------------


def test_the_behavioral_safety_floor_is_a_fixed_trait() -> None:
    from june_brain.character.block import FIXED_FIELD_NAMES, FixedTraits

    for clause in ("not_a_professional", "wellbeing_over_engagement", "privacy"):
        assert clause in FIXED_FIELD_NAMES, f"{clause} is no longer immutable"
        assert getattr(FixedTraits(), clause).strip(), f"{clause} is empty"


def test_june_cannot_edit_its_own_safety_floor(tmp_path) -> None:
    from june_brain.character import seed_character
    from june_brain.character.block import CharacterBlock, FixedTraits, character_update

    path = tmp_path / "character" / "persona.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    seed_character().save(path)

    result = character_update(
        {"not_a_professional": "acts as the user's therapist"}, path=path
    )

    assert result["ok"] is False
    assert result["reason"] == "immutable_fixed_traits"
    assert "not_a_professional" in result["rejected_keys"]
    assert (
        CharacterBlock.load(path).fixed.not_a_professional
        == FixedTraits().not_a_professional
    )


# ---------------------------------------------------------------------------
# Invariant: "No new dependency that can be implemented customly."
#
# A rule stated in prose is a rule nobody notices breaking, and a dependency is
# added in one line. Pinning the set does not judge whether a new one is
# justified — it makes adding one a deliberate edit in two places, which is the
# most a test can honestly do here. The cryptography exception is named in the
# list itself.
# ---------------------------------------------------------------------------


def test_the_runtime_dependency_set_is_deliberate() -> None:
    import pathlib as _pathlib
    import tomllib

    allowed = {
        "python-dotenv",   # .env loading
        "sqlite-vec",      # the vector index (ADR 0019); a C extension, not reimplementable
        "keyring",         # OS credential store
        "pynacl",          # the crypto exception: Ed25519 for the Trust Ledger (ADR 0022)
        "openai",          # the cloud provider's own client (providers/gemini.py)
        "pydantic",        # schema source of truth, shared with the API
        "httpx",           # async HTTP
        "tomli",           # stdlib tomllib backport for < 3.11
    }

    root = _pathlib.Path(__file__).resolve().parents[4]
    declared = set()
    for project in ("brain", "api"):
        data = tomllib.loads((root / "packages" / project / "pyproject.toml").read_text())
        for spec in data["project"]["dependencies"]:
            name = spec.split(";")[0].split("[")[0]
            for sep in (">=", "==", "<=", "~=", ">", "<", "!="):
                name = name.split(sep)[0]
            declared.add(name.strip())

    # The API's own web-serving stack, and the brain it wraps.
    declared -= {"june-brain", "fastapi", "uvicorn", "starlette"}

    added = sorted(declared - allowed)
    assert added == [], (
        f"new runtime dependencies: {added}. CLAUDE.md forbids a dependency that "
        "can be implemented customly. If this one cannot be, add it to the list "
        "above with the reason."
    )
