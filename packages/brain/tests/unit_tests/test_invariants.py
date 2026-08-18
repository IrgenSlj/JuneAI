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
# Provider.stream() yields str. A model that accepts an advertised tool returns
# tool_calls with empty content, so nothing streams, nothing dispatches, and the
# user gets a blank reply. Until stream() yields a typed delta (D.4b), the
# streaming path must not advertise tools (D.4a).
# ---------------------------------------------------------------------------


def test_stream_turn_does_not_advertise_tools_it_cannot_receive() -> None:
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
    assert all(t is None for t in seen), (
        f"stream_turn advertised tools on a seam that cannot return them: {seen}"
    )
