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
