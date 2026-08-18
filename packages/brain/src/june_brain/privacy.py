"""The one implementation of "may this call leave the machine?".

Before this module the question was answered in three places — the loop's tool
partition, the provider egress chokepoint, and the update check — and the three
disagreed about what to do when the dial could not be read. Two returned "not
local-only" (egress proceeds); one returned "local-only" with the comment "a
config failure must not open the gate". The comment was right and the majority
was wrong, which is the failure mode a duplicated invariant produces: it does
not drift where anyone is looking.

So there is one function, it fails closed, and `check.sh` fails the build if a
second implementation appears (`get_privacy_dial` may only be imported here and
by the settings routes that let the user read and set the value).

Fails closed means: if we cannot prove egress is permitted, it is not permitted.
The cost of a false "blocked" is a turn that degrades and says so. The cost of a
false "permitted" is data leaving a machine whose owner set the dial to stop
exactly that, and a provenance frame that does not mention it. Those are not
comparable, so they do not get the same default.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def local_only() -> bool:
    """True when the user's privacy dial is Local-only, or cannot be read.

    The "or cannot be read" is the whole point — see the module docstring.
    """
    try:
        from .config_store import get_privacy_dial
        from .routing import UserPrivacyDial

        return get_privacy_dial() == UserPrivacyDial.LOCAL_ONLY
    except Exception:
        # Not `logger.exception` at error level on a hot path, but never silent:
        # a dial we cannot read is a fact worth having in the log when someone
        # asks why a turn degraded.
        logger.warning(
            "privacy dial unreadable; treating as Local-only (failing closed)",
            exc_info=True,
        )
        return True


def egress_permitted() -> bool:
    """True when a call is allowed to leave the machine.

    The inverse of :func:`local_only`, named for the question the caller is
    actually asking. Prefer it at call sites that gate an outbound action, so
    the code reads as the permission it needs rather than the state it infers.
    """
    return not local_only()


def dial_value() -> str:
    """The active dial as a string, for ledger and provenance payloads.

    Returns ``"unknown"`` rather than failing closed: this value is *recorded*,
    never used to decide anything, and writing "local_only" into the ledger when
    we could not read the dial would put a false statement in the audit trail.
    Deciding and describing have different failure modes, so they get different
    functions.
    """
    try:
        from .config_store import get_privacy_dial

        return str(get_privacy_dial().value)
    except Exception:
        logger.warning("privacy dial unreadable; recording as unknown", exc_info=True)
        return "unknown"
