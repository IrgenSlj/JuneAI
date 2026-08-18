"""The update check — June's only automatic network call (ADR 0031).

Everything else June sends leaves because the user said something. This does
not, and it is the single exception, added for a reason the threat model forced:
a published security posture with no way to deliver a security fix is worth less
than one with a channel.

The six constraints from ADR 0031 are implemented here rather than promised:

1. **Not a timer.** ``maybe_check`` is evaluated at read time, when something
   already happening asks. No thread, no scheduler, no wakeup. The 24-hour
   interval is a floor on frequency, not a schedule — open June once a month and
   it checks once a month.
2. **Local-only blocks it** before a request is built, exactly as it blocks a
   cloud model call.
3. **Ledgered as egress**, so it appears in Receipts next to model calls.
4. **Carries no user data** — an unauthenticated GET to one public endpoint.
   What it unavoidably reveals is IP, User-Agent and timing, which is stated in
   the ADR rather than minimised.
5. **Separately disableable**, so refusing updates does not mean refusing cloud
   models.
6. **Never installs anything.** It reports; the user decides.

Failure is always silent to the user: no network, a rate limit, a malformed
response and a missing config all degrade to "no update information", never to
an error in a turn.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

RELEASES_URL = "https://api.github.com/repos/IrgenSlj/JuneAI/releases/latest"
UPDATE_HOST = "api.github.com"

# The floor between checks. Long on purpose: a security fix that reaches a user
# a day late is fine; a request every hour is a telemetry signal nobody asked to
# emit.
MIN_INTERVAL_SECONDS = 24 * 60 * 60

_TIMEOUT_S = 8.0
_LAST_CHECK_KEY = "update_last_check_at"
_ENABLED_KEY = "update_check_enabled"


@dataclass(frozen=True)
class UpdateStatus:
    """What a check found. ``checked`` is False when no request was made."""

    checked: bool
    reason: str = ""
    latest: str | None = None
    current: str | None = None
    url: str | None = None

    @property
    def update_available(self) -> bool:
        """True only when both versions are known and they differ.

        An unknown current version means "do not claim anything". Guessing here
        would nag every developer build forever, which is how an update prompt
        becomes something users learn to dismiss without reading.
        """
        if not self.latest or not self.current or self.current == "unknown":
            return False
        return self.latest != self.current


_SKIPPED_LOCAL_ONLY = UpdateStatus(False, "local-only mode blocks all egress")
_SKIPPED_DISABLED = UpdateStatus(False, "update checks are turned off")
_SKIPPED_RECENT = UpdateStatus(False, "checked within the last 24 hours")


def is_enabled() -> bool:
    """Whether the update check is on. Defaults to on; the user can turn it off."""
    from .config_store import get_setting

    value = get_setting(_ENABLED_KEY)
    return True if value is None else bool(value)


def set_enabled(enabled: bool) -> None:
    from .config_store import set_setting

    set_setting(_ENABLED_KEY, bool(enabled))


def _last_check_epoch() -> float:
    from .config_store import get_setting

    value = get_setting(_LAST_CHECK_KEY)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _record_check(now: float) -> None:
    from .config_store import set_setting

    set_setting(_LAST_CHECK_KEY, float(now))


def _local_only() -> bool:
    """A config failure must not open the gate — see ``june_brain.privacy``,
    which now owns this rule for every caller."""
    from .privacy import local_only

    return local_only()


def _ledger(outcome: str, latest: str | None) -> None:
    """Record the call as egress. Best-effort, like every other ledger write."""
    try:
        from .trust import get_writer

        get_writer().append(
            kind="egress",
            actor="june",
            payload={
                "kind": "update_check",
                "host": UPDATE_HOST,
                "outcome": outcome,
                "latest": latest,
                "at": datetime.now(UTC).isoformat(),
            },
        )
    except Exception:  # noqa: BLE001
        logger.debug("trust-ledger update-check append failed", exc_info=True)


def maybe_check(*, now: float, force: bool = False) -> UpdateStatus:
    """Check for a newer release if all the gates allow it.

    ``now`` is injected rather than read, so the interval logic is deterministic
    and testable — the same discipline as the temporal block and the Silence
    Model. ``force`` bypasses only the interval, never local-only and never the
    enabled setting: a user-pressed "check now" is still bound by the privacy
    dial they chose.
    """
    if _local_only():
        return _SKIPPED_LOCAL_ONLY
    if not is_enabled():
        return _SKIPPED_DISABLED
    if not force and (now - _last_check_epoch()) < MIN_INTERVAL_SECONDS:
        return _SKIPPED_RECENT

    # Recorded before the request, not after: a hung or failing endpoint must not
    # turn into a retry on every single turn.
    _record_check(now)

    # The *release* version, not the git SHA: comparing a SHA to a release tag
    # is never equal, so it would report an update on every single check.
    from .build_info import release_version

    current = release_version()
    try:
        latest, url = _fetch_latest()
    except Exception as exc:  # noqa: BLE001 - never surfaces as an error in a turn
        logger.debug("update check failed: %s", exc)
        _ledger("failed", None)
        return UpdateStatus(True, f"the check did not complete ({exc})", current=current)

    _ledger("ok", latest)
    return UpdateStatus(True, "", latest=latest, current=current, url=url)


def _fetch_latest() -> tuple[str | None, str | None]:
    """Return (tag, html_url) for the latest release.

    Deliberately not routed through the SSRF guard: the destination is a module
    constant, so there is no caller-supplied URL for anything to redirect. It is
    also deliberately unauthenticated — sending a token would mean having one.
    """
    from .build_info import build_version

    request = urllib.request.Request(
        RELEASES_URL,
        headers={
            "Accept": "application/vnd.github+json",
            # Named honestly. GitHub sees this either way; pretending to be a
            # browser would be a lie in a module about not making silent calls.
            "User-Agent": f"JuneAI/{build_version()} (+https://github.com/IrgenSlj/JuneAI)",
        },
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
        payload = json.loads(response.read().decode("utf-8", "replace"))

    if not isinstance(payload, dict):
        return None, None
    from .build_info import normalize_version

    tag = payload.get("tag_name")
    url = payload.get("html_url")
    return (normalize_version(str(tag)) if tag else None, str(url) if url else None)
