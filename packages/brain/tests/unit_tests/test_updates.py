"""The update check — June's only automatic network call (ADR 0031).

Every constraint the ADR states is a test here. That is the whole point of the
ADR: an automatic outbound call is a privacy-boundary change, and a boundary
that is documented but unverified has moved without anyone being able to tell.

No test in this file touches the network.
"""

from __future__ import annotations

from typing import Any

import pytest
from june_brain import updates

NOW = 1_800_000_000.0
DAY = 24 * 60 * 60


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Point config.json and the ledger at a temp dir for every test."""
    monkeypatch.setenv("JUNE_DATA_DIR", str(tmp_path))
    import june_brain.config as config
    import june_brain.config_store as config_store

    monkeypatch.setattr(config, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(config_store, "config_path", lambda: tmp_path / "config.json")
    yield


@pytest.fixture
def calls(monkeypatch) -> list[str]:
    """Record every fetch that would have gone out, and answer without a network."""
    made: list[str] = []

    def _fake_fetch() -> tuple[str | None, str | None]:
        made.append(updates.RELEASES_URL)
        return "v0.9.9", "https://example.com/releases/v0.9.9"

    monkeypatch.setattr(updates, "_fetch_latest", _fake_fetch)
    monkeypatch.setattr(updates, "_local_only", lambda: False)
    return made


@pytest.fixture
def ledgered(monkeypatch) -> list[tuple[str, str | None]]:
    entries: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        updates, "_ledger", lambda outcome, latest: entries.append((outcome, latest))
    )
    return entries


# -- ADR 0031 §2: local-only blocks it ----------------------------------


def test_local_only_blocks_the_check_before_a_request_is_built(monkeypatch) -> None:
    """The strongest promise June makes. An automatic call cannot be an exception."""
    made: list[str] = []
    monkeypatch.setattr(updates, "_local_only", lambda: True)
    monkeypatch.setattr(updates, "_fetch_latest", lambda: made.append("x") or (None, None))

    status = updates.maybe_check(now=NOW)
    assert status.checked is False
    assert "local-only" in status.reason
    assert made == [], "a request was built despite local-only"


def test_force_does_not_override_local_only(monkeypatch) -> None:
    """A user pressing 'check now' is still bound by the dial they chose."""
    monkeypatch.setattr(updates, "_local_only", lambda: True)
    assert updates.maybe_check(now=NOW, force=True).checked is False


def test_a_broken_privacy_config_fails_closed(monkeypatch) -> None:
    """If the dial cannot be read, assume local-only rather than assume egress."""
    import june_brain.config_store as config_store

    def boom() -> Any:
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(config_store, "get_privacy_dial", boom)
    assert updates._local_only() is True


# -- ADR 0031 §1: not a timer, and rate limited -------------------------


def test_two_checks_inside_a_day_make_one_request(calls, ledgered) -> None:
    updates.maybe_check(now=NOW)
    updates.maybe_check(now=NOW + 60)
    updates.maybe_check(now=NOW + DAY - 1)
    assert len(calls) == 1


def test_a_check_a_day_later_is_allowed(calls, ledgered) -> None:
    updates.maybe_check(now=NOW)
    updates.maybe_check(now=NOW + DAY + 1)
    assert len(calls) == 2


def test_force_bypasses_only_the_interval(calls, ledgered) -> None:
    updates.maybe_check(now=NOW)
    updates.maybe_check(now=NOW + 5, force=True)
    assert len(calls) == 2


def test_the_interval_survives_a_restart(calls, ledgered) -> None:
    """Persisted in config.json, so relaunching does not reset the floor."""
    updates.maybe_check(now=NOW)
    assert updates._last_check_epoch() == NOW
    updates.maybe_check(now=NOW + 10)
    assert len(calls) == 1


def test_a_failing_endpoint_does_not_retry_every_turn(monkeypatch, ledgered) -> None:
    """The timestamp is recorded before the request, not after."""
    attempts: list[int] = []

    def _boom() -> Any:
        attempts.append(1)
        raise OSError("network down")

    monkeypatch.setattr(updates, "_local_only", lambda: False)
    monkeypatch.setattr(updates, "_fetch_latest", _boom)

    updates.maybe_check(now=NOW)
    updates.maybe_check(now=NOW + 60)
    updates.maybe_check(now=NOW + 120)
    assert len(attempts) == 1


# -- ADR 0031 §5: separately disableable --------------------------------


def test_the_check_is_on_by_default(calls, ledgered) -> None:
    assert updates.is_enabled() is True
    assert updates.maybe_check(now=NOW).checked is True


def test_turning_it_off_stops_it(calls, ledgered) -> None:
    updates.set_enabled(False)
    status = updates.maybe_check(now=NOW)
    assert status.checked is False
    assert "turned off" in status.reason
    assert calls == []


def test_turning_it_back_on_works(calls, ledgered) -> None:
    updates.set_enabled(False)
    updates.set_enabled(True)
    assert updates.maybe_check(now=NOW).checked is True


# -- ADR 0031 §3: ledgered as egress ------------------------------------


def test_a_successful_check_is_ledgered(calls, ledgered) -> None:
    updates.maybe_check(now=NOW)
    assert ledgered == [("ok", "v0.9.9")]


def test_a_failed_check_is_also_ledgered(monkeypatch, ledgered) -> None:
    """A call that left the machine and failed still left the machine."""
    monkeypatch.setattr(updates, "_local_only", lambda: False)
    monkeypatch.setattr(updates, "_fetch_latest", lambda: (_ for _ in ()).throw(OSError("x")))
    updates.maybe_check(now=NOW)
    assert ledgered == [("failed", None)]


def test_a_blocked_check_is_not_ledgered_as_egress(monkeypatch, ledgered) -> None:
    """Nothing left the machine, so nothing belongs in the egress log."""
    monkeypatch.setattr(updates, "_local_only", lambda: True)
    updates.maybe_check(now=NOW)
    assert ledgered == []


# -- ADR 0031 §6 and failure behaviour ----------------------------------


def test_the_result_reports_and_never_installs(calls, ledgered) -> None:
    status = updates.maybe_check(now=NOW)
    assert status.latest == "v0.9.9"
    assert status.url == "https://example.com/releases/v0.9.9"
    # The surface is data. There is no install/apply/download entry point.
    assert not [n for n in dir(updates) if n.startswith(("install", "apply", "download"))]


def test_failure_degrades_to_no_information_not_an_error(monkeypatch, ledgered) -> None:
    monkeypatch.setattr(updates, "_local_only", lambda: False)
    monkeypatch.setattr(updates, "_fetch_latest", lambda: (_ for _ in ()).throw(OSError("dns")))
    status = updates.maybe_check(now=NOW)
    assert status.latest is None
    assert status.update_available is False


def test_a_malformed_response_does_not_claim_an_update(monkeypatch, ledgered) -> None:
    monkeypatch.setattr(updates, "_local_only", lambda: False)
    monkeypatch.setattr(updates, "_fetch_latest", lambda: (None, None))
    status = updates.maybe_check(now=NOW)
    assert status.update_available is False


def test_matching_versions_are_not_an_update(monkeypatch, ledgered) -> None:
    monkeypatch.setattr(updates, "_local_only", lambda: False)
    monkeypatch.setattr(updates, "_fetch_latest", lambda: ("1.2.3", "u"))
    monkeypatch.setattr("june_brain.build_info.release_version", lambda: "1.2.3")
    assert updates.maybe_check(now=NOW).update_available is False


# -- ADR 0031 §4: what leaves -------------------------------------------


def test_the_endpoint_is_a_constant_with_no_user_data() -> None:
    """No identifiers, no query string, no memory — one public URL."""
    assert updates.RELEASES_URL == (
        "https://api.github.com/repos/IrgenSlj/JuneAI/releases/latest"
    )
    assert "?" not in updates.RELEASES_URL


# -- the bug the live endpoint found ------------------------------------


def test_a_git_sha_is_never_compared_against_a_release_tag(monkeypatch, ledgered) -> None:
    """The first live run reported an update against its own newest build.

    `build_version()` is a git SHA and `tag_name` is a release tag, so
    `latest != current` was true forever. Every user would have been told an
    update existed, permanently — the surest way to train people to dismiss an
    update prompt without reading it. Caught only by calling the real endpoint.
    """
    monkeypatch.setattr(updates, "_local_only", lambda: False)
    monkeypatch.setattr(updates, "_fetch_latest", lambda: ("0.1.0", "u"))
    monkeypatch.setattr("june_brain.build_info.release_version", lambda: "0.1.0")

    status = updates.maybe_check(now=NOW)
    assert status.current == "0.1.0", "the SHA leaked back into the comparison"
    assert status.update_available is False


def test_the_v_prefix_does_not_create_a_phantom_update(monkeypatch, ledgered) -> None:
    """GitHub tags are `v0.1.0`; package metadata is `0.1.0`."""
    from june_brain.build_info import normalize_version

    assert normalize_version("v0.1.0") == "0.1.0"
    assert normalize_version("0.1.0") == "0.1.0"
    assert normalize_version("V2.0") == "2.0"


def test_an_unknown_current_version_claims_nothing() -> None:
    """A dev build with no package metadata must not nag."""
    status = updates.UpdateStatus(True, latest="9.9.9", current="unknown")
    assert status.update_available is False
