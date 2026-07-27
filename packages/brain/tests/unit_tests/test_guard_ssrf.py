"""SSRF defence on outbound fetches (ADR 0021, Phase 7.1).

The bypass cases are the point of this file. Blocking `http://127.0.0.1/` is
trivial and proves nothing; every real SSRF filter has been defeated by an
alternate encoding of the same address, by a hostname that resolves to it, or by
a redirect that arrives at it. Each case here is a documented technique.
"""

from __future__ import annotations

from typing import Any

import pytest
from june_brain.guard.ssrf import (
    MAX_REDIRECTS,
    SsrfBlocked,
    check_url,
    fetch_guarded,
)

# -- addresses that must never be fetched -------------------------------

BLOCKED = [
    ("http://127.0.0.1:8000/memory", "loopback, June's own API"),
    ("http://localhost:11434/api/tags", "loopback by name — Ollama"),
    ("http://192.168.1.1/admin", "the user's router"),
    ("http://10.0.0.5/", "private class A"),
    ("http://172.16.4.4/", "private class B"),
    ("http://169.254.169.254/latest/meta-data/", "cloud metadata"),
    ("http://[::1]:8000/", "IPv6 loopback"),
    ("http://[fc00::1]/", "IPv6 unique-local"),
    ("http://[::ffff:127.0.0.1]/", "IPv4-mapped IPv6 loopback"),
    ("http://0.0.0.0:8000/", "unspecified, routes to localhost on Linux"),
    ("http://100.64.0.1/", "CGNAT space"),
    ("http://224.0.0.1/", "multicast"),
]


@pytest.mark.parametrize("url,why", BLOCKED, ids=[u for u, _ in BLOCKED])
def test_internal_addresses_are_refused(url: str, why: str) -> None:
    verdict = check_url(url)
    assert not verdict.allowed, f"{url} was allowed but is {why}"
    assert verdict.reason, "a refusal with no reason is unusable to the user"


@pytest.mark.parametrize(
    "url",
    [
        "http://2130706433/",  # decimal 127.0.0.1
        "http://0x7f000001/",  # hex
        "http://017700000001/",  # octal
    ],
    ids=["decimal", "hex", "octal"],
)
def test_alternate_encodings_of_loopback_are_refused(url: str) -> None:
    """The classic filter bypass: same address, different notation.

    These reach the resolver as hostnames rather than IP literals, so what
    stops them is resolving before judging rather than pattern-matching text.
    """
    assert not check_url(url).allowed


def test_a_hostname_that_resolves_to_loopback_is_refused(monkeypatch) -> None:
    """DNS is the bypass a text filter cannot see."""
    import june_brain.guard.ssrf as ssrf

    monkeypatch.setattr(
        ssrf.socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 0))],
    )
    verdict = check_url("http://totally-normal.example.com/page")
    assert not verdict.allowed
    assert "127.0.0.1" in verdict.reason


def test_one_bad_address_among_several_is_enough(monkeypatch) -> None:
    """A host answering with one public and one private address is an attack."""
    import june_brain.guard.ssrf as ssrf

    monkeypatch.setattr(
        ssrf.socket,
        "getaddrinfo",
        lambda *a, **k: [
            (2, 1, 6, "", ("93.184.216.34", 0)),
            (2, 1, 6, "", ("10.1.2.3", 0)),
        ],
    )
    assert not check_url("http://split-horizon.example.com/").allowed


# -- addresses that must still work -------------------------------------


def test_ordinary_public_urls_are_allowed(monkeypatch) -> None:
    import june_brain.guard.ssrf as ssrf

    monkeypatch.setattr(
        ssrf.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))]
    )
    for url in ("https://example.com/", "http://example.com/a?b=c"):
        assert check_url(url).allowed, url


def test_a_public_ip_literal_is_allowed() -> None:
    assert check_url("https://93.184.216.34/").allowed


# -- malformed input ----------------------------------------------------


@pytest.mark.parametrize(
    "url",
    ["", "   ", "file:///etc/passwd", "ftp://example.com/", "gopher://x/", "http://"],
    ids=["empty", "blank", "file", "ftp", "gopher", "no-host"],
)
def test_non_web_schemes_and_junk_are_refused(url: str) -> None:
    assert not check_url(url).allowed


def test_an_unresolvable_host_is_refused_not_crashed(monkeypatch) -> None:
    import june_brain.guard.ssrf as ssrf

    def boom(*a: Any, **k: Any) -> Any:
        raise OSError("Name or service not known")

    monkeypatch.setattr(ssrf.socket, "getaddrinfo", boom)
    verdict = check_url("http://nope.invalid/")
    assert not verdict.allowed
    assert "resolved" in verdict.reason


# -- redirects ----------------------------------------------------------


class _Response:
    def __init__(self, status: int = 200, location: str | None = None, length: str | None = None):
        self.status_code = status
        self.headers: dict[str, str] = {}
        if location:
            self.headers["location"] = location
        if length:
            self.headers["content-length"] = length
        self.text = "body"


class _Client:
    """Records what was actually requested, so we can assert on the last hop."""

    def __init__(self, responses: list[_Response]):
        self._responses = responses
        self.requested: list[str] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.requested.append(url)
        assert kwargs.get("follow_redirects") is False, (
            "the client must not follow redirects itself — every hop is checked"
        )
        return self._responses[len(self.requested) - 1]


def test_a_redirect_to_loopback_is_blocked(monkeypatch) -> None:
    """The whole reason redirects are followed by hand.

    `follow_redirects=True` completes the chain before returning, so the request
    to localhost has already happened by the time anything could check it.
    """
    import june_brain.guard.ssrf as ssrf

    monkeypatch.setattr(
        ssrf.socket, "getaddrinfo", lambda host, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))]
    )
    client = _Client([_Response(302, location="http://127.0.0.1:8000/secrets")])

    with pytest.raises(SsrfBlocked, match="loopback"):
        fetch_guarded(client, "https://example.com/start")

    # The public first hop was fetched; the loopback second hop never was.
    assert client.requested == ["https://example.com/start"]


def test_a_safe_redirect_chain_is_followed(monkeypatch) -> None:
    import june_brain.guard.ssrf as ssrf

    monkeypatch.setattr(
        ssrf.socket, "getaddrinfo", lambda host, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))]
    )
    client = _Client(
        [
            _Response(301, location="https://example.com/b"),
            _Response(200),
        ]
    )
    response = fetch_guarded(client, "https://example.com/a")
    assert response.status_code == 200
    assert client.requested == ["https://example.com/a", "https://example.com/b"]


def test_a_relative_redirect_resolves_against_the_current_url(monkeypatch) -> None:
    import june_brain.guard.ssrf as ssrf

    monkeypatch.setattr(
        ssrf.socket, "getaddrinfo", lambda host, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))]
    )
    client = _Client([_Response(302, location="/moved"), _Response(200)])
    fetch_guarded(client, "https://example.com/deep/page")
    assert client.requested[-1] == "https://example.com/moved"


def test_a_redirect_loop_terminates(monkeypatch) -> None:
    import june_brain.guard.ssrf as ssrf

    monkeypatch.setattr(
        ssrf.socket, "getaddrinfo", lambda host, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))]
    )
    client = _Client([_Response(302, location="https://example.com/loop")] * 20)
    with pytest.raises(SsrfBlocked, match="redirect"):
        fetch_guarded(client, "https://example.com/loop")
    assert len(client.requested) <= MAX_REDIRECTS + 1


def test_an_oversized_response_is_refused(monkeypatch) -> None:
    import june_brain.guard.ssrf as ssrf

    monkeypatch.setattr(
        ssrf.socket, "getaddrinfo", lambda host, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))]
    )
    client = _Client([_Response(200, length="999999999")])
    with pytest.raises(SsrfBlocked, match="larger than"):
        fetch_guarded(client, "https://example.com/huge")


def test_the_refusal_message_is_written_for_the_user() -> None:
    """It reaches the user through the tool result, so it has to read like prose."""
    reason = check_url("http://192.168.1.1/admin").reason
    assert "192.168.1.1" in reason
    assert "local network" in reason
