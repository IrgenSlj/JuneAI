"""Destination defence for outbound fetches — SSRF (ADR 0021, Phase 7.1).

The rest of the guard defends the *content* that comes back: framed as
untrusted, scanned for injection, taint-tracked, gated. None of that says
anything about **where June was pointed**, and until this module existed the
answer was "anywhere". `fetch_url` and `read_webpage` accepted any http(s) URL
and followed redirects blindly, so June could be aimed at the user's router
admin page, at Ollama on 11434, at any local development service, at cloud
metadata on 169.254.169.254 — or at a perfectly ordinary public URL that
redirects to one of those.

That matters more here than in most applications. June's whole pitch is that
consequential actions are visible and gated; a fetch of `http://192.168.1.1/`
looked like an ordinary `read_network` and told the user nothing.

## What this does

`check_url` resolves the hostname and rejects the request when *any* resolved
address is private, loopback, link-local, multicast, reserved, or unspecified.
`fetch_guarded` then follows redirects **one hop at a time**, re-checking every
hop, because a public URL that 302s to `http://127.0.0.1:8000/` is the same
attack with an extra step.

## What this does not do

Stated here rather than in a footnote, because the residual is real:

- **DNS rebinding is not closed.** We resolve, validate, and then hand the URL
  to httpx, which resolves again. A hostile resolver can answer differently the
  second time. Closing it properly means pinning the connection to the validated
  address and carrying the Host header ourselves, which is a bigger change to
  every call site. The window is small and the attack is loud; it is recorded in
  the threat model rather than papered over.
- **It does not stop a user deliberately fetching their own router.** The block
  is unconditional. If that turns out to be a real workflow it needs a user-
  visible allow-list, not a silent exception.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

# Redirect chains longer than this are hostile or broken; either way, stop.
MAX_REDIRECTS = 5

ALLOWED_SCHEMES = frozenset({"http", "https"})

# Networks with no legitimate reason to appear in a fetch June makes on the
# user's behalf. `ipaddress` already classifies most of these; the explicit
# extras are the ones its properties miss or under-cover.
_EXTRA_BLOCKED = (
    ipaddress.ip_network("100.64.0.0/10"),  # CGNAT — carrier-side, not the web
    ipaddress.ip_network("192.0.0.0/24"),  # IETF protocol assignments
    ipaddress.ip_network("198.18.0.0/15"),  # benchmarking
    ipaddress.ip_network("::ffff:0:0/96"),  # IPv4-mapped IPv6, checked unmapped
)


@dataclass(frozen=True)
class UrlVerdict:
    """The result of checking one URL. ``reason`` is empty when allowed."""

    allowed: bool
    reason: str
    resolved: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.allowed


def _classify_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    """Return a human reason when this address must not be fetched, else ""."""
    # IPv4-mapped IPv6 (::ffff:127.0.0.1) would otherwise dodge every IPv4 check.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return _classify_ip(ip.ipv4_mapped)

    if ip.is_loopback:
        return "a loopback address"
    if ip.is_link_local:
        # 169.254.169.254 lives here: the cloud metadata endpoint.
        return "a link-local address"
    if ip.is_private:
        return "a private network address"
    if ip.is_multicast:
        return "a multicast address"
    if ip.is_reserved:
        return "a reserved address"
    if ip.is_unspecified:
        return "an unspecified address"
    for net in _EXTRA_BLOCKED:
        if ip in net:
            return f"inside the blocked range {net}"
    return ""


def _resolve(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Every address a hostname resolves to, or the literal if it is one.

    All of them are checked, not just the first: a host that resolves to one
    public and one private address is a bypass, not a coincidence.
    """
    literal = host.strip("[]")
    try:
        return [ipaddress.ip_address(literal)]
    except ValueError:
        pass

    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    out: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        addr = info[4][0]
        try:
            parsed = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if parsed not in out:
            out.append(parsed)
    return out


def check_url(url: str) -> UrlVerdict:
    """Whether June may fetch ``url``. Pure apart from a DNS lookup.

    Rejects on the first unsafe address, and names it in terms the user can act
    on — the message reaches them through the tool result.
    """
    if not url or not url.strip():
        return UrlVerdict(False, "the URL is empty")

    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        return UrlVerdict(False, f"the scheme '{parts.scheme}' is not http or https")
    if not parts.hostname:
        return UrlVerdict(False, "the URL has no host")

    try:
        addresses = _resolve(parts.hostname)
    except OSError as exc:
        return UrlVerdict(False, f"the host could not be resolved ({exc})")
    if not addresses:
        return UrlVerdict(False, "the host resolved to no addresses")

    for ip in addresses:
        reason = _classify_ip(ip)
        if reason:
            return UrlVerdict(
                False,
                f"{parts.hostname} resolves to {ip}, which is {reason}. "
                "June does not fetch addresses on your own machine or local network.",
                tuple(str(a) for a in addresses),
            )

    return UrlVerdict(True, "", tuple(str(a) for a in addresses))


def fetch_guarded(
    client: object,
    url: str,
    *,
    timeout: float = 15.0,
    headers: dict[str, str] | None = None,
    max_bytes: int = 5_000_000,
) -> object:
    """Fetch ``url``, checking every redirect hop. Raises ``SsrfBlocked``.

    ``client`` is anything with httpx's ``get`` signature, so this stays
    testable without a network and without httpx in the guard's imports.

    Redirects are followed manually because ``follow_redirects=True`` performs
    the whole chain before returning — by which point a redirect to localhost has
    already been requested, and checking afterwards is checking the wrong thing.
    """
    current = url
    for _hop in range(MAX_REDIRECTS + 1):
        verdict = check_url(current)
        if not verdict.allowed:
            raise SsrfBlocked(verdict.reason)

        response = client.get(  # type: ignore[attr-defined]
            current,
            follow_redirects=False,
            timeout=timeout,
            headers=headers or {},
        )

        status = getattr(response, "status_code", 200)
        location = (getattr(response, "headers", {}) or {}).get("location")
        if status in (301, 302, 303, 307, 308) and location:
            current = _absolutise(current, location)
            continue

        _guard_size(response, max_bytes)
        return response

    raise SsrfBlocked(f"more than {MAX_REDIRECTS} redirects")


def _absolutise(base: str, location: str) -> str:
    from urllib.parse import urljoin

    return urljoin(base, location)


def _guard_size(response: object, max_bytes: int) -> None:
    """Reject a response too large to be a page June should read.

    Checked on the declared length; the fetchers truncate the decoded text
    anyway, so this is about not buying a multi-gigabyte body at all.
    """
    headers = getattr(response, "headers", {}) or {}
    declared = headers.get("content-length")
    if declared:
        try:
            if int(declared) > max_bytes:
                raise SsrfBlocked(f"the response is larger than {max_bytes} bytes")
        except ValueError:
            pass


class SsrfBlocked(Exception):
    """Raised when a fetch is refused. The message is shown to the user."""
