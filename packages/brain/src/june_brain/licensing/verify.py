"""Offline Ed25519 license verification.

Flow (numbered to match the design doc):
1. Parse the outer envelope (license body + sig + key_id).
2. Look up the public key by key_id; unknown key -> FREE.
3. Serialize the inner license object to canonical JSON.
4. Verify the Ed25519 signature; bad sig -> FREE.
5. Check schema_version == 1; unknown version -> FREE.
6. Check tier in KNOWN_TIERS; unknown tier -> FREE.
7. Check expires_at vs local UTC now; expired -> FREE with expiry status.
8. Intersect features with KNOWN_FEATURES so no unknown key can grant anything.
9. Return a non-FREE Entitlement.

NEVER raises.  Every failure path returns a FREE Entitlement with an honest
status string.  No network I/O, no socket, ever.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from .keys import PUBLIC_KEYS
from .models import (
    KNOWN_FEATURES,
    KNOWN_TIERS,
    TIER_FREE,
    Entitlement,
    LicenseBody,
)


def _nacl() -> Any:  # type: ignore[return]
    """Import PyNaCl lazily; return the nacl package or None if absent."""
    try:
        import nacl
        import nacl.encoding  # noqa: F401
        import nacl.exceptions  # noqa: F401
        import nacl.signing  # noqa: F401

        return nacl
    except ImportError:
        return None


def _canonical_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace, UTF-8 safe.

    Local copy of the house convention (trust/ledger.py) so this module
    stays independent of ledger internals.

    LOAD-BEARING CONVENTION: the license signing tool MUST serialize the body
    with these exact options — ``sort_keys=True``, ``separators=(",", ":")``,
    and ``ensure_ascii=False`` — or the signature will not verify byte-for-byte
    and a legitimately paid holder is locked out.  ``ensure_ascii=False`` means
    non-ASCII holder names (e.g. accented characters) are emitted as raw UTF-8,
    not ``\\uXXXX`` escapes; the signer must match.  This convention is the
    single source of truth; keep it and license-design.md in lockstep.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _parse_iso_utc(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp to a timezone-aware UTC datetime.

    Returns None on any malformed input (never raises).  A *naive* timestamp
    (no offset, e.g. ``"2027-01-01T00:00:00"``) is interpreted as UTC rather
    than rejected: our signing tool always emits a ``Z`` suffix, but a license
    that omits it must still grant the tier the holder paid for, not silently
    lock them out.  This normalization is load-bearing — comparing a naive
    datetime against an aware one raises ``TypeError`` (not ``ValueError``),
    which would otherwise escape the narrow except below and degrade a paid
    holder to FREE.
    """
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def _free(status: str) -> Entitlement:
    """Return a FREE Entitlement with the given honest status string."""
    return Entitlement(
        tier=TIER_FREE,
        features=frozenset(),
        holder="",
        valid_until=None,
        status=status,
    )


def entitlement_from_raw(
    raw: Any,
    *,
    public_keys: dict[str, str] = PUBLIC_KEYS,
    now: datetime | None = None,
) -> Entitlement:
    """Parse a raw license envelope dict and return an Entitlement.

    Every failure path returns FREE with an honest status.  Never raises.

    Args:
        raw: The parsed JSON envelope (dict expected).
        public_keys: Map of key_id -> public hex to verify against.
            Defaults to the shipped PUBLIC_KEYS (empty pre-launch).
        now: Override for the current UTC time (for testing expiry).
    """
    try:
        return _verify(raw, public_keys=public_keys, now=now)
    except Exception:  # noqa: BLE001 - absolute last resort; should never reach here
        return _free("free — license verification encountered an unexpected error")


def _verify(
    raw: Any,
    *,
    public_keys: dict[str, str],
    now: datetime | None,
) -> Entitlement:
    # Step 1: parse envelope.
    if not isinstance(raw, dict):
        return _free("free — license file is not a valid JSON object")

    license_dict = raw.get("license")
    sig_hex = raw.get("sig")
    key_id = raw.get("key_id")

    if not isinstance(license_dict, dict):
        return _free("free — license envelope missing 'license' object")
    if not isinstance(sig_hex, str):
        return _free("free — license envelope missing 'sig'")
    if not isinstance(key_id, str):
        return _free("free — license envelope missing 'key_id'")

    # Step 2: look up public key.
    pub_hex = public_keys.get(key_id)
    if not pub_hex:
        return _free(f"free — unknown license key_id '{key_id}'")

    # Step 3: canonical JSON of the inner body.
    message_bytes = _canonical_json(license_dict).encode("utf-8")

    # Step 4: verify signature (requires PyNaCl).
    nacl = _nacl()
    if nacl is None:
        return _free("free — license crypto library unavailable")

    try:
        from nacl.encoding import HexEncoder
        from nacl.exceptions import BadSignatureError
        from nacl.signing import VerifyKey

        vk = VerifyKey(pub_hex.encode("ascii"), encoder=HexEncoder)
        try:
            vk.verify(message_bytes, bytes.fromhex(sig_hex))
        except BadSignatureError:
            return _free("free — license signature did not verify")
    except Exception:  # noqa: BLE001 - malformed key/sig must not crash
        return _free("free — license signature could not be checked")

    # Step 5: parse body and check schema_version.
    body = LicenseBody.from_dict(license_dict)
    if body is None:
        return _free("free — license body has unexpected shape")
    if body.schema_version != 1:
        return _free(f"free — unknown license schema version {body.schema_version}")

    # Step 6: check tier.
    if body.tier not in KNOWN_TIERS or body.tier == TIER_FREE:
        return _free(f"free — unknown license tier '{body.tier}'")

    # Step 7: check expiry.
    utc_now = now if now is not None else datetime.now(UTC)
    if body.expires_at is not None:
        expires = _parse_iso_utc(body.expires_at)
        if expires is None:
            return _free("free — license has an invalid expires_at value")
        if utc_now >= expires:
            exp_date = expires.strftime("%Y-%m-%d")
            return _free(f"free — license expired on {exp_date}")
        valid_until: str | None = body.expires_at
    else:
        # null expires_at means lifetime/founder.
        valid_until = None

    # Step 8: intersect features with known set.
    safe_features = frozenset(body.features) & KNOWN_FEATURES

    # Step 9: build status string and return.
    if valid_until is not None:
        exp_dt = _parse_iso_utc(valid_until)
        exp_str = exp_dt.strftime("%Y-%m-%d") if exp_dt is not None else valid_until
        status = f"{body.tier} — licensed to {body.holder}, valid until {exp_str}"
    else:
        status = f"{body.tier} — lifetime, licensed to {body.holder}"

    return Entitlement(
        tier=body.tier,
        features=safe_features,
        holder=body.holder,
        valid_until=valid_until,
        status=status,
    )
