"""Tests for the offline license / entitlement system (Slice 1).

Uses pytest.importorskip("nacl") for signing tests so the suite stays
green on environments without PyNaCl (though PyNaCl is a declared
dependency of packages/brain and should always be present).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from june_brain.licensing import (
    FEATURE_BACKUP_SYNC,
    FEATURE_CLOUD_RELAY,
    FEATURE_GOOGLE_SKILLS,
    FEATURE_SUPPORTER,
    FREE,
    KNOWN_FEATURES,
    KNOWN_TIERS,
    TIER_FOUNDER,
    TIER_FREE,
    TIER_PRO,
    Entitlement,
    entitlement_from_raw,
    get_entitlement,
    reset_for_tests,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _future(days: int = 365) -> str:
    return (datetime.now(UTC) + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _past(days: int = 1) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _make_body(
    tier: str = TIER_PRO,
    expires_at: str | None = None,
    features: list[str] | None = None,
    schema_version: int = 1,
    holder: str = "Test User <test@example.com>",
) -> dict[str, Any]:
    if expires_at is None:
        expires_at = _future()
    if features is None:
        features = [FEATURE_BACKUP_SYNC, FEATURE_CLOUD_RELAY, FEATURE_GOOGLE_SKILLS, FEATURE_SUPPORTER]
    return {
        "schema_version": schema_version,
        "license_id": "JUNE-TEST-000001",
        "tier": tier,
        "holder": holder,
        "seats": 1,
        "issued_at": "2026-07-01T00:00:00Z",
        "expires_at": expires_at,
        "features": features,
    }


def _make_envelope(
    body: dict[str, Any],
    sk_hex: str,
    key_id: str = "dev-1",
) -> dict[str, Any]:
    """Sign a license body with the given private key and wrap in the envelope."""
    from nacl.encoding import HexEncoder
    from nacl.signing import SigningKey

    sk = SigningKey(sk_hex.encode("ascii"), encoder=HexEncoder)
    message = _canonical_json(body).encode("utf-8")
    signed = sk.sign(message)
    sig_hex = signed.signature.hex()
    return {"license": body, "key_id": key_id, "sig": sig_hex}


# ---------------------------------------------------------------------------
# Fixture: ephemeral keypair (requires nacl)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def keypair():
    """Generate an ephemeral Ed25519 keypair for signing tests."""
    pytest.importorskip("nacl")
    from nacl.encoding import HexEncoder
    from nacl.signing import SigningKey

    sk = SigningKey.generate()
    sk_hex = sk.encode(encoder=HexEncoder).decode("ascii")
    pk_hex = sk.verify_key.encode(encoder=HexEncoder).decode("ascii")
    return sk_hex, pk_hex


# ---------------------------------------------------------------------------
# Static / no-nacl tests (always run)
# ---------------------------------------------------------------------------


def test_free_tier_is_free() -> None:
    assert FREE.tier == TIER_FREE


def test_free_has_no_features() -> None:
    assert FREE.features == frozenset()


def test_free_has_returns_false() -> None:
    assert FREE.has(FEATURE_BACKUP_SYNC) is False
    assert FREE.has(FEATURE_CLOUD_RELAY) is False
    assert FREE.has(FEATURE_GOOGLE_SKILLS) is False
    assert FREE.has(FEATURE_SUPPORTER) is False
    assert FREE.has("nonexistent") is False


def test_known_tiers_contains_expected() -> None:
    assert TIER_FREE in KNOWN_TIERS
    assert TIER_PRO in KNOWN_TIERS
    assert TIER_FOUNDER in KNOWN_TIERS


def test_known_features_contains_expected() -> None:
    for f in (FEATURE_BACKUP_SYNC, FEATURE_CLOUD_RELAY, FEATURE_GOOGLE_SKILLS, FEATURE_SUPPORTER, "commercial_use"):
        assert f in KNOWN_FEATURES


def test_missing_file_returns_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No license file -> exactly FREE."""
    monkeypatch.setenv("JUNE_LICENSE_PATH", str(tmp_path / "nonexistent.json"))
    reset_for_tests()
    ent = get_entitlement()
    assert ent.tier == TIER_FREE
    assert ent.features == frozenset()


def test_nacl_absent_returns_free(monkeypatch: pytest.MonkeyPatch) -> None:
    """When PyNaCl is monkeypatched away, verification returns FREE."""
    import june_brain.licensing.verify as verify_mod

    monkeypatch.setattr(verify_mod, "_nacl", lambda: None)
    body = _make_body()
    envelope = {"license": body, "key_id": "any-key", "sig": "deadbeef"}
    ent = entitlement_from_raw(envelope, public_keys={"any-key": "aabbcc"})
    assert ent.tier == TIER_FREE
    assert "unavailable" in ent.status


def test_empty_public_keys_returns_free() -> None:
    """PUBLIC_KEYS is empty pre-launch; every license degrades to FREE."""
    body = _make_body()
    envelope = {"license": body, "key_id": "prod-1", "sig": "deadbeef"}
    ent = entitlement_from_raw(envelope, public_keys={})
    assert ent.tier == TIER_FREE


def test_unknown_key_id_returns_free() -> None:
    body = _make_body()
    envelope = {"license": body, "key_id": "unknown-key", "sig": "deadbeef"}
    ent = entitlement_from_raw(envelope, public_keys={"prod-1": "aabbcc"})
    assert ent.tier == TIER_FREE
    assert "unknown" in ent.status


def test_malformed_envelope_returns_free() -> None:
    ent = entitlement_from_raw("not a dict")
    assert ent.tier == TIER_FREE


def test_missing_license_key_returns_free() -> None:
    ent = entitlement_from_raw({"key_id": "k", "sig": "s"})
    assert ent.tier == TIER_FREE


def test_entitlement_has_method() -> None:
    ent = Entitlement(
        tier=TIER_PRO,
        features=frozenset({FEATURE_BACKUP_SYNC}),
        holder="x",
        valid_until=None,
        status="ok",
    )
    assert ent.has(FEATURE_BACKUP_SYNC) is True
    assert ent.has(FEATURE_CLOUD_RELAY) is False


# ---------------------------------------------------------------------------
# Signing tests (require nacl)
# ---------------------------------------------------------------------------


def test_valid_pro_verifies(keypair: tuple[str, str]) -> None:
    """A correctly signed, unexpired pro license returns a pro Entitlement."""
    pytest.importorskip("nacl")
    sk_hex, pk_hex = keypair
    body = _make_body(tier=TIER_PRO)
    envelope = _make_envelope(body, sk_hex, key_id="dev-1")
    ent = entitlement_from_raw(envelope, public_keys={"dev-1": pk_hex})
    assert ent.tier == TIER_PRO
    assert ent.has(FEATURE_BACKUP_SYNC) is True
    assert ent.has(FEATURE_GOOGLE_SKILLS) is True
    assert TIER_PRO in ent.status


def test_valid_pro_has_feature_true(keypair: tuple[str, str]) -> None:
    pytest.importorskip("nacl")
    sk_hex, pk_hex = keypair
    body = _make_body(tier=TIER_PRO, features=[FEATURE_CLOUD_RELAY])
    envelope = _make_envelope(body, sk_hex)
    ent = entitlement_from_raw(envelope, public_keys={"dev-1": pk_hex})
    assert ent.has(FEATURE_CLOUD_RELAY) is True
    assert ent.has(FEATURE_BACKUP_SYNC) is False


def test_tampered_body_returns_free(keypair: tuple[str, str]) -> None:
    """Mutating the body after signing invalidates the signature -> FREE."""
    pytest.importorskip("nacl")
    sk_hex, pk_hex = keypair
    body = _make_body(tier=TIER_PRO)
    envelope = _make_envelope(body, sk_hex)
    # Tamper: change the tier inside the already-signed envelope.
    envelope["license"]["tier"] = TIER_FOUNDER
    ent = entitlement_from_raw(envelope, public_keys={"dev-1": pk_hex})
    assert ent.tier == TIER_FREE
    assert "did not verify" in ent.status


def test_expired_license_returns_free(keypair: tuple[str, str]) -> None:
    pytest.importorskip("nacl")
    sk_hex, pk_hex = keypair
    body = _make_body(tier=TIER_PRO, expires_at=_past(10))
    envelope = _make_envelope(body, sk_hex)
    ent = entitlement_from_raw(envelope, public_keys={"dev-1": pk_hex})
    assert ent.tier == TIER_FREE
    assert "expired" in ent.status


def test_naive_expires_at_still_grants_paid(keypair: tuple[str, str]) -> None:
    """A license whose expires_at omits the timezone (naive ISO) must still
    grant the paid tier, not silently crash into FREE.

    Comparing a naive datetime against an aware UTC now raises TypeError (not
    ValueError); if that escaped the parse guard it would lock out a paying
    holder. Regression for the verify.py naive-datetime bug.
    """
    pytest.importorskip("nacl")
    sk_hex, pk_hex = keypair
    naive_future = (datetime.now(UTC) + timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S")
    assert "Z" not in naive_future and "+" not in naive_future
    body = _make_body(tier=TIER_PRO, expires_at=naive_future)
    envelope = _make_envelope(body, sk_hex)
    ent = entitlement_from_raw(envelope, public_keys={"dev-1": pk_hex})
    assert ent.tier == TIER_PRO
    assert ent.has(FEATURE_BACKUP_SYNC) is True


def test_naive_expired_license_returns_free(keypair: tuple[str, str]) -> None:
    """A naive (tz-less) expiry in the past is still correctly treated as expired."""
    pytest.importorskip("nacl")
    sk_hex, pk_hex = keypair
    naive_past = (datetime.now(UTC) - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%S")
    body = _make_body(tier=TIER_PRO, expires_at=naive_past)
    envelope = _make_envelope(body, sk_hex)
    ent = entitlement_from_raw(envelope, public_keys={"dev-1": pk_hex})
    assert ent.tier == TIER_FREE
    assert "expired" in ent.status


def test_non_ascii_holder_verifies(keypair: tuple[str, str]) -> None:
    """A holder name with non-ASCII characters must verify.

    Locks in the canonical-JSON convention (ensure_ascii=False) end to end: the
    signer (_make_envelope) and the verifier must serialize the body to the same
    UTF-8 bytes, or a legitimately paid holder with an accented name is rejected.
    """
    pytest.importorskip("nacl")
    sk_hex, pk_hex = keypair
    body = _make_body(tier=TIER_PRO, holder="José Söderström <jose@example.com>")
    envelope = _make_envelope(body, sk_hex)
    ent = entitlement_from_raw(envelope, public_keys={"dev-1": pk_hex})
    assert ent.tier == TIER_PRO
    assert "José" in ent.status


def test_lifetime_founder_valid(keypair: tuple[str, str]) -> None:
    """A founder license with expires_at=null is valid indefinitely."""
    pytest.importorskip("nacl")
    sk_hex, pk_hex = keypair
    body = _make_body(tier=TIER_FOUNDER, expires_at=None)
    # Manually set expires_at to None in the dict (default helper sets a date).
    body = dict(body)
    body["expires_at"] = None
    envelope = _make_envelope(body, sk_hex)
    ent = entitlement_from_raw(envelope, public_keys={"dev-1": pk_hex})
    assert ent.tier == TIER_FOUNDER
    assert ent.valid_until is None
    assert "lifetime" in ent.status


def test_unknown_feature_dropped(keypair: tuple[str, str]) -> None:
    """A feature key not in KNOWN_FEATURES is silently dropped."""
    pytest.importorskip("nacl")
    sk_hex, pk_hex = keypair
    body = _make_body(features=[FEATURE_BACKUP_SYNC, "unknown_future_feature"])
    envelope = _make_envelope(body, sk_hex)
    ent = entitlement_from_raw(envelope, public_keys={"dev-1": pk_hex})
    assert ent.has(FEATURE_BACKUP_SYNC) is True
    assert ent.has("unknown_future_feature") is False
    assert "unknown_future_feature" not in ent.features


def test_unknown_tier_returns_free(keypair: tuple[str, str]) -> None:
    pytest.importorskip("nacl")
    sk_hex, pk_hex = keypair
    body = _make_body(tier="enterprise")
    envelope = _make_envelope(body, sk_hex)
    ent = entitlement_from_raw(envelope, public_keys={"dev-1": pk_hex})
    assert ent.tier == TIER_FREE


def test_get_entitlement_with_license_file(
    keypair: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writing a valid license.json and setting JUNE_LICENSE_PATH -> pro entitlement."""
    pytest.importorskip("nacl")
    sk_hex, pk_hex = keypair
    body = _make_body(tier=TIER_PRO)
    envelope = _make_envelope(body, sk_hex)

    license_file = tmp_path / "license.json"
    license_file.write_text(json.dumps(envelope), encoding="utf-8")

    monkeypatch.setenv("JUNE_LICENSE_PATH", str(license_file))
    # Monkeypatch PUBLIC_KEYS in the store so the shipped empty map doesn't gate us.
    import june_brain.licensing.store as store_mod

    monkeypatch.setattr(store_mod, "PUBLIC_KEYS", {"dev-1": pk_hex})
    reset_for_tests()

    ent = get_entitlement()
    assert ent.tier == TIER_PRO
    assert ent.has(FEATURE_BACKUP_SYNC) is True


def test_unreadable_file_returns_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable (bad JSON) file returns FREE, not an exception."""
    bad_file = tmp_path / "bad_license.json"
    bad_file.write_text("not valid json{{{{", encoding="utf-8")
    monkeypatch.setenv("JUNE_LICENSE_PATH", str(bad_file))
    reset_for_tests()
    ent = get_entitlement()
    assert ent.tier == TIER_FREE


def test_get_entitlement_no_file_returns_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No license file present -> exactly FREE (tier and features)."""
    monkeypatch.setenv("JUNE_LICENSE_PATH", str(tmp_path / "does_not_exist.json"))
    reset_for_tests()
    ent = get_entitlement()
    assert ent is FREE or (ent.tier == TIER_FREE and ent.features == frozenset())


def test_public_keys_ships_empty() -> None:
    """Confirm PUBLIC_KEYS is empty (safe pre-launch default)."""
    from june_brain.licensing.keys import PUBLIC_KEYS

    assert PUBLIC_KEYS == {}
