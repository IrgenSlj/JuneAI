"""Tests for the entitlement surface — GET /system (entitlement field) and GET /system/entitlement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from june_api.app import create_app

# ---------------------------------------------------------------------------
# Shared client fixture (mirrors test_system_routes.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Fresh app + isolated SQLite + no license file."""
    import june_brain.activity as activity_pkg
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    activity_pkg.reset_for_tests()

    import june_brain.trust as trust_pkg

    trust_pkg.reset_for_tests()

    # No license file present.
    monkeypatch.setenv("JUNE_LICENSE_PATH", str(tmp_path / "no_license.json"))
    import june_brain.licensing as licensing_pkg

    licensing_pkg.reset_for_tests()

    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Free-default tests (always run, no nacl dependency)
# ---------------------------------------------------------------------------


def test_system_entitlement_endpoint_free_default(client: TestClient) -> None:
    """GET /system/entitlement returns free tier when no license is installed."""
    res = client.get("/system/entitlement")
    assert res.status_code == 200
    body = res.json()
    assert body["tier"] == "free"
    assert isinstance(body["status"], str)
    assert len(body["status"]) > 0
    assert isinstance(body["features"], list)
    assert body["features"] == []


def test_system_includes_entitlement_field(client: TestClient) -> None:
    """GET /system includes an 'entitlement' object with tier == 'free' by default."""
    res = client.get("/system")
    assert res.status_code == 200
    body = res.json()
    assert "entitlement" in body
    ent = body["entitlement"]
    assert ent is not None
    assert ent["tier"] == "free"
    assert isinstance(ent["status"], str)
    assert isinstance(ent["features"], list)


def test_system_entitlement_shape(client: TestClient) -> None:
    """GET /system/entitlement response has all required fields."""
    body = client.get("/system/entitlement").json()
    for key in ("tier", "status", "features"):
        assert key in body
    # holder and valid_until are nullable but must be present
    assert "holder" in body
    assert "valid_until" in body


# ---------------------------------------------------------------------------
# Pro-license fixture tests (require nacl)
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _make_pro_envelope(sk_hex: str, pk_hex: str, key_id: str = "dev-1") -> dict[str, Any]:
    from datetime import UTC, datetime, timedelta

    from nacl.encoding import HexEncoder
    from nacl.signing import SigningKey

    expires_at = (datetime.now(UTC) + timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
    body: dict[str, Any] = {
        "schema_version": 1,
        "license_id": "JUNE-API-TEST-001",
        "tier": "pro",
        "holder": "API Test User <apitest@example.com>",
        "seats": 1,
        "issued_at": "2026-07-01T00:00:00Z",
        "expires_at": expires_at,
        "features": ["backup_sync", "cloud_relay", "google_skills", "supporter"],
    }
    sk = SigningKey(sk_hex.encode("ascii"), encoder=HexEncoder)
    message = _canonical_json(body).encode("utf-8")
    signed = sk.sign(message)
    sig_hex = signed.signature.hex()
    return {"license": body, "key_id": key_id, "sig": sig_hex}


@pytest.fixture
def pro_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """App with a valid signed pro license on disk."""
    pytest.importorskip("nacl")

    from nacl.encoding import HexEncoder
    from nacl.signing import SigningKey

    # Generate ephemeral keypair.
    sk = SigningKey.generate()
    sk_hex = sk.encode(encoder=HexEncoder).decode("ascii")
    pk_hex = sk.verify_key.encode(encoder=HexEncoder).decode("ascii")

    # Write signed license file.
    envelope = _make_pro_envelope(sk_hex, pk_hex, key_id="dev-1")
    license_file = tmp_path / "license.json"
    license_file.write_text(json.dumps(envelope), encoding="utf-8")

    # Wire up app fixtures.
    import june_brain.activity as activity_pkg
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    activity_pkg.reset_for_tests()

    import june_brain.trust as trust_pkg

    trust_pkg.reset_for_tests()

    # Point licensing at the file and inject the public key.
    monkeypatch.setenv("JUNE_LICENSE_PATH", str(license_file))
    import june_brain.licensing.store as store_mod

    monkeypatch.setattr(store_mod, "PUBLIC_KEYS", {"dev-1": pk_hex})
    import june_brain.licensing as licensing_pkg

    licensing_pkg.reset_for_tests()

    return TestClient(create_app())


def test_entitlement_endpoint_pro(pro_client: TestClient) -> None:
    """GET /system/entitlement returns pro tier with features when a valid license is present."""
    pytest.importorskip("nacl")
    res = pro_client.get("/system/entitlement")
    assert res.status_code == 200
    body = res.json()
    assert body["tier"] == "pro"
    assert "pro" in body["status"]
    assert "backup_sync" in body["features"]
    assert "cloud_relay" in body["features"]
    assert body["features"] == sorted(body["features"])


def test_system_includes_pro_entitlement(pro_client: TestClient) -> None:
    """GET /system embeds pro entitlement when a valid license is present."""
    pytest.importorskip("nacl")
    res = pro_client.get("/system")
    assert res.status_code == 200
    ent = res.json()["entitlement"]
    assert ent is not None
    assert ent["tier"] == "pro"
    assert "backup_sync" in ent["features"]
