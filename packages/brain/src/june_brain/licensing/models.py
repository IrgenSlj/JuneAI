"""Entitlement model and license body for the offline license system.

Entitlement is the single answer to "what can this installation do?".
It is always safe to read — the FREE constant is returned on every failure path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Tier constants
# ---------------------------------------------------------------------------

TIER_FREE = "free"
TIER_PRO = "pro"
TIER_FOUNDER = "founder"

KNOWN_TIERS: frozenset[str] = frozenset({TIER_FREE, TIER_PRO, TIER_FOUNDER})

# ---------------------------------------------------------------------------
# Feature key constants
# ---------------------------------------------------------------------------

FEATURE_BACKUP_SYNC = "backup_sync"
FEATURE_CLOUD_RELAY = "cloud_relay"
FEATURE_GOOGLE_SKILLS = "google_skills"
FEATURE_SUPPORTER = "supporter"
FEATURE_COMMERCIAL_USE = "commercial_use"

KNOWN_FEATURES: frozenset[str] = frozenset(
    {
        FEATURE_BACKUP_SYNC,
        FEATURE_CLOUD_RELAY,
        FEATURE_GOOGLE_SKILLS,
        FEATURE_SUPPORTER,
        FEATURE_COMMERCIAL_USE,
    }
)


# ---------------------------------------------------------------------------
# Entitlement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entitlement:
    """What a given installation is entitled to.

    Always safe to read. The FREE singleton is returned on every failure path
    so no code that reads entitlement can crash June or gate a free feature.
    """

    tier: str
    features: frozenset[str]
    holder: str
    valid_until: str | None  # ISO-8601 UTC or None for lifetime
    status: str  # honest, non-alarming, UI-ready description

    def has(self, feature: str) -> bool:
        """True when this entitlement includes the named pro feature."""
        return feature in self.features


# The default returned on every failure path (no file, bad sig, expired, etc.).
FREE = Entitlement(
    tier=TIER_FREE,
    features=frozenset(),
    holder="",
    valid_until=None,
    status="free — no license installed",
)


# ---------------------------------------------------------------------------
# LicenseBody
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LicenseBody:
    """The inner ``license`` object from the signed envelope.

    ``from_dict`` returns ``None`` on any shape or type mismatch so callers
    never have to handle exceptions.
    """

    schema_version: int
    license_id: str
    tier: str
    holder: str
    seats: int
    issued_at: str
    expires_at: str | None  # None means lifetime
    features: list[str]

    @staticmethod
    def from_dict(data: Any) -> LicenseBody | None:
        """Parse a dict into a LicenseBody; return None on any mismatch."""
        try:
            if not isinstance(data, dict):
                return None
            schema_version = data.get("schema_version")
            if not isinstance(schema_version, int):
                return None
            license_id = data.get("license_id")
            if not isinstance(license_id, str):
                return None
            tier = data.get("tier")
            if not isinstance(tier, str):
                return None
            holder = data.get("holder")
            if not isinstance(holder, str):
                return None
            seats = data.get("seats")
            if not isinstance(seats, int):
                return None
            issued_at = data.get("issued_at")
            if not isinstance(issued_at, str):
                return None
            expires_at = data.get("expires_at")
            if expires_at is not None and not isinstance(expires_at, str):
                return None
            features = data.get("features")
            if not isinstance(features, list):
                return None
            if not all(isinstance(f, str) for f in features):
                return None
            return LicenseBody(
                schema_version=schema_version,
                license_id=license_id,
                tier=tier,
                holder=holder,
                seats=seats,
                issued_at=issued_at,
                expires_at=expires_at,
                features=features,
            )
        except Exception:  # noqa: BLE001
            return None
