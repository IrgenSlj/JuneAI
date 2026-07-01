"""Offline license / entitlement system for June.

Flat re-export of the public API.  Import from here; do not reach into
sub-modules directly.
"""

from __future__ import annotations

from .keys import PUBLIC_KEYS
from .models import (
    FEATURE_BACKUP_SYNC,
    FEATURE_CLOUD_RELAY,
    FEATURE_COMMERCIAL_USE,
    FEATURE_GOOGLE_SKILLS,
    FEATURE_SUPPORTER,
    FREE,
    KNOWN_FEATURES,
    KNOWN_TIERS,
    TIER_FOUNDER,
    TIER_FREE,
    TIER_PRO,
    Entitlement,
    LicenseBody,
)
from .store import get_entitlement, reset_for_tests
from .verify import entitlement_from_raw

__all__ = [
    # Core types
    "Entitlement",
    "LicenseBody",
    # Tier constants
    "TIER_FREE",
    "TIER_PRO",
    "TIER_FOUNDER",
    "KNOWN_TIERS",
    # Feature constants
    "FEATURE_BACKUP_SYNC",
    "FEATURE_CLOUD_RELAY",
    "FEATURE_COMMERCIAL_USE",
    "FEATURE_GOOGLE_SKILLS",
    "FEATURE_SUPPORTER",
    "KNOWN_FEATURES",
    # Default / safe value
    "FREE",
    # Public keys (shipped empty pre-launch)
    "PUBLIC_KEYS",
    # Runtime API
    "get_entitlement",
    "reset_for_tests",
    "entitlement_from_raw",
]
