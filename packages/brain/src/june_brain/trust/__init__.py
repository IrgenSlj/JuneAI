"""Trust Ledger — tamper-evident, hash-chained local provenance (ADR 0022)."""

from __future__ import annotations

from .ledger import (
    GENESIS_PREV,
    VALID_ACTORS,
    VALID_KINDS,
    LedgerEntry,
    LedgerReader,
    LedgerWriter,
    VerifyResult,
    compute_entry_hash,
    get_reader,
    get_writer,
    last_verification,
    reset_for_tests,
)
from .signing import (
    Ed25519Signer,
    Signer,
    device_public_key,
    device_signer,
    ensure_device_key,
    generate_keypair,
    signing_available,
    verify_sig,
)

__all__ = [
    "GENESIS_PREV",
    "VALID_ACTORS",
    "VALID_KINDS",
    "LedgerEntry",
    "LedgerReader",
    "LedgerWriter",
    "VerifyResult",
    "compute_entry_hash",
    "get_reader",
    "get_writer",
    "last_verification",
    "reset_for_tests",
    "Ed25519Signer",
    "Signer",
    "device_public_key",
    "device_signer",
    "ensure_device_key",
    "generate_keypair",
    "signing_available",
    "verify_sig",
]
