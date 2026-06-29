"""Tests for the Trust Ledger hash chain (ADR 0022, slice 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from june_brain.trust import (
    GENESIS_PREV,
    Ed25519Signer,
    LedgerReader,
    LedgerWriter,
    compute_entry_hash,
    generate_keypair,
    reset_for_tests,
    signing_available,
)
from june_brain.trust.ledger import _canonical_json


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import june_brain.memory as memory_pkg
    import june_brain.memory.sqlite as memory_sqlite

    monkeypatch.setattr(memory_pkg, "MEMORY_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(memory_sqlite, "_local", type(memory_sqlite._local)())
    reset_for_tests()
    yield
    reset_for_tests()


def _append_some(writer: LedgerWriter, n: int = 3) -> None:
    for i in range(n):
        writer.append(kind="egress", actor="june", payload={"model": "gemini", "i": i})


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------


def test_first_entry_links_to_genesis() -> None:
    writer = LedgerWriter()
    entry = writer.append(kind="system", actor="june", payload={"note": "boot"})
    assert entry.seq == 1
    assert entry.prev_hash == GENESIS_PREV
    assert GENESIS_PREV == "0" * 64
    assert len(entry.entry_hash) == 64


def test_chain_is_gap_free_and_linked() -> None:
    writer = LedgerWriter()
    _append_some(writer, 4)
    reader = LedgerReader()
    page = list(reversed(reader.page(limit=100)))  # oldest-first
    assert [e.seq for e in page] == [1, 2, 3, 4]
    prev = GENESIS_PREV
    for e in page:
        assert e.prev_hash == prev
        prev = e.entry_hash


def test_entry_hash_matches_canonical_recompute() -> None:
    writer = LedgerWriter()
    e = writer.append(kind="action", actor="user", payload={"b": 2, "a": 1})
    # payload is stored as canonical JSON; recompute over that exact text.
    payload_text = _canonical_json({"a": 1, "b": 2})
    expected = compute_entry_hash(
        seq=e.seq,
        id=e.id,
        ts=e.ts,
        kind="action",
        actor="user",
        payload=payload_text,
        prev_hash=GENESIS_PREV,
    )
    assert e.entry_hash == expected


def test_invalid_kind_and_actor_raise() -> None:
    writer = LedgerWriter()
    with pytest.raises(ValueError):
        writer.append(kind="bogus", actor="june", payload={})
    with pytest.raises(ValueError):
        writer.append(kind="egress", actor="robot", payload={})


# ---------------------------------------------------------------------------
# Verification + tamper detection
# ---------------------------------------------------------------------------


def test_clean_chain_verifies() -> None:
    writer = LedgerWriter()
    _append_some(writer, 5)
    result = LedgerReader().verify_chain()
    assert result.ok is True
    assert result.first_broken_seq is None
    assert result.signed is False


def test_tamper_with_payload_is_detected_at_right_seq() -> None:
    writer = LedgerWriter()
    _append_some(writer, 5)
    reader = LedgerReader()
    # Mutate the payload of seq 3 directly, as an attacker editing the db would.
    reader._conn.execute(
        "UPDATE trust_ledger SET payload=? WHERE seq=3",
        (_canonical_json({"model": "evil", "i": 99}),),
    )
    reader._conn.commit()
    result = reader.verify_chain()
    assert result.ok is False
    assert result.first_broken_seq == 3


def test_tamper_with_recomputed_hash_breaks_next_link() -> None:
    # If the attacker also fixes seq 3's entry_hash but not the rest of the chain,
    # seq 4's stored prev_hash no longer matches → break surfaces at seq 4.
    writer = LedgerWriter()
    _append_some(writer, 5)
    reader = LedgerReader()
    new_payload = _canonical_json({"model": "evil", "i": 99})
    forged_hash = compute_entry_hash(
        seq=3,
        id=str(reader._conn.execute("SELECT id FROM trust_ledger WHERE seq=3").fetchone()["id"]),
        ts=str(reader._conn.execute("SELECT ts FROM trust_ledger WHERE seq=3").fetchone()["ts"]),
        kind="egress",
        actor="june",
        payload=new_payload,
        prev_hash=str(
            reader._conn.execute("SELECT prev_hash FROM trust_ledger WHERE seq=3").fetchone()[
                "prev_hash"
            ]
        ),
    )
    reader._conn.execute(
        "UPDATE trust_ledger SET payload=?, entry_hash=? WHERE seq=3",
        (new_payload, forged_hash),
    )
    reader._conn.commit()
    result = reader.verify_chain()
    assert result.ok is False
    assert result.first_broken_seq == 4


def test_deleting_a_row_creates_a_gap_that_breaks() -> None:
    writer = LedgerWriter()
    _append_some(writer, 4)
    reader = LedgerReader()
    reader._conn.execute("DELETE FROM trust_ledger WHERE seq=2")
    reader._conn.commit()
    result = reader.verify_chain()
    assert result.ok is False
    # seq 3 is now where the expected sequence (1,2,...) first diverges.
    assert result.first_broken_seq == 3


def test_empty_ledger_verifies() -> None:
    assert LedgerReader().verify_chain().ok is True


# ---------------------------------------------------------------------------
# Redaction — no secret ever enters the ledger
# ---------------------------------------------------------------------------


def test_configured_secret_is_redacted_before_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "AIzaSyABCDEF1234567890abcdefghij"  # key-shaped + a configured value
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    writer = LedgerWriter()
    writer.append(
        kind="egress",
        actor="june",
        payload={"prompt_summary": f"used key {secret} to call cloud"},
    )
    reader = LedgerReader()
    row = reader._conn.execute("SELECT payload FROM trust_ledger WHERE seq=1").fetchone()
    assert secret not in row["payload"]
    assert "[REDACTED]" in row["payload"]
    # The chain still verifies (hash committed to the redacted text).
    assert reader.verify_chain().ok is True


# ---------------------------------------------------------------------------
# Pagination + summary helpers
# ---------------------------------------------------------------------------


def test_page_is_newest_first_with_cursor() -> None:
    writer = LedgerWriter()
    _append_some(writer, 6)
    reader = LedgerReader()
    first = reader.page(limit=2)
    assert [e.seq for e in first] == [6, 5]
    nxt = reader.page(cursor=first[-1].seq, limit=2)
    assert [e.seq for e in nxt] == [4, 3]


def test_summary_helpers() -> None:
    writer = LedgerWriter()
    writer.append(kind="egress", actor="june", payload={"m": 1})
    writer.append(kind="action", actor="june", payload={"a": 1})
    writer.append(kind="egress", actor="june", payload={"m": 2})
    reader = LedgerReader()
    assert reader.count() == 3
    assert reader.egress_count_since("0000-01-01T00:00:00+00:00") == 2
    assert reader.last_entry_ts() is not None


# ---------------------------------------------------------------------------
# Optional Ed25519 signing
# ---------------------------------------------------------------------------


def test_signing_round_trip() -> None:
    pytest.importorskip("nacl")
    sk_hex, pk_hex = generate_keypair()
    writer = LedgerWriter(signer=Ed25519Signer(sk_hex))
    writer.append(kind="egress", actor="june", payload={"m": 1})
    writer.append(kind="action", actor="june", payload={"a": 1})
    reader = LedgerReader()
    rows = reader._conn.execute("SELECT sig FROM trust_ledger").fetchall()
    assert all(r["sig"] for r in rows)
    result = reader.verify_chain(verify_key_hex=pk_hex)
    assert result.ok is True
    assert result.signed is True


def test_signed_chain_tamper_is_detected() -> None:
    pytest.importorskip("nacl")
    sk_hex, pk_hex = generate_keypair()
    writer = LedgerWriter(signer=Ed25519Signer(sk_hex))
    _append_some(writer, 3)
    reader = LedgerReader()
    reader._conn.execute(
        "UPDATE trust_ledger SET payload=? WHERE seq=2",
        (_canonical_json({"tampered": True}),),
    )
    reader._conn.commit()
    result = reader.verify_chain(verify_key_hex=pk_hex)
    assert result.ok is False
    assert result.first_broken_seq == 2


def test_signing_available_is_bool() -> None:
    assert isinstance(signing_available(), bool)
