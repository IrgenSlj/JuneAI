"""Trust Ledger — append-only, hash-chained, locally verifiable provenance (ADR 0022).

June already makes egress *visible* (the privacy dial, provenance frame, glass-box
trace). The ledger makes it *provable*: an append-only chain of every cloud egress
and every consequential action, where altering or removing one entry breaks the
chain detectably. It lives in the single ``june.db`` (ADR 0019), is device-global
(one monotonic chain, not per-user), and is written centrally at the seams that
already see these events so no skill can perform egress and skip the record.

Two layers of integrity:

1. **Hash chain (stdlib).** Each entry stores ``entry_hash = blake2b(canonical_json
   ({seq,id,ts,kind,actor,payload,prev_hash}))``. ``prev_hash`` links to the prior
   row's ``entry_hash``; the genesis ``prev_hash`` is 64 zeros. Tamper-evidence
   needs no key.
2. **Optional Ed25519 signature** (``signing.py``, lazy PyNaCl) over ``entry_hash``,
   adding authenticity. Unsigned is a valid mode.

Payloads are redacted (``guard/redaction.py``) *before* they are written, centrally
in the writer, so "no secret ever enters the ledger" does not depend on any caller
remembering. Writes are best-effort at the call sites (a ledger failure must never
break a model call), but the chain itself is always internally consistent because
appends are serialized under a process-wide lock and assign ``seq`` explicitly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ..guard.redaction import redact_secrets
from ..memory.sqlite import _get_connection, db_path
from .signing import Signer, device_public_key, verify_sig

logger = logging.getLogger(__name__)

# blake2b with a 32-byte digest yields 64 hex chars, matching the 64-zero genesis.
_DIGEST_SIZE = 32
GENESIS_PREV = "0" * (_DIGEST_SIZE * 2)

VALID_KINDS = frozenset({"egress", "action", "approval", "system"})
VALID_ACTORS = frozenset({"june", "user"})

# Serializes appends across threads so the (read-last, compute-hash, insert) step
# is atomic — two racing appends can never share a prev_hash and fork the chain.
_APPEND_LOCK = threading.Lock()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trust_ledger (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    id          TEXT NOT NULL,
    ts          TEXT NOT NULL,
    kind        TEXT NOT NULL,
    actor       TEXT NOT NULL,
    payload     TEXT NOT NULL,
    prev_hash   TEXT NOT NULL,
    entry_hash  TEXT NOT NULL,
    sig         TEXT
);
CREATE INDEX IF NOT EXISTS idx_trust_ledger_ts ON trust_ledger(ts);
CREATE INDEX IF NOT EXISTS idx_trust_ledger_kind ON trust_ledger(kind, seq);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(obj: Any) -> str:
    """Sorted-keys, no-whitespace, UTF-8 JSON — the canonical form the chain hashes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_entry_hash(
    *,
    seq: int,
    id: str,
    ts: str,
    kind: str,
    actor: str,
    payload: str,
    prev_hash: str,
) -> str:
    """The blake2b digest committing to one entry. ``payload`` is the stored text."""
    material = _canonical_json(
        {
            "actor": actor,
            "id": id,
            "kind": kind,
            "payload": payload,
            "prev_hash": prev_hash,
            "seq": seq,
            "ts": ts,
        }
    )
    return hashlib.blake2b(material.encode("utf-8"), digest_size=_DIGEST_SIZE).hexdigest()


@dataclass(frozen=True)
class LedgerEntry:
    """One immutable row of the chain."""

    seq: int
    id: str
    ts: str
    kind: str
    actor: str
    payload: dict[str, Any]
    prev_hash: str
    entry_hash: str
    sig: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "id": self.id,
            "ts": self.ts,
            "kind": self.kind,
            "actor": self.actor,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
            "sig": self.sig,
        }


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    first_broken_seq: int | None
    signed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "first_broken_seq": self.first_broken_seq,
            "signed": self.signed,
        }


def _parse_payload(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (TypeError, ValueError):
        return {"raw": text}


class LedgerWriter:
    """Append-only writer. Exposes ``append`` only — no update/delete surface."""

    def __init__(self, *, signer: Signer | None = None) -> None:
        self._db_path = db_path()
        self._signer = signer
        conn = _get_connection(self._db_path)
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    @property
    def _conn(self) -> sqlite3.Connection:
        return _get_connection(self._db_path)

    def append(
        self,
        *,
        kind: str,
        actor: str,
        payload: dict[str, Any],
    ) -> LedgerEntry:
        """Append one entry, redacting the payload and linking the hash chain.

        ``kind`` must be one of :data:`VALID_KINDS`; ``actor`` one of
        :data:`VALID_ACTORS`. The payload is canonicalized and run through
        ``redact_secrets`` before storage, so a configured secret can never land
        in the ledger regardless of which caller built the payload.
        """
        if kind not in VALID_KINDS:
            raise ValueError(f"invalid ledger kind {kind!r}")
        if actor not in VALID_ACTORS:
            raise ValueError(f"invalid ledger actor {actor!r}")

        # Canonicalize, then redact the serialized text. Redaction replaces secret
        # substrings inside JSON string values, leaving valid JSON.
        payload_text = redact_secrets(_canonical_json(payload))
        entry_id = uuid4().hex
        ts = _now()

        with _APPEND_LOCK:
            conn = self._conn
            row = conn.execute(
                "SELECT seq, entry_hash FROM trust_ledger ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            if row is None:
                seq = 1
                prev_hash = GENESIS_PREV
            else:
                seq = int(row["seq"]) + 1
                prev_hash = str(row["entry_hash"])

            entry_hash = compute_entry_hash(
                seq=seq,
                id=entry_id,
                ts=ts,
                kind=kind,
                actor=actor,
                payload=payload_text,
                prev_hash=prev_hash,
            )
            sig = None
            if self._signer is not None:
                try:
                    sig = self._signer.sign(entry_hash)
                except Exception:  # noqa: BLE001 - signing must never break the append
                    logger.debug("ledger entry signing failed; storing unsigned", exc_info=True)
                    sig = None

            conn.execute(
                "INSERT INTO trust_ledger (seq, id, ts, kind, actor, payload, prev_hash, entry_hash, sig) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (seq, entry_id, ts, kind, actor, payload_text, prev_hash, entry_hash, sig),
            )
            conn.commit()

        return LedgerEntry(
            seq=seq,
            id=entry_id,
            ts=ts,
            kind=kind,
            actor=actor,
            payload=_parse_payload(payload_text),
            prev_hash=prev_hash,
            entry_hash=entry_hash,
            sig=sig,
        )


class LedgerReader:
    """Read side: paginated receipts, verification, and a summary for /system."""

    def __init__(self) -> None:
        self._db_path = db_path()
        conn = _get_connection(self._db_path)
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    @property
    def _conn(self) -> sqlite3.Connection:
        return _get_connection(self._db_path)

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM trust_ledger").fetchone()
        return int(row["n"]) if row else 0

    def page(self, *, cursor: int | None = None, limit: int = 50) -> list[LedgerEntry]:
        """Newest-first page. ``cursor`` is an exclusive upper bound on ``seq``."""
        limit = max(1, min(int(limit), 500))
        if cursor is not None:
            rows = self._conn.execute(
                "SELECT * FROM trust_ledger WHERE seq < ? ORDER BY seq DESC LIMIT ?",
                (int(cursor), limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM trust_ledger ORDER BY seq DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def egress_count_since(self, since_iso: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM trust_ledger WHERE kind='egress' AND ts >= ?",
            (since_iso,),
        ).fetchone()
        return int(row["n"]) if row else 0

    def last_entry_ts(self) -> str | None:
        row = self._conn.execute(
            "SELECT ts FROM trust_ledger ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return str(row["ts"]) if row else None

    def verify_chain(self, *, verify_key_hex: str | None = None) -> VerifyResult:
        """Recompute every hash, check linkage, verify signatures when keyed.

        Returns ``ok=True`` with ``first_broken_seq=None`` when intact, otherwise
        the first ``seq`` where the chain breaks (a gap, a bad link, a payload that
        no longer hashes to its stored ``entry_hash``, or a failed signature).

        Records the result + timestamp in a process-local cache so the cheap
        ``/system`` summary can report when the chain was last verified without
        re-hashing on every status poll.
        """
        result = self._verify(verify_key_hex=verify_key_hex)
        _record_verification(self._db_path, result.ok)
        return result

    def _verify(self, *, verify_key_hex: str | None = None) -> VerifyResult:
        rows = self._conn.execute(
            "SELECT * FROM trust_ledger ORDER BY seq ASC"
        ).fetchall()

        if verify_key_hex is None:
            verify_key_hex = device_public_key()
        has_sig = any(r["sig"] for r in rows)
        can_check_sig = bool(verify_key_hex) and has_sig

        prev = GENESIS_PREV
        expected_seq = 1
        for row in rows:
            seq = int(row["seq"])
            if seq != expected_seq:
                return VerifyResult(ok=False, first_broken_seq=seq, signed=can_check_sig)
            if str(row["prev_hash"]) != prev:
                return VerifyResult(ok=False, first_broken_seq=seq, signed=can_check_sig)
            recomputed = compute_entry_hash(
                seq=seq,
                id=str(row["id"]),
                ts=str(row["ts"]),
                kind=str(row["kind"]),
                actor=str(row["actor"]),
                payload=str(row["payload"]),
                prev_hash=str(row["prev_hash"]),
            )
            if recomputed != str(row["entry_hash"]):
                return VerifyResult(ok=False, first_broken_seq=seq, signed=can_check_sig)
            if can_check_sig and row["sig"]:
                if not verify_sig(str(verify_key_hex), str(row["entry_hash"]), str(row["sig"])):
                    return VerifyResult(ok=False, first_broken_seq=seq, signed=can_check_sig)
            prev = str(row["entry_hash"])
            expected_seq += 1

        # Tail-truncation check. A plain hash chain cannot, on its own, detect
        # that the most recent entries were deleted: rows 1..N-k still form a
        # valid chain. But ``seq`` is AUTOINCREMENT, so sqlite keeps a high-water
        # mark in ``sqlite_sequence`` that does NOT drop when rows are deleted. If
        # it exceeds the last seq we verified, entries were truncated from the
        # tail. (An attacker with full db write access can still evade this by
        # also rewriting sqlite_sequence — see ADR 0022's verification contract.)
        hwm = self._high_water_seq()
        if hwm >= expected_seq:
            return VerifyResult(ok=False, first_broken_seq=expected_seq, signed=can_check_sig)

        return VerifyResult(ok=True, first_broken_seq=None, signed=can_check_sig)

    def _high_water_seq(self) -> int:
        """Highest seq ever allocated (survives row deletion via AUTOINCREMENT)."""
        try:
            row = self._conn.execute(
                "SELECT seq FROM sqlite_sequence WHERE name='trust_ledger'"
            ).fetchone()
            return int(row["seq"]) if row and row["seq"] is not None else 0
        except sqlite3.OperationalError:
            return 0


def _row_to_entry(row: sqlite3.Row) -> LedgerEntry:
    return LedgerEntry(
        seq=int(row["seq"]),
        id=str(row["id"]),
        ts=str(row["ts"]),
        kind=str(row["kind"]),
        actor=str(row["actor"]),
        payload=_parse_payload(str(row["payload"])),
        prev_hash=str(row["prev_hash"]),
        entry_hash=str(row["entry_hash"]),
        sig=row["sig"],
    )


# ---------------------------------------------------------------------------
# Process-wide convenience handles (used by the central wiring in slice 3).
# ---------------------------------------------------------------------------
_writer: LedgerWriter | None = None
_reader: LedgerReader | None = None
_handle_lock = threading.Lock()

# Process-local record of the last verification per db, so the /system summary
# can show "chain verified at <ts>" without re-hashing the whole chain on every
# poll. Verification stays an explicit act; this only caches its outcome.
_last_verify: dict[str, tuple[str, bool]] = {}


def _record_verification(db: str, ok: bool) -> None:
    _last_verify[db] = (_now(), ok)


def last_verification(db: str | None = None) -> tuple[str, bool] | None:
    """Return ``(ts, ok)`` of the last verify_chain for ``db`` (default: active), or None."""
    return _last_verify.get(db if db is not None else db_path())


def get_writer() -> LedgerWriter:
    """Process-wide unsigned writer for production wiring.

    Cached, but keyed on the active ``db_path()`` so swapping ``MEMORY_DIR``
    (as tests do) rebuilds against the new path rather than pinning to a stale
    db. Tests that want a signed writer construct ``LedgerWriter`` directly.
    """
    global _writer
    current = db_path()
    if _writer is None or _writer._db_path != current:
        with _handle_lock:
            if _writer is None or _writer._db_path != current:
                _writer = LedgerWriter()
    return _writer


def get_reader() -> LedgerReader:
    global _reader
    current = db_path()
    if _reader is None or _reader._db_path != current:
        with _handle_lock:
            if _reader is None or _reader._db_path != current:
                _reader = LedgerReader()
    return _reader


def reset_for_tests() -> None:
    """Drop cached handles so fixtures that swap MEMORY_DIR get a fresh db path."""
    global _writer, _reader
    with _handle_lock:
        _writer = None
        _reader = None
        _last_verify.clear()
