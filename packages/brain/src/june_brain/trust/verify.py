"""The chain verification algorithm, separated from where the chain is stored.

ADR 0022 says the ledger is verifiable. Until now it was verifiable *by June* —
`LedgerReader.verify_chain` reads June's database, runs June's code, and reports that
everything is fine. Asking a system whether it has been honest is not an audit.

So the algorithm lives here, as a pure function over a sequence of entries, with
three callers that share it exactly: the reader (live database), the CLI
(`june-verify`), and the exported-file check. An export can be handed to someone
who does not run June and does not trust it, and checked against this
description:

    entry_hash = blake2b_256(canonical_json({
        actor, id, kind, payload, prev_hash, seq, ts
    }))

where canonical_json is sorted-keys, no whitespace, UTF-8, and `payload` is the
*stored string*, not a re-serialised object. prev_hash chains to the previous
row's entry_hash; the first is 64 zeros. That is the whole scheme, and it is
short enough to reimplement in any language in an afternoon — which is the point
of writing it down rather than shipping a verifier and asking for trust.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .ledger import GENESIS_PREV, VerifyResult, compute_entry_hash
from .signing import verify_sig

# The fields an entry must carry to be checkable. An export missing any of them
# cannot be verified, and saying so is better than verifying a subset silently.
REQUIRED_FIELDS = ("seq", "id", "ts", "kind", "actor", "payload", "prev_hash", "entry_hash")


def verify_entries(
    entries: Iterable[Mapping[str, Any]],
    *,
    verify_key_hex: str | None = None,
    high_water: int | None = None,
) -> VerifyResult:
    """Verify a hash chain. Pure — no database, no globals, no clock.

    ``entries`` must be ordered by ``seq`` ascending, each carrying
    :data:`REQUIRED_FIELDS` with ``payload`` as the exact stored string.

    ``high_water`` is the highest ``seq`` ever allocated, which sqlite keeps in
    ``sqlite_sequence`` and does not decrement when rows are deleted. Without it
    a chain truncated at the tail still verifies, because rows 1..N-k are a
    perfectly good chain — pass it when checking a live database, and accept the
    gap when checking an exported file.
    """
    rows = list(entries)
    # Whether signatures were checkable is a property of the whole chain, not of
    # how far verification got, so it is decided before the walk. A chain that
    # breaks at entry 1 was still a signed chain.
    can_check_sig = bool(verify_key_hex) and any(r.get("sig") for r in rows)

    prev = GENESIS_PREV
    expected_seq = 1

    for entry in rows:
        missing = [f for f in REQUIRED_FIELDS if entry.get(f) is None]
        if missing:
            return VerifyResult(ok=False, first_broken_seq=expected_seq, signed=False)

        seq = int(entry["seq"])
        if seq != expected_seq or str(entry["prev_hash"]) != prev:
            return VerifyResult(ok=False, first_broken_seq=seq, signed=can_check_sig)

        recomputed = compute_entry_hash(
            seq=seq,
            id=str(entry["id"]),
            ts=str(entry["ts"]),
            kind=str(entry["kind"]),
            actor=str(entry["actor"]),
            payload=str(entry["payload"]),
            prev_hash=str(entry["prev_hash"]),
        )
        if recomputed != str(entry["entry_hash"]):
            return VerifyResult(ok=False, first_broken_seq=seq, signed=can_check_sig)

        sig = entry.get("sig")
        if can_check_sig and sig:
            if not verify_sig(str(verify_key_hex or ""), str(entry["entry_hash"]), str(sig)):
                return VerifyResult(ok=False, first_broken_seq=seq, signed=True)

        prev = str(entry["entry_hash"])
        expected_seq += 1

    if high_water is not None and high_water >= expected_seq:
        # Entries were removed from the tail: seq numbers were allocated that no
        # longer have rows. (An attacker with write access to the file can also
        # rewrite sqlite_sequence — ADR 0022's verification contract says so.)
        return VerifyResult(ok=False, first_broken_seq=expected_seq, signed=can_check_sig)

    return VerifyResult(ok=True, first_broken_seq=None, signed=can_check_sig)
