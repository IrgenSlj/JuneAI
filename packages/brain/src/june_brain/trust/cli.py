"""``june-verify`` — check the Trust Ledger without asking June (ADR 0022).

June's pitch is that she can prove what she did. A proof that can only be
inspected through her own UI, while she is running, is not one — it is the
system reporting on itself. This command reads the database directly, exits
non-zero when the chain is broken, and can export the chain for someone else to
check with code that is not June's.

    june-verify                     verify the live ledger
    june-verify --json              the same, machine-readable
    june-verify --export chain.jsonl
    june-verify --check chain.jsonl verify an export, no database involved
    june-verify --db /path/june.db  point at a specific file

Exit codes: 0 intact, 1 broken, 2 usable error (missing file, bad arguments).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .ledger import GENESIS_PREV, VerifyResult
from .verify import REQUIRED_FIELDS, verify_entries

_COLUMNS = (*REQUIRED_FIELDS, "sig")


def _resolve_db(explicit: str | None) -> str:
    if explicit:
        return explicit
    from ..memory.sqlite import db_path

    return db_path()


def _read_rows(db: str) -> list[dict[str, Any]]:
    """Read the chain with a plain read-only connection.

    Deliberately not the app's pooled connection helper: this command must work
    against a copied file, on a machine that has never run June, while the app
    is open, and without triggering schema creation or corruption recovery.
    """
    uri = f"file:{Path(db).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cols = ", ".join(_COLUMNS)
        rows = conn.execute(f"SELECT {cols} FROM trust_ledger ORDER BY seq ASC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _high_water(db: str) -> int | None:
    uri = f"file:{Path(db).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        row = conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name='trust_ledger'"
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def _summarise(rows: list[dict[str, Any]], result: VerifyResult, *, truncation: bool) -> str:
    if not rows:
        return "Ledger is empty. Nothing to verify."

    kinds = Counter(str(r["kind"]) for r in rows)
    breakdown = ", ".join(f"{k} {n}" for k, n in sorted(kinds.items()))
    signed = "signed (Ed25519)" if result.signed else "unsigned (hash chain only)"

    lines = [
        f"Entries    {len(rows)}  ({breakdown})",
        f"Range      {rows[0]['ts']}  ->  {rows[-1]['ts']}",
        f"Integrity  {signed}",
        f"Head       {str(rows[-1]['entry_hash'])[:16]}...",
    ]
    if not truncation:
        lines.append(
            "Note       tail-truncation cannot be detected from an export; "
            "run against the database for that"
        )
    lines.append("")
    if result.ok:
        lines.append("OK — every entry hashes to its stored value and links to the one before it.")
    else:
        lines.append(
            f"BROKEN — the chain first fails at entry {result.first_broken_seq}. "
            "Entries before it are intact; that one was altered, removed, or inserted."
        )
    return "\n".join(lines)


def _export(rows: list[dict[str, Any]], path: str) -> None:
    """Write the chain as JSONL, one entry per line, in chain order.

    ``payload`` stays the exact stored string rather than a parsed object: the
    hash commits to those bytes, so re-serialising it — even to equivalent
    JSON — would produce a file that cannot be verified.
    """
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({k: row[k] for k in _COLUMNS}, ensure_ascii=False) + "\n")


def _read_export(path: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except ValueError as exc:
                raise SystemExit(f"june-verify: {path}:{lineno} is not valid JSON ({exc})")
    return entries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="june-verify",
        description="Verify June's Trust Ledger hash chain.",
    )
    ap.add_argument("--db", help="path to june.db (default: June's data directory)")
    ap.add_argument("--check", metavar="FILE", help="verify an exported chain instead")
    ap.add_argument("--export", metavar="FILE", help="write the chain as JSONL and exit")
    ap.add_argument("--key", help="Ed25519 public key hex (default: this device's)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    if args.check and args.export:
        print("june-verify: --check and --export are mutually exclusive", file=sys.stderr)
        return 2

    if args.check:
        rows, high_water, source = _read_export(args.check), None, args.check
    else:
        db = _resolve_db(args.db)
        if not Path(db).exists():
            print(f"june-verify: no ledger at {db}", file=sys.stderr)
            return 2
        try:
            rows = _read_rows(db)
        except sqlite3.DatabaseError as exc:
            print(f"june-verify: cannot read {db}: {exc}", file=sys.stderr)
            return 2
        high_water, source = _high_water(db), db

        if args.export:
            _export(rows, args.export)
            print(f"Wrote {len(rows)} entries to {args.export}")
            print("Verify it anywhere with:  june-verify --check " + args.export)
            return 0

    key = args.key
    if key is None and not args.check:
        from .signing import device_public_key

        key = device_public_key()

    result = verify_entries(rows, verify_key_hex=key, high_water=high_water)

    if args.json:
        print(
            json.dumps(
                {
                    "source": source,
                    "entries": len(rows),
                    "genesis_prev": GENESIS_PREV,
                    **result.to_dict(),
                },
                indent=2,
            )
        )
    else:
        print(f"Ledger     {source}")
        print(_summarise(rows, result, truncation=high_water is not None))

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
