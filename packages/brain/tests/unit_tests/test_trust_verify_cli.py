"""``june-verify`` — the ledger checked from outside June (ADR 0022, Phase 5.4).

Every test here tampers with a real database and asserts the command *fails*.
A verification test that only ever checks the intact case proves that the code
runs, not that it detects anything.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from june_brain.trust import cli
from june_brain.trust.verify import verify_entries

_SCHEMA = """
CREATE TABLE trust_ledger (
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
"""


def _chain(db: Path, count: int = 4) -> None:
    """Build a small valid chain with the same hashing the writer uses."""
    from june_brain.trust.ledger import GENESIS_PREV, compute_entry_hash

    conn = sqlite3.connect(db)
    conn.executescript(_SCHEMA)
    prev = GENESIS_PREV
    for seq in range(1, count + 1):
        payload = json.dumps({"tool": f"t{seq}", "action_class": "write_network"})
        entry = {
            "seq": seq,
            "id": f"id-{seq}",
            "ts": f"2026-07-26T10:0{seq}:00+00:00",
            "kind": "egress" if seq % 2 else "action",
            "actor": "june",
            "payload": payload,
            "prev_hash": prev,
        }
        entry_hash = compute_entry_hash(**entry)  # type: ignore[arg-type]
        conn.execute(
            "INSERT INTO trust_ledger "
            "(seq,id,ts,kind,actor,payload,prev_hash,entry_hash,sig) "
            "VALUES (?,?,?,?,?,?,?,?,NULL)",
            (*entry.values(), entry_hash),
        )
        prev = entry_hash
    conn.commit()
    conn.close()


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "june.db"
    _chain(path)
    return path


# -- the intact case ----------------------------------------------------


def test_an_intact_chain_exits_zero(db: Path, capsys) -> None:
    assert cli.main(["--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "OK" in out
    assert "action 2, egress 2" in out


def test_json_output_is_machine_readable(db: Path, capsys) -> None:
    assert cli.main(["--db", str(db), "--json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["ok"] is True
    assert parsed["entries"] == 4
    assert parsed["first_broken_seq"] is None


# -- tampering ----------------------------------------------------------


def test_an_edited_payload_is_detected(db: Path, capsys) -> None:
    """The point of the whole exercise: rewriting history must not verify."""
    conn = sqlite3.connect(db)
    conn.execute("UPDATE trust_ledger SET payload='{\"tool\":\"innocent\"}' WHERE seq=3")
    conn.commit()
    conn.close()

    assert cli.main(["--db", str(db)]) == 1
    assert "BROKEN" in capsys.readouterr().out


def test_a_deleted_entry_is_detected(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM trust_ledger WHERE seq=2")
    conn.commit()
    conn.close()
    assert cli.main(["--db", str(db)]) == 1


def test_a_truncated_tail_is_detected(db: Path, capsys) -> None:
    """Deleting the most recent entries leaves a chain that is valid on its own.

    Only the AUTOINCREMENT high-water mark remembers those rows existed, which
    is why the command reads sqlite_sequence and why an export cannot check it.
    """
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM trust_ledger WHERE seq >= 3")
    conn.commit()
    conn.close()

    assert cli.main(["--db", str(db)]) == 1
    assert "entry 3" in capsys.readouterr().out


def test_a_reordered_chain_is_detected(db: Path) -> None:
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT seq, ts FROM trust_ledger ORDER BY seq").fetchall()
    conn.execute("UPDATE trust_ledger SET ts=? WHERE seq=1", (rows[3][1],))
    conn.commit()
    conn.close()
    assert cli.main(["--db", str(db)]) == 1


# -- export and third-party verification --------------------------------


def test_an_export_round_trips(db: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "chain.jsonl"
    assert cli.main(["--db", str(db), "--export", str(out)]) == 0
    assert "Wrote 4 entries" in capsys.readouterr().out

    assert cli.main(["--check", str(out)]) == 0
    assert "OK" in capsys.readouterr().out


def test_a_tampered_export_fails_the_check(db: Path, tmp_path: Path) -> None:
    out = tmp_path / "chain.jsonl"
    cli.main(["--db", str(db), "--export", str(out)])

    lines = out.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[2])
    entry["payload"] = '{"tool":"innocent"}'
    lines[2] = json.dumps(entry)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert cli.main(["--check", str(out)]) == 1


def test_the_export_can_be_verified_without_june_s_code(db: Path, tmp_path: Path) -> None:
    """A third party should not have to import June to check her ledger.

    This reimplements the documented scheme with nothing but the standard
    library. If it ever stops matching, either the format changed silently or
    docs/product/trust-ledger-verification.md is now wrong.
    """
    import hashlib

    out = tmp_path / "chain.jsonl"
    cli.main(["--db", str(db), "--export", str(out)])

    prev = "0" * 64
    for line in out.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        material = json.dumps(
            {
                "actor": e["actor"],
                "id": e["id"],
                "kind": e["kind"],
                "payload": e["payload"],
                "prev_hash": e["prev_hash"],
                "seq": e["seq"],
                "ts": e["ts"],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        digest = hashlib.blake2b(material.encode("utf-8"), digest_size=32).hexdigest()
        assert e["prev_hash"] == prev
        assert e["entry_hash"] == digest
        prev = digest


def test_an_export_says_what_it_cannot_prove(db: Path, tmp_path: Path, capsys) -> None:
    """Honesty about the gap is part of the artifact."""
    out = tmp_path / "chain.jsonl"
    cli.main(["--db", str(db), "--export", str(out)])
    cli.main(["--check", str(out)])
    assert "tail-truncation cannot be detected" in capsys.readouterr().out


# -- usability ----------------------------------------------------------


def test_a_missing_database_is_a_clear_error(tmp_path: Path, capsys) -> None:
    assert cli.main(["--db", str(tmp_path / "nope.db")]) == 2
    assert "no ledger at" in capsys.readouterr().err


def test_contradictory_flags_are_rejected(db: Path, tmp_path: Path) -> None:
    assert cli.main(["--db", str(db), "--check", "a", "--export", "b"]) == 2


def test_an_empty_ledger_is_not_a_failure(tmp_path: Path, capsys) -> None:
    path = tmp_path / "june.db"
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()

    assert cli.main(["--db", str(path)]) == 0
    assert "empty" in capsys.readouterr().out


def test_verification_does_not_write_to_the_database(db: Path) -> None:
    """It opens read-only, so it is safe to run against a live June."""
    before = db.stat().st_mtime_ns
    cli.main(["--db", str(db)])
    assert db.stat().st_mtime_ns == before


# -- the pure function --------------------------------------------------


def test_missing_fields_fail_rather_than_verify_a_subset() -> None:
    result = verify_entries([{"seq": 1, "id": "x"}])
    assert result.ok is False


def test_an_empty_chain_verifies() -> None:
    assert verify_entries([]).ok is True
