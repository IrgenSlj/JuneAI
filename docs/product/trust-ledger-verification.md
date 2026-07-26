# Verifying June's Trust Ledger

June's claim is that she can prove what she did. This page is how you check it
without taking her word for anything — including without running her code.

## The quick version

```
june-verify                          # verify the ledger on this machine
june-verify --json                   # machine-readable, for scripts and CI
june-verify --export chain.jsonl     # hand the chain to someone else
june-verify --check chain.jsonl      # verify an export, no database needed
```

Exit codes: `0` intact, `1` broken, `2` a usable error. It opens the database
read-only, so it is safe to run while June is open.

Intact:

```
Ledger     ~/Library/Application Support/June/june.db
Entries    4  (action 1, approval 1, egress 1, system 1)
Range      2026-07-26T18:21:11+00:00  ->  2026-07-26T18:21:11+00:00
Integrity  unsigned (hash chain only)
Head       88bf617f4861f805...

OK — every entry hashes to its stored value and links to the one before it.
```

Tampered with:

```
BROKEN — the chain first fails at entry 2. Entries before it are intact;
that one was altered, removed, or inserted.
```

## What is in the chain

| Kind | What it records |
|---|---|
| `egress` | A call that left the device: which provider, how many tokens |
| `action` | A consequential action June took: which tool, which action class, whether the arguments were tainted |
| `approval` | The user approving or refusing something |
| `system` | Guard events — including an injection detection: which signals fired, never the content |
| `mcp_access` | Another program reading memory over MCP: which client, which tool, how many facts |

Payloads are redacted before they are written, centrally in the writer, so "no
secret enters the ledger" does not depend on any caller remembering.

## The scheme

Short on purpose. You should be able to reimplement it in an afternoon, in any
language, and that is the only reason to believe the verifier.

```
entry_hash = blake2b_256(canonical_json({
    actor, id, kind, payload, prev_hash, seq, ts
}))
```

- `canonical_json` is sorted keys, no whitespace (`,`/`:` separators), UTF-8.
- `payload` is the **stored string**, not a re-parsed object. The hash commits
  to those exact bytes; re-serialising to equivalent JSON produces a different
  digest.
- `prev_hash` is the previous entry's `entry_hash`. The first entry's is 64
  zeros.
- `seq` starts at 1 and increments by 1.

Optionally each `entry_hash` carries an Ed25519 signature in `sig`. Unsigned is
a valid mode: the hash chain gives tamper-evidence with no key, and the
signature adds authenticity on top.

Twelve lines of Python, standard library only:

```python
import hashlib, json

prev = "0" * 64
for line in open("chain.jsonl"):
    e = json.loads(line)
    material = json.dumps(
        {k: e[k] for k in ("actor", "id", "kind", "payload", "prev_hash", "seq", "ts")},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    digest = hashlib.blake2b(material.encode(), digest_size=32).hexdigest()
    assert e["prev_hash"] == prev, f"broken link at {e['seq']}"
    assert e["entry_hash"] == digest, f"altered entry at {e['seq']}"
    prev = digest
```

That snippet is not decoration. It is asserted in
`packages/brain/tests/unit_tests/test_trust_verify_cli.py` against a real
export, so if the format ever changes silently, the test fails and this page is
known to be wrong.

## What verification proves, and what it does not

Named plainly, because a verification page that only lists its strengths is
advertising.

**It detects:**

- An altered payload, timestamp, kind or actor on any entry.
- An entry deleted from the middle, or inserted anywhere.
- Reordering.
- Entries removed from the **tail** — but only against the database, not an
  export. A chain with its last k entries deleted is a perfectly valid chain of
  length N−k. What gives it away is sqlite's `AUTOINCREMENT` high-water mark,
  which does not decrease when rows are deleted. `--check` on an export says so
  in its output rather than quietly checking less.

**It does not detect:**

- **A rewritten chain.** Someone with write access to the file and this document
  can delete entries and recompute every hash after them. The chain is
  tamper-*evident*, not tamper-*proof*. Ed25519 signing raises the bar to
  needing the device key; nothing here is stronger than the OS account.
- **Something that was never recorded.** The ledger proves the integrity of what
  was written, never the completeness of it. Completeness rests on the write
  seams being the only paths to egress and consequential action — a code
  property, checkable by reading `loop/wiring.py` and `mcp/server.py`, not by
  this command.
- **That the payload is true.** It proves the row has not changed since it was
  written, not that it described reality when it was.
- **Tail truncation combined with a rewritten `sqlite_sequence`.** Recorded in
  ADR 0022's verification contract.

The honest summary: this makes silent revision of history impossible and
detectable revision cheap to find. It does not make June's ledger evidence
against a determined attacker who already owns the machine — and no local-first
system's does.

## Exporting for someone else

`--export` writes JSONL, one entry per line, in chain order, with `payload` kept
as its exact stored string. Anyone can verify it with the snippet above and
nothing else. The export contains real ledger payloads — redacted of secrets,
but still a record of what June did — so treat it as you would a log file.

## Related

- [`../decisions/0022-trust-ledger.md`](../decisions/0022-trust-ledger.md) — the decision and its contract
- [`injection-benchmark.md`](injection-benchmark.md) — what the `system` guard entries mean
- `packages/brain/src/june_brain/trust/verify.py` — the algorithm, shared by all three callers
