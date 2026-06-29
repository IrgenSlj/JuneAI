# ADR 0022 — Trust Ledger: Tamper-Evident Provenance

## Status

Accepted; implementation in progress. Anchored by the Silence/Trust handoff
(`docs/product/CLAUDE_HANDOFF_silence_and_trust.md`) and the senior review it
records. Extends the guard layer (ADR 0021) and the single-engine storage
decision (ADR 0019). Implements the privacy/honesty invariants as something the
user can *prove*, not just see.

## Context

June already makes egress visible: the privacy dial, the per-turn provenance
frame, and the glass-box trace show what left the device and why. Visibility is
necessary but not sufficient for the wedge June is built for — people who may one
day need to *demonstrate* to someone else (a lawyer, an auditor, themselves)
exactly what an AI did and did not do with sensitive material. A scrollable list
of events the app could have rewritten proves nothing. What proves something is a
record that cannot be altered after the fact without the alteration being
detectable.

A cloud, engagement-funded assistant is structurally barred from offering this:
its provenance lives on the vendor's servers, under the vendor's control, and the
vendor is the party you would most want to hold accountable. A local-first
assistant can put the tamper-evident record on the user's own machine, where the
user — not June, not a vendor — is the source of truth. That asymmetry is the
point.

## Decision

A `june_brain/trust/` package maintains an **append-only, hash-chained, locally
verifiable ledger** of every cloud egress and every consequential action, stored
in the single SQLite database (ADR 0019). Four properties define it.

1. **Append-only, centrally written.** `LedgerWriter` exposes only `append`; there
   is no update or delete API surface. Entries are written at the seams that
   already see the events — cloud egress inside the provider provenance path
   (`providers/provenance.py`), gated actions/approvals at the guard execution and
   approval seams — so a skill cannot perform egress or a consequential action and
   skip the ledger. This is the same "written centrally so nothing can bypass it"
   principle as the guard's untrusted-content framing (ADR 0021).

2. **Hash chain (stdlib only).** Each entry stores
   `entry_hash = blake2b(canonical_json({seq, id, ts, kind, actor, payload, prev_hash}))`,
   where `canonical_json` is sorted-keys, no-whitespace, UTF-8. The first entry's
   `prev_hash` is `"0" * 64`; every later entry's `prev_hash` is the previous row's
   `entry_hash`. `seq` is a gap-free, monotonic device-wide counter
   (`INTEGER PRIMARY KEY AUTOINCREMENT`). The chain is device-global, not
   per-user, so it is a single verifiable history. Hashing uses stdlib `hashlib`;
   no scheme is invented.

3. **Optional Ed25519 signing.** If a device key exists, each `entry_hash` is
   signed with Ed25519 (via PyNaCl / libsodium — the sanctioned "use a vetted
   crypto library, never hand-roll" exception). The keypair is stored through the
   existing `secret_store.py` (OS keychain). Unsigned is a fully valid mode: chain
   integrity (tamper-evidence) holds without a key; signing adds authenticity
   (proof the entry came from *this* device) on top. Signing is lazily imported so
   the brain runs whether or not PyNaCl is installed.

4. **Verifiable and exportable.** `verify_chain()` recomputes every `entry_hash`,
   checks linkage, and (if signed) verifies signatures, returning
   `{ok, first_broken_seq, signed}` — `first_broken_seq` is the exact row where the
   chain first breaks, or `None` when intact. The full `trust_ledger` table is
   included in the one-command memory export so the record is verifiable offline,
   away from the app that produced it.

**Redaction guarantee.** Payloads are scrubbed by `guard/redaction.py` *before*
they are written — the same guarantee already enforced for traces. A configured
secret must never enter the ledger. Payloads record summaries (model/role, the
plain-English rationale, a byte/field summary of what left the device, the privacy
mode at the time) — never the raw content that left the device.

**Verification contract.** Tamper-evidence is the promise, not tamper-*proofing*:
a sufficiently privileged local process can rewrite the whole chain (recompute
every hash). `verify_chain` detects, and reports the first broken `seq` for:
in-place edits of any entry, reordering, a forged interior entry, and
**interior deletions** (which leave a gap in the otherwise gap-free `seq`).

**Tail-truncation** — deleting the *most recent* entries — is the one case a bare
hash chain cannot catch (rows `1..N-k` still chain validly). We close it with the
`seq` column's `AUTOINCREMENT` high-water mark, which sqlite keeps in
`sqlite_sequence` and does **not** lower when rows are deleted: if the high-water
mark exceeds the last verified `seq`, `verify_chain` reports truncation. The
residual, stated honestly: an attacker with full db write access can still
truncate the tail *and* rewrite `sqlite_sequence` to match — defeating this check.
Full truncation-resistance against such an attacker requires an external anchor
(a periodically exported/notarized head hash), which is out of scope this round
because it would add egress. Signing raises the bar on the other classes:
rewriting an entry without the device private key is detectable because the
signature no longer verifies (though it does not by itself stop truncation —
the attacker simply drops signed rows).

## Alternatives Considered

- **Rely on the existing activity log / traces.** Rejected: both are mutable
  rolling tables (the activity log prunes to 1000 rows). They answer "what
  happened recently," not "prove this was not changed."
- **Cloud-anchored or blockchain-style ledger.** Rejected: adds egress and a
  third party, contradicting local-first; the user must not have to trust a
  network service to audit their own machine. A local hash chain gives
  tamper-evidence with zero new egress.
- **Hand-rolled signing / homemade MAC.** Rejected by the no-hand-rolled-crypto
  invariant. Hashing is stdlib; signing is an established library.
- **Per-user chains.** Rejected: a single device-wide chain is simpler to verify
  (one monotonic sequence) and matches the single-machine threat model. User
  scoping lives in the payload, not the chain structure.

## Consequences

Positive: June's transparency becomes provable, not merely visible — the
differentiator a cloud competitor cannot copy. Egress and consequential actions
are auditable after the fact, offline, by the user. The guarantee is enforced in
code at central seams, so coverage does not depend on every call site
remembering.

Negative / accepted: the ledger grows unbounded by design (an audit trail you can
prune is not an audit trail) — acceptable because entries are small summaries, and
a future retention policy, if added, must itself be a ledgered, user-driven
action. Writing on the egress path adds a small synchronous SQLite insert;
best-effort error handling must never let a ledger write break a model call.
Tamper-evidence is not tamper-proofing (stated in the verification contract).
