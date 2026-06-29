# Claude Code Handoff — Silence Model & Trust Ledger

> **For:** Claude Code, working in the JuneAI monorepo.
> **Status:** v1, 2026-06-29. Implementation handoff for the two flagship
> differentiators agreed in the senior review.
> **Authority order if anything conflicts:** `docs/vision.md` (why) → `docs/product/rebuild-plan.md`
> (load-bearing decisions) → `docs/product/development-plan.md` (active sequence) → the ADRs →
> this handoff (how to build these two features). If this handoff contradicts an invariant, the
> invariant wins and you flag it.

---

## 0. What you are building and why

Two features, designed so each is a capability a cloud, engagement-funded competitor is
*structurally* barred from copying:

1. **Trust Ledger** — an append-only, hash-chained, locally-verifiable record of every cloud
   egress and every consequential action. Turns June's existing provenance/glass-box transparency
   into something the user can *prove* was not tampered with. Serves the privacy-serious wedge
   (people who may need to demonstrate what an AI did/did not do with sensitive material).

2. **Silence Model** — a real, local, inspectable decision policy for *June-initiated* surfacing
   (deadlines, found contradictions, promise nudges): surface now, batch for the next natural
   opening, or stay silent — each decision carrying a plain-English reason. Implements the fourth
   inversion ("knows when to stay quiet") as actual code and as an explicit anti-engagement engine.

These two are deliberately self-contained. The larger "organism" (belief/contradiction engine,
presence-triggered consolidation, local preference learning, Silence v2 trained model) is **out of
scope for this round** — see §6 — so this work ships and is used before the harder builds begin.

---

## 1. Invariants this work must honor (non-negotiable)

These come from the vision and ADRs. Treat any violation as a stop-and-flag.

- **No heartbeat (ADR 0016).** Nothing here may wake on a timer to scan and maybe act. The Silence
  Model's "batch" output is drained at **event boundaries only** (session open, the user shows up,
  a subscribed event changes) — never by a clock.
- **Defer, not act (inversion 1).** The Silence Model and any consolidation output produce
  *proposals / surfacings*, never silent consequential actions. The user remains the resolver.
- **Local-first, zero new egress.** Neither feature may add any network call. Both must function
  fully in `local-only` mode. The Trust Ledger and Silence policy run entirely on-device.
- **Glass-box (existing).** Every Silence decision and every ledger entry is inspectable in the UI.
- **One-loop engine (ADR 0018).** The Silence Model is a **classifier / policy function, not a
  second agent.** Do not introduce a background agent, a scheduler-driven LLM turn, or any new
  control loop. It is a synchronous decision called at the surfacing seam.
- **Honesty core (vision §4).** Reasons surfaced to the user must be truthful descriptions of the
  actual decision features, never reassuring fictions.
- **No hand-rolled cryptography (vision non-goals).** The ledger hash chain uses stdlib `hashlib`;
  optional signing uses an established library (PyNaCl / libsodium). Do **not** invent a scheme.
- **Ship small slices; keep the gate green.** `./tools/check.sh` (pytest + svelte-check + OpenAPI
  drift) passes at the end of every slice. Commit and push per slice. Regenerate the typed client
  with `./tools/codegen.sh` whenever a Pydantic schema or route changes.
- **Tests are part of the change.** Match the repo's existing discipline (≈1:1.6 source:test). Every
  behavior gets a test; the brain modules must be testable without Ollama via injectable seams.

---

## 2. ADRs to write first (slice 1, docs only)

- **ADR 0022 — Trust Ledger (tamper-evident provenance).** Decision: append-only hash-chained
  local ledger of consequential events; stdlib hashing + optional Ed25519 device signature; lives in
  the single SQLite db (ADR 0019); written centrally so no skill can bypass it (same principle as
  guard framing). Record the redaction guarantee and the verification contract.
- **ADR 0023 — Silence Model (surface-vs-defer policy).** Decision: a local, rules-first decision
  function governing **June-initiated surfacing only**, never responses to direct user input; output
  is `now | batch | suppress` + a reason + features; batch drained at event boundaries (consistent
  with ADR 0016); v1 is rules, v2 (future) is a local trained model over the feedback log; the policy
  optimizes for appropriate restraint, explicitly not engagement. Note the relationship to D.2
  proactivity in the roadmap.

---

## 3. Feature 1 — Trust Ledger

### 3.1 Storage (slice 2)
New table in the existing single SQLite db under `JUNE_DATA_DIR` (do not add a new store; ADR 0019).

```
trust_ledger(
  seq         INTEGER PRIMARY KEY AUTOINCREMENT,  -- monotonic, gap-free per db
  id          TEXT NOT NULL,                       -- uuid4
  ts          TEXT NOT NULL,                        -- ISO-8601 UTC
  kind        TEXT NOT NULL,                         -- egress | action | approval | system
  actor       TEXT NOT NULL,                          -- june | user
  payload     TEXT NOT NULL,                           -- canonical JSON, ALREADY redacted
  prev_hash   TEXT NOT NULL,                            -- hex; genesis = 64 * "0"
  entry_hash  TEXT NOT NULL,                             -- see 3.3
  sig         TEXT                                       -- optional Ed25519 hex; NULL if unsigned
)
```

Implement as `packages/brain/src/june_brain/trust/ledger.py` with a `LedgerWriter` (append-only;
no update/delete API surface) and a `LedgerReader` (paginated by `seq`). Place under a new
`june_brain/trust/` package.

### 3.2 What gets recorded, and where it is wired (slice 3)
Append **centrally**, at the seams that already see these events, so a skill cannot bypass the
ledger:
- **Cloud egress** — in the provider provenance path (`providers/provenance.py`): one entry per
  cloud-routed model call and per external service call. Payload: model/role, the
  plain-English rationale already produced, byte/field summary of what left the device (not the raw
  content), the privacy-dial mode at the time.
- **Consequential actions / approvals** — in the guard action path (`guard/actions.py`): one entry
  when a gated action (`write_network`, `execute`, tainted `read_network`) is approved and executed,
  recording the action class and the user's approval.
- Reuse **`guard/redaction.py`** to scrub the payload *before* it is written. A secret must never
  enter the ledger (same guarantee already enforced for traces).

### 3.3 Hash chain + verification
- `entry_hash = hashlib.blake2b(canonical_json({seq, id, ts, kind, actor, payload, prev_hash})).hexdigest()`
  where `canonical_json` is sorted-keys, no-whitespace, UTF-8.
- `prev_hash` of the first entry is `"0"*64`. Each subsequent entry's `prev_hash` is the previous
  row's `entry_hash`.
- **Optional signing:** if a device key exists, sign `entry_hash` with Ed25519 (PyNaCl). Store the
  device public key and private key via the existing `secret_store.py` (OS keychain). Unsigned is a
  valid mode (chain integrity still holds); signing adds authenticity on top.
- `verify_chain() -> {ok: bool, first_broken_seq: int | None, signed: bool}` recomputes every
  `entry_hash`, checks linkage, and (if signed) verifies signatures. Returns the first seq where the
  chain breaks, or `ok=True`.

### 3.4 Export
Include the full `trust_ledger` in the existing one-command export so it is verifiable offline.

### 3.5 API (slice 4) — `packages/api`
Pydantic schemas are the source of truth; run `./tools/codegen.sh` after.
- `GET /system/ledger?cursor=<seq>&limit=<n>` → page of receipts (newest-first), with cursor.
- `POST /system/ledger/verify` → `{ok, first_broken_seq, signed}`.
- Extend the existing `/system` (Trust) payload with a `ledger_summary` (count, last_entry_ts,
  `egress_today` count, `chain_verified_at`).

### 3.6 Tests (with each slice)
- Append produces a correct, gap-free chain; genesis `prev_hash` is zeros.
- Tamper test: mutate a payload row directly → `verify_chain` returns `ok=False` with the correct
  `first_broken_seq`.
- Redaction test: a configured secret present in an event payload never appears in the stored row.
- Bypass test: a tool/skill that performs egress routes through the central append (assert an entry
  exists after a simulated cloud call).
- Signing round-trip when a device key is present; unsigned path when it is absent.

### 3.7 Acceptance
Local-only mode produces no `egress` entries; cloud calls produce exactly one each; the chain
verifies; tampering is detected at the right point; no secret is ever persisted; the ledger appears
in export.

---

## 4. Feature 2 — Silence Model (v1, rules)

### 4.1 Scope guard — read this twice
The Silence Model governs **only June-initiated surfacing**. It must **never** intercept, delay, or
suppress June's response to a direct user message. June always answers when spoken to. The policy is
called at the surfacing seam (deadline fired, contradiction found by a future consolidation pass,
promise needs a nudge), not in the reply path. State this as a guard assertion in code.

### 4.2 The decision function (slice 5)
`packages/brain/src/june_brain/silence/policy.py`:

```python
SurfacingAction = Literal["now", "batch", "suppress"]

@dataclass
class SurfacingDecision:
    action: SurfacingAction
    reason: str               # plain-English, truthful (honesty core)
    features: dict[str, Any]  # the inputs that produced the decision (for /system + v2 training)

def decide(candidate: SurfacingCandidate, ctx: SurfacingContext) -> SurfacingDecision: ...
```

**Features** (all locally available; no egress): `candidate.salience`, `candidate.kind`
(deadline | contradiction | promise_nudge | …), `deadline_delta` (time to a hard deadline, if any),
`dismissals_for_similar` (from `surfacing_decisions` history), `presence_state` (present-active /
present-idle / absent — derived from existing session/activity signals, **not** a timer),
`active_thread_open` (is the user mid-task), `local_time_bucket`.

**v1 policy = transparent rules** (good on day one, solves cold-start; v2 trained model comes later
and falls back to these rules):
- Hard deadline within the urgent window → `now` ("deadline in 2h").
- High salience + user present-idle + no recent dismissal of similar → `now`.
- Anything dismissed ≥2× recently → `suppress` ("you dismissed similar twice").
- User present-active / mid-task, non-urgent → `batch` ("you're mid-task; held for later").
- Default for non-urgent, low-salience → `batch`.
The reason string is always the actual deciding feature(s).

### 4.3 Storage + batching (slice 6)
- Table `surfacing_decisions(id, candidate_id, kind, action, reason, features TEXT, ts, outcome
  TEXT NULL)`. `outcome` is filled later (engaged | dismissed | expired) and is the v2 training
  signal.
- **Batched** items accumulate into a digest. The digest is **drained at event boundaries only**
  (session open / user shows up) — implement the drain as a function the API/session-open path calls,
  **never** as a scheduled job (ADR 0016). Surfaced digest items become the home "what I held for
  you" list (§5).
- Each decision is also a Trust Ledger entry (`kind="action"`, `actor="june"`) — June's restraint is
  itself auditable.
- **Hard deadlines remain OS notifications** (per roadmap D.2); the Silence Model decides *in-app*
  surfacing vs batch, it does not replace the OS-notification path for true deadlines.

### 4.4 API (slice 7)
- `GET /system/surfacing?cursor=&limit=` → recent decisions with reasons (the "why June stayed
  quiet" surface).
- `POST /system/surfacing/{id}/feedback` body `{verdict: "good" | "bad"}` → records user judgment of
  the decision (feeds v2; also adjusts `dismissals_for_similar` signal).
- Run `./tools/codegen.sh`.

### 4.5 Tests
- Each rule branch produces the expected action + a reason naming the deciding feature.
- **Invariant test:** a direct user message is never routed through `decide` (the reply path has no
  dependency on the silence policy).
- Batching accumulates and the digest drains only when the session-open/drain function is called,
  never on a timer (assert no scheduler registration).
- Feedback writes `outcome`/verdict and influences subsequent `dismissals_for_similar`.

### 4.6 Acceptance
June surfaces proactively only when the rules say `now`; batched items appear at the next opening
with truthful one-line reasons; suppressed items are visible in `/system` but never pushed; direct
replies are unaffected; no timer/heartbeat is introduced; each decision is in the ledger.

---

## 5. Connective tissue — control-room home data (slice 8)

The home screen ("what June is holding") is the surface where both features become visible and
valuable. Backend only in this round (UI is slices 9–10, after design):
- `GET /home/holdings` → `{ open_promises, waiting_on_user, blocked_by_local_only,
  next_deadline, held_digest (batched surfacings), egress_today (from ledger), chain_verified }`.
- This is read-only aggregation over existing promise state + the Silence digest + the ledger
  summary. No new state.

---

## 6. Explicitly OUT of scope this round (future, separate plans)

Do not build these now; they depend on the above being shipped and used:
- Belief / contradiction engine (memory belief-state, contradiction surfacing).
- Presence-triggered consolidation ("reflection on arrival", local sleep-time-style passes).
- Local preference learning (closing the feedback → adaptation loop).
- Silence Model v2 (local trained classifier over `surfacing_decisions`).
When this round is green and dogfooded, request the next handoff.

---

## 7. Slice plan (each ends gate-green, committed)

1. ADR 0022 + ADR 0023 (docs).
2. Trust Ledger storage + `LedgerWriter`/`LedgerReader` + hash chain + `verify_chain` + tests.
3. Central wiring of ledger into `providers/provenance.py` + `guard/actions.py`, with
   `guard/redaction.py` reuse + tests (incl. tamper + bypass + redaction).
4. Ledger API (`/system/ledger`, `/system/ledger/verify`, `/system` summary) + schemas + codegen +
   tests.
5. Silence `policy.py` (rules) + features + `SurfacingDecision` + tests (incl. the direct-reply
   invariant).
6. Silence storage + batching + event-boundary drain + ledger entries for decisions + tests.
7. Silence API (`/system/surfacing`, feedback) + schemas + codegen + tests.
8. `/home/holdings` aggregation API + schemas + codegen + tests.
9. **UI (after design):** `/system` Trust — receipts list + verify affordance + surfacing
   decisions explainer. Implement against the approved Claude Design output.
10. **UI (after design):** control-room home consuming `/home/holdings`. Implement against the
    approved Claude Design output.

Slices 1–8 are backend and can proceed immediately. Slices 9–10 wait on the Design brief
(`docs/product/CLAUDE_DESIGN_BRIEF_silence_and_trust.md`).

## 8. Definition of done

`./tools/check.sh` green; every slice committed; ADRs 0022/0023 merged; the ledger is verifiable and
demonstrably tamper-evident (passing red-team test) and contains no secrets; the Silence Model
produces inspectable, truthful reasons, never blocks a direct reply, and introduces no timer; the
home exposes holdings + held digest + egress-today + chain-verified; **local-only mode is unchanged
and no new egress exists anywhere in this work**; export includes the ledger.
