# JUNE_V02_BRIEF.md — JuneAI Next Phase Development Brief

> **SUPERSEDED (2026-07-24).** The lead plan is
> [`docs/product/v0.3-development-plan.md`](docs/product/v0.3-development-plan.md)
> (the what and why) with
> [`docs/product/v0.3-execution-plan.md`](docs/product/v0.3-execution-plan.md)
> (the order and acceptance criteria). Current state of the project is
> [`docs/CURRENT.md`](docs/CURRENT.md). This file is kept in place, not moved,
> because append-only ADRs reference it by path — see
> [`docs/archive/README.md`](docs/archive/README.md) for that convention.
> **Do not follow this document for new work.** Several of its workstreams
> (W2 retrieval, the egress gate) shipped; others were dropped.

**Version:** 1.0 (LOCKED)
**Date:** 2026-07-06
**Author:** Irgen Salianji (strategy co-developed with Claude)
**Audience:** Claude Code (implementation agent)
**Supersedes:** Open items in `JUNE_REBUILD_PLAN.md` (S0–S13) that remain unfinished. This brief does NOT replace completed rebuild work; it builds on it.
**Repo:** https://github.com/IrgenSlj/JuneAI

---

## 0. How to use this brief (read first)

1. **Reconcile before you build.** This brief was written against the repo state as of early July 2026. Your FIRST task (W0.1) is to audit the actual repo state against the assumptions in §2 and produce a `RECONCILIATION.md` diff. Where reality differs from an assumption, flag it and adapt the spec — do not silently build against a wrong assumption.
2. **One workstream per session where possible.** Workstreams W0–W7 are ordered by dependency. Do not start W4 (Night Shift) before W2 (Retrieval v2) and W3 (Provenance) schemas are merged, because Night Shift writes through both.
3. **Everything through the single gate.** All work must pass the existing unified test/lint/type gate. No workstream is "done" until its acceptance criteria (each §'s final block) pass in CI.
4. **ADR discipline.** Each workstream that changes architecture requires an ADR (numbered continuation of the existing ADR sequence). Draft the ADR *before* implementation; treat this brief as the ADR's context section.
5. **No new heavyweight dependencies** without an explicit ADR justifying them. Preference order: stdlib > existing deps > small single-purpose libs > nothing else. SQLite-native solutions (FTS5, triggers, JSON1) are always preferred over new services.
6. **Founder-gated items** are marked `[FOUNDER]`. Stop and ask rather than deciding these yourself.

---

## 1. Strategic frame (why this phase exists)

June's thesis — governed autonomy, auditable memory, local-first — was validated the hard way in H1 2026 by the OpenClaw security crisis: the fastest-growing repo in GitHub history shipped ungoverned personal-agent autonomy and produced a one-click RCE (CVE-2026-25253), tens of thousands of exposed instances, 800+ malicious marketplace skills, and a new attack class (web-content prompt injection that *poisons persistent memory*). The market now contains millions of people who want a local personal AI with memory, and who have just learned what happens without a trust layer.

June v0.2's job is to convert June's existing trust primitives (Trust Ledger, guard layer, Silence Model) into **visible, demonstrable product features**, close the retrieval-quality gap against the 2026 memory-system state of the art (multi-signal retrieval, temporal validity, sleep-time consolidation), and remove the two biggest funnel killers (unsigned binary, multi-GB cold start).

**Positioning sentence (use in README/landing):** *"A personal AI you can audit. June remembers you, forgets gracefully, explains every action, and never phones home."*

The four inversions remain the north star: verification→deference, completion→continuity, accumulation→graceful forgetting, speed→good timing. Every feature below maps to at least one inversion; if a proposed implementation detail fights an inversion, the inversion wins.

---

## 2. Assumed current state (verify in W0.1)

- Tauri desktop shell (macOS Apple Silicon), Python backend sidecar supervised by the shell; watchdog + corrupt-DB recovery in place.
- Storage: single SQLite file; sqlite-vec for embeddings; graph-style entity/relation tables; memory facade module is the single entry point for all memory reads/writes.
- Trust Ledger: append-only, hash-chained event log with UI surface ("Trust" screen).
- Guard layer: policy checks on tool/skill execution paths.
- Silence Model: notification/proactivity governor.
- Durable promises: resumable long-running tasks.
- Model roster: local Ollama (Gemma 4 tiers incl. `gemma4:e4b` class) + Gemini cloud tier; explicit non-goal of a third provider. Roster selection logic exists.
- Tests: 763+ backend tests behind one gate; Playwright UI coverage largely absent (known standing TODO).
- Packaging: `June.app` + DMG build proven; CI signing workflow exists but degrades gracefully without secrets (Apple Developer enrollment pending `[FOUNDER]`).
- Licensing: offline Ed25519 entitlement core present, dormant (empty `PUBLIC_KEYS`).
- Docs: extensive `docs/product/` tree; README known-stale (still references ChromaDB/LangGraph-era architecture).

If any of the above is materially wrong, record it in `RECONCILIATION.md` and adjust downstream specs accordingly.

---

## 3. Non-goals for v0.2 (carried over + new)

- No third model provider. No OpenAI/Anthropic/Mistral integration paths.
- No cloud sync, no hosted relay, no server-side anything operated by us.
- No skill/plugin marketplace. No remote code execution surface of any kind.
- No messenger bridge (Signal/Telegram/WhatsApp) in v0.2. Documented as v0.3+ candidate only (see §12).
- No payments wiring. Licensing core stays dormant.
- No Windows/Linux builds this phase (keep code portable; do not add mac-only assumptions below the shell layer without need).
- No Privacy Mode 2/3 (encrypted backup, Google skills) — parked pending founder ADRs.

---

## 4. Workstream overview & sequencing

| WS | Name | Depends on | Gates release? |
|----|------|-----------|----------------|
| W0 | Repo reconciliation, doc consolidation, README rewrite | — | YES |
| W1 | Release engineering: signing, notarization, clean-machine funnel | W0 | YES |
| W2 | Retrieval v2: four-signal fusion + temporal validity + benchmark harness | W0 | YES |
| W3 | Memory provenance & quarantine | W2 schemas | YES |
| W4 | Night Shift: auditable sleep-time consolidation | W2, W3 | YES (flagship) |
| W5 | Apple Foundation Models instant tier (spike → feature-flagged) | W0 | NO (flag-gated) |
| W6 | Opt-in health telemetry + feedback channel | W1 | YES |
| W7 | Playwright UI regression slices 1–3 | W0 | YES |
| W8 | Local voice capture (stretch) | W5 outcome | NO |

Release target: **v0.2.0** = W0+W1+W2+W3+W4+W6+W7 complete. W5/W8 ship dark behind flags if ready.

---
## W0 — Repo reconciliation, doc consolidation, README rewrite

### W0.1 Reconciliation
- Audit repo against §2. Output `docs/RECONCILIATION.md`: table of {assumption, actual, impact on this brief}.
- Inventory all planning docs under `docs/`. Classify each as CURRENT / SUPERSEDED / HISTORICAL.

### W0.2 Doc consolidation
- Create `docs/CURRENT.md`: the single authoritative "state of the project" page — architecture summary (one paragraph per subsystem), active plan (link to this brief), decision log pointer (ADR index), release status.
- Move SUPERSEDED plans to `docs/archive/` with a one-line tombstone at their old path is NOT needed — just move them and fix inbound links. Rationale: stale plans are context pollution for agent sessions.
- ADR index: generate/refresh `docs/adr/INDEX.md` listing every ADR with status (accepted/superseded).

### W0.3 README rewrite (this is a product surface, treat it as one)
Structure, in order:
1. One-line positioning (from §1) + hero screenshot/GIF of the Trust screen and a chat exchange.
2. "Why June" — 4 short paragraphs mapped to the four inversions, written for a smart non-expert. No architecture jargon in this section.
3. "What makes it different" — Trust Ledger (every action, hash-chained, inspectable), Silence Model (speaks only when worth it), graceful forgetting (Night Shift, once W4 ships), provably local (loopback-only, no telemetry by default).
4. Quickstart: current real install path (DMG download once W1 lands; dev setup otherwise). Must be executable by a stranger on a clean machine.
5. Architecture diagram (mermaid): Tauri shell → sidecar → memory facade → SQLite(vec+FTS5+graph) / ledger / models(Ollama, Gemini opt-in).
6. Accurate model roster + hardware guidance (8/16/32GB tiers).
7. Security posture: threat model link, responsible disclosure, what June never does.
8. Status/roadmap: honest v0.2 state, link to CURRENT.md.
- Delete every reference to ChromaDB, LangGraph, and superseded route names. Grep for them; CI check (W0.4) keeps them out.

### W0.4 Doc hygiene CI check
- Add a CI step: fail if README or `docs/CURRENT.md` contains banned stale tokens (`chromadb`, `langgraph`, old route names — populate list during W0.1).

### Acceptance criteria
- `RECONCILIATION.md` and `CURRENT.md` exist and are accurate.
- README passes the "stranger test": a fresh reader can state what June is, why it's different, and install it, in <5 minutes.
- Stale-token CI check green.

---

## W1 — Release engineering

### W1.1 Signing & notarization `[FOUNDER: Apple Developer enrollment]`
- Once enrollment completes: wire Developer ID Application cert + notarytool into the existing CI workflow (secrets via GitHub Actions encrypted secrets; never in repo).
- Produce: signed, notarized, stapled `June.dmg` on every tag matching `v*`.
- Verify Gatekeeper: `spctl -a -vv June.app` clean; first-launch has no unidentified-developer dialog and no multi-second Gatekeeper stall (re-run the cold-start profile; record numbers in the release notes).

### W1.2 Versioning & release process
- Adopt semver with `v0.2.0-alpha.N` prereleases. Single source of version truth (one file), propagated to Tauri config, sidecar `--version`, and About screen at build time.
- `RELEASING.md`: exact tag→CI→artifact→GitHub Release checklist, including rollback note.
- Release artifact set: DMG, SHA256SUMS, and a detached signature of SHA256SUMS (minisign or ssh-keygen -Y; pick one, document verification command in README).

### W1.3 Clean-machine funnel test
- Scripted checklist executed on a Mac that has never run the toolchain: download DMG → drag install → first run → onboarding → first model download → first chat → first Trust screen visit. Record timings and every friction point into `docs/funnel-test-YYYYMMDD.md`.
- Fix all P0 friction (crashes, dead ends, >10s unexplained waits) before v0.2.0.

### W1.4 In-app update check (no auto-update this phase)
- On launch, at most once/24h, fetch latest release tag from GitHub Releases API (this is the ONE permitted network call outside model providers; it must be listed in the Trust screen network disclosure and be disableable in settings). Show non-nagging "update available" affordance. No download/exec — link out only.

### Acceptance criteria
- Tagged build produces a notarized DMG that installs and launches cleanly on a clean machine with zero security dialogs.
- Cold-start time recorded pre/post signing; regression budget: first interactive paint of shell <3s on M1/16GB.
- Update check visible in network disclosure, off-switch works, covered by tests.

---

## W2 — Retrieval v2: four-signal fusion, temporal validity, benchmarks

**ADR required:** "Retrieval v2: multi-signal fusion and bi-temporal facts."

### W2.1 Schema — temporal validity (bi-temporal facts)
Extend the memory fact store (adapt names to the actual schema found in W0.1):

```sql
ALTER TABLE memories ADD COLUMN valid_from TEXT;        -- ISO8601, when the fact became true in the world (nullable)
ALTER TABLE memories ADD COLUMN valid_to TEXT;          -- ISO8601, when it stopped being true (nullable = still valid)
ALTER TABLE memories ADD COLUMN observed_at TEXT NOT NULL DEFAULT (datetime('now')); -- when June learned it
ALTER TABLE memories ADD COLUMN superseded_by INTEGER REFERENCES memories(id);       -- forward pointer on contradiction
CREATE INDEX idx_memories_validity ON memories(valid_to, valid_from);
```

Rules:
- Never hard-delete a superseded fact; set `valid_to` + `superseded_by`. (Ledger/audit depends on this; Night Shift's FORGET op is the only sanctioned removal path and it, too, tombstones — see W4.)
- Retrieval default: only currently-valid facts. Temporal queries ("what did I think in March") may include superseded rows, clearly marked.
- Relative-date absolutization: at write time, resolve "yesterday/last week" to absolute dates using the session timestamp. Store absolute only.

### W2.2 FTS5 keyword channel
```sql
CREATE VIRTUAL TABLE memories_fts USING fts5(
  content, tokenize='unicode61 remove_diacritics 2',
  content='memories', content_rowid='id'
);
-- Triggers: AFTER INSERT/UPDATE/DELETE on memories keep memories_fts in sync.
```
- BM25 scores via `bm25(memories_fts)`. Multilingual note: user operates in EN/NL/GR — unicode61 is adequate; do NOT add language-specific stemmers this phase.

### W2.3 Fusion scoring
Four signals per query:
1. `s_vec`: cosine similarity from sqlite-vec (existing).
2. `s_bm25`: normalized BM25 (min-max over candidate set; note SQLite bm25 is lower-is-better — invert).
3. `s_entity`: entity overlap boost — extract entities from the query with the existing local pipeline; +boost per matched entity edge in the graph tables.
4. `s_time`: recency/validity prior — currently-valid facts get 1.0; superseded decay by half-life (default 90 days from `valid_to`), floor 0.1.

Fusion: Reciprocal Rank Fusion across the vec and bm25 ranked lists (k=60), then multiply by `(1 + w_e * s_entity)` and `s_time`. Defaults: `w_e = 0.15`. All weights in one config dataclass `RetrievalConfig`, overridable via settings file, logged (values only) at startup. Candidate pool: top-50 per channel → fuse → return top-k (default 8).

Implementation constraints:
- Single facade function signature change only; all callers go through the facade. No caller may query FTS/vec tables directly (add a lint/test guarding this).
- Latency budget: p95 retrieval <120ms on M1/16GB with 50k memories. Add a perf test with a synthetic 50k corpus fixture.

### W2.4 Benchmark harness
- `benchmarks/` package, runnable via `make bench-memory`:
  - LongMemEval subset runner and LoCoMo runner (download scripts with checksums; datasets are NOT vendored into the repo).
  - Adapter that pipes benchmark conversations through the real memory facade (ingest → retrieve → answer with the local deep-tier model).
  - Output: JSON results + markdown summary table into `benchmarks/results/DATE/`.
- Publish honest numbers in README once stable. `[FOUNDER]` reviews before publishing.

### Acceptance criteria
- Migration runs on an existing v0.1 database without data loss (test fixture: real-shaped seeded DB).
- Fusion retrieval beats vec-only on the internal eval set (create a 100-case golden retrieval test: query → expected memory ids; ≥10% recall@8 improvement required).
- Perf budget met; golden tests + migration tests in the single gate.

---

## W3 — Memory provenance & quarantine

**ADR required:** "Provenance-gated memory writes."
**Threat driver:** web-content prompt injection that poisons persistent memory (ClawJacked-class). June must make memory poisoning structurally detectable and reversible.

### W3.1 Schema
```sql
ALTER TABLE memories ADD COLUMN provenance TEXT NOT NULL DEFAULT 'unknown';
-- enum: 'user_direct'      (typed/spoken by user)
--       'user_document'    (file the user explicitly imported)
--       'assistant_inferred' (June's own inference/consolidation output)
--       'web_derived'      (any content fetched from the network)
--       'import_bootstrap' (ChatGPT/Claude export importers)
ALTER TABLE memories ADD COLUMN trust_level INTEGER NOT NULL DEFAULT 0;
-- 3=user_direct, 2=user_document/import_bootstrap, 1=assistant_inferred, 0=web_derived/unknown
ALTER TABLE memories ADD COLUMN quarantined INTEGER NOT NULL DEFAULT 0; -- boolean
ALTER TABLE memories ADD COLUMN source_ref TEXT; -- URL, file path hash, session id — enough to audit, not to exfiltrate
```

### W3.2 Write-path rules (enforced in the facade, tested exhaustively)
1. Every write MUST carry provenance; facade rejects writes without it (no default-by-omission in new code; the SQL default exists only for migration).
2. `web_derived` writes land with `quarantined=1`. Quarantined memories:
   - are NEVER injected into prompts/context,
   - are NEVER used by Night Shift consolidation as sources,
   - appear in a "Pending review" section of the memory UI with source_ref, Confirm/Discard actions.
3. Confirm → `quarantined=0`, `trust_level` promoted to 2, ledger event `MEMORY_CONFIRMED`. Discard → tombstone + ledger event.
4. Instruction-shaped content filter: before any `web_derived` write, run the guard layer's injection heuristics (imperative-instruction patterns, "ignore previous", tool-invocation phrasing). Positive hits are BLOCKED from memory entirely (not just quarantined) + ledger event `MEMORY_WRITE_BLOCKED{reason}`.
5. Trust-weighted retrieval: `s_final *= trust_multiplier` (3→1.0, 2→0.95, 1→0.85, 0→never retrieved while quarantined).

### W3.3 Ledger integration
New ledger event types (extend existing event enum + hash chain unchanged):
`MEMORY_WRITE{provenance}`, `MEMORY_CONFIRMED`, `MEMORY_DISCARDED`, `MEMORY_WRITE_BLOCKED{reason}`, `MEMORY_SUPERSEDED{old,new}`.
Trust screen: add a "Memory provenance" filter view — count by provenance, list of quarantined items, one-click jump to review.

### W3.4 Migration
- Existing memories: infer provenance where cheaply possible (session-origin metadata), else `unknown` at trust 1 (NOT 0 — don't nuke existing users' retrieval quality). Document the inference rules in the ADR.

### Acceptance criteria
- Property test: no code path can insert a prompt-context memory with `quarantined=1` (fuzz the facade).
- Injection corpus test: ≥20 crafted poisoning payloads (build fixture set) — 100% blocked or quarantined, 0 reach context.
- UI review flow covered by a Playwright slice (W7).

---

## W4 — Night Shift: auditable sleep-time consolidation (flagship)

**ADR required:** "Night Shift: ledgered offline memory consolidation."
**Inversion mapping:** accumulation→graceful forgetting, speed→good timing.
**Differentiator:** every consolidation system in the field (Letta sleep-time agents, Anthropic Dreams, Claude Code AutoDream) is a black box. June's is fully audited: every operation is a hash-chained ledger event, and the user gets a morning report they can drill into and reverse.

### W4.1 Trigger conditions (all must hold; evaluated by the sidecar scheduler)
- Machine on AC power OR battery >50% `[configurable]`
- No user interaction with June for ≥30 min
- System idle (no active June chat session; respect macOS user-activity where accessible from the sidecar without new entitlements — if not cleanly accessible, June-app-idle is sufficient this phase)
- ≥8h since last completed run `[configurable: night_shift.min_interval_hours]`
- ≥N new/changed memories since last run (default N=10) — don't churn on empty days
- Local model available (Ollash reachable, deep-tier model pulled). Night Shift NEVER uses the cloud tier. Hard rule, tested.
- Kill switch: `night_shift.enabled` (default ON after onboarding opt-in screen; `[FOUNDER]` may flip default)

### W4.2 Pipeline (each stage resumable via existing durable-promise machinery)
Stage the run as a durable promise `night_shift_run{id}` with per-stage checkpoints:

1. **SELECT** — build the working set: memories with `observed_at` or modified since last run, plus their graph neighborhoods (1-hop), EXCLUDING quarantined and tombstoned rows. Cap working set (default 500 memories) — overflow deferred to next run, oldest-first.
2. **DEDUPE/MERGE** — cluster near-duplicates (vec similarity >0.92 AND same entity set). For each cluster, the local model writes one consolidated memory (`provenance='assistant_inferred'`, `trust_level` = min of sources); sources get `valid_to=now`, `superseded_by=new_id`. Op: `MERGE{source_ids[], new_id}`.
3. **CONTRADICTION SCAN** — for entity-sharing memory pairs with opposing predicates (model-judged, conservative prompt: only flag CLEAR contradictions), keep the newer-observed as valid, supersede the older. Never auto-resolve user_direct vs user_direct contradictions — those go to the morning report as questions. Op: `CONTRADICT{kept_id, superseded_id}` or `CONTRADICT_ASK{id_a, id_b}`.
4. **DATE ABSOLUTIZATION** — sweep for remaining relative-date strings in memory content; rewrite to absolute using observed_at. Op: `ABSOLUTIZE{id}`.
5. **DECAY/FORGET** — score each memory: retrieval_count (add a lightweight counter to the facade read path if absent), recency, trust_level, graph degree. Below threshold → propose forgetting. FORGET is two-phase: run K marks `forget_proposed`; if not retrieved and not user-rescued by run K+2, tombstone (content preserved in an encrypted-at-rest coldstore table `memories_cold`, excluded from all retrieval; restorable from morning report for 90 days, then eligible for hard purge `[configurable]`). Ops: `FORGET_PROPOSE{id}`, `FORGET_COMMIT{id}`, `FORGET_RESCUE{id}`.
6. **PROMOTE/THEME** — recurring entities/topics across ≥3 recent memories get a synthesized theme memory ("Irgen has mentioned the Varna library project in 4 sessions this week") with `provenance='assistant_inferred'`. Op: `PROMOTE{new_id, evidence_ids[]}`.
7. **REPORT** — write `night_shift_reports` row: run id, started/ended, counts per op, token usage, model used, list of ops with human-readable one-liners, open questions from CONTRADICT_ASK.

Budgets: max wall-clock 20 min/run; max tokens per run `[configurable, default 150k]`; abort cleanly at budget → checkpoint → resume next window. Thermal courtesy: run model calls at low priority.

### W4.3 Ledger & reversibility
- Every op above is a ledger event (same hash chain). Event payloads reference memory ids + op metadata, never full content (content lives in the DB; ledger stays compact).
- Every op is reversible for 90 days: MERGE→unmerge (restore sources, tombstone merged), FORGET_COMMIT→restore from coldstore, CONTRADICT→swap validity. Reversal is itself a ledger event `REVERT{original_event_id}`.

### W4.4 Morning report (UI)
- On first interaction after a completed run, Silence-Model-gated (i.e., it must EARN the interruption: suppress if run produced <3 material ops): a single card — "While you were away: merged 14, resolved 2 contradictions, proposed forgetting 6, noticed 1 theme. Review."
- Review screen: grouped ops, each expandable to before/after, each with Revert. CONTRADICT_ASK items rendered as questions with two-tap resolution.
- Settings: aggressiveness dial reusing the house vocabulary — **Conservative / Hybrid / Full**: Conservative = MERGE+ABSOLUTIZE only; Hybrid (default) = + contradictions + forget-propose; Full = + auto forget-commit at K+1 and theme promotion without report gating.

### Acceptance criteria
- Simulated 30-day corpus test: seeded DB with planted duplicates, contradictions, relative dates, stale trivia → after 3 runs, planted issues resolved with zero false-positive forgets of high-trust memories (golden assertions).
- Ledger chain verifies end-to-end across a run + a revert.
- Kill switch, budgets, cloud-tier prohibition, and quarantine exclusion all covered by tests.
- Interrupted-run resume test (kill sidecar mid-stage-3, restart, run completes).

---

## W5 — Apple Foundation Models instant tier (spike → flag)

**Goal:** zero-download first-run on Apple silicon. New users get a working June conversation in <60s while Ollama models pull in the background.

### W5.1 Spike (timeboxed: 2 sessions) — answer these, write `docs/spikes/apple-fm.md`
1. Can the Python sidecar call Apple's on-device Foundation Model via the new Python SDK on macOS 26, on an EU-region Apple ID / EU-located machine? (Known risk: EU availability restrictions on parts of Apple Intelligence; verify the framework itself.)
2. Latency/quality of the on-device model for June's SHALLOW tier tasks (chat small-talk, classification, retrieval-answer synthesis on short contexts). It does NOT need to handle the deep tier.
3. Does calling it require app entitlements/provisioning incompatible with our current signing setup?
4. Fallback behavior when Apple Intelligence is disabled on the machine.

### W5.2 Implementation (only if spike passes; feature flag `models.apple_fm_tier`)
- Add as a roster tier BELOW the local Ollama tiers: used only when (a) flag on, (b) requested tier's Ollama model not yet available, (c) task classified shallow. This does not violate the "no third provider" non-goal — `[FOUNDER]` confirm this interpretation in the ADR; the model runs on-device, no network, no account, which honors the non-goal's intent (no new data-leaving-device path, no new vendor account).
- Onboarding copy change: "You can start chatting now; June's full brain is still downloading (2.1 GB, ~4 min)." Progress affordance.
- Trust screen: model-used is already per-message-attributable (verify; if not, add) — Apple FM responses must be labeled.

### Acceptance criteria
- Fresh install on Apple-Intelligence-enabled Mac: first useful response <60s from first launch, zero model download completed.
- Flag off ⇒ behavior identical to v0.1 path. Roster selection covered by tests.

---

## W6 — Opt-in telemetry + feedback channel

**Principle:** June's default is zero phone-home. Telemetry must be opt-in, boring, inspectable, and honest.

- First-run onboarding screen (after privacy explanation): "Share anonymous health pings?" default OFF. Copy lists exactly what is sent.
- Payload (when ON): app version, macOS major version, arch, coarse RAM bucket, crash count since last ping, night_shift runs completed, DB size bucket. NO content, NO identifiers beyond a random install UUID (regenerable via settings button). One ping per 7 days max.
- Endpoint: static ingestion (simplest possible; `[FOUNDER]` chooses: Cloudflare Worker vs GitHub-issues-based crash reporter vs plausible-style). Whatever is chosen, the full payload is viewable in-app before/after send, and the sender is listed in the Trust screen network disclosure.
- Crash reporting: on sidecar crash, offer (per-incident, never automatic) to open a prefilled GitHub issue with redacted traceback — user sees exact text before submission.
- Feedback: in-app "Send feedback" → prefilled GitHub Discussion/issue link. No custom backend.

### Acceptance criteria
- With telemetry OFF (default), a network-traffic test proves zero calls except: model providers user configured + the W1.4 update check (if enabled).
- Payload snapshot test; opt-out deletes install UUID.

---

## W7 — Playwright UI regression slices

Standing debt; now release-gating. Test app runs against a seeded fixture DB + a stub model server (deterministic responses) — no live model in CI.

- **Slice 1 (smoke):** launch → onboarding happy path → first chat send/receive → app relaunch state persistence.
- **Slice 2 (core screens):** memory list renders seeded items; Trust screen renders ledger events incl. chain-verify badge; settings toggles persist.
- **Slice 3 (v0.2 features):** provenance review confirm/discard flow; Night Shift morning report render + one revert; update-available affordance.
- CI: run headless on macOS runner; artifacts (screenshots/videos) on failure. Flake policy: a test that flakes twice gets fixed or quarantined-with-issue same week, never silently retried forever.

### Acceptance criteria
- All three slices green in CI on two consecutive main-branch runs; wired into the single gate for release tags.

---

## W8 — Local voice capture (stretch, flag `voice.enabled`)

- Path A (preferred if W5 hardware survey supports it): Gemma 4 12B native audio via Ollama for machines ≥16GB — direct audio-token ingestion, no transcript intermediary.
- Path B (fallback): whisper.cpp small model for transcription → normal text path. Ship whichever is robust; do not ship both this phase.
- UX: push-to-talk in chat input; audio buffer NEVER written to disk unencrypted; discarded after processing; ledger event `VOICE_CAPTURE{duration_s}` (no content).
- Out of scope: voice output/TTS, wake word, always-listening. Explicitly banned this phase.

---

## 9. New ADRs to author (sequence after current highest)

1. Retrieval v2: multi-signal fusion & bi-temporal facts (W2)
2. Provenance-gated memory writes (W3)
3. Night Shift: ledgered offline consolidation (W4)
4. Apple FM instant tier & the "no third provider" interpretation (W5) `[FOUNDER sign-off]`
5. Opt-in telemetry design (W6)
6. Update-check network call (W1.4)

## 10. Global engineering constraints (restated, binding)

- Single SQLite file remains the only store. No new databases, services, or daemons.
- All memory access through the facade; add an import-linter (or equivalent) rule enforcing it.
- Cloud tier (Gemini) NEVER receives: quarantined memories, Night Shift work, voice audio, ledger contents. Add a centralized "cloud egress allowlist" module through which all cloud-bound payloads pass, with tests.
- Performance budgets are tests, not aspirations: retrieval p95 <120ms @50k memories; app cold start <3s shell-interactive; Night Shift ≤20min.
- Every user-facing string for new features goes through the existing i18n/string table if one exists (W0.1 verify); if none exists, create a minimal one — EN only this phase, but structured for NL/GR later.
- Migrations: forward-only, idempotent, tested against a seeded v0.1-shaped DB AND an empty DB.

## 11. Milestones & release gates

- **M1 (end W0+W1):** `v0.2.0-alpha.1` — notarized DMG, honest README, clean-machine funnel documented. Distribute hand-to-hand to 10–15 known testers `[FOUNDER]`.
- **M2 (end W2+W3):** `v0.2.0-alpha.2` — retrieval v2 live, provenance live, internal benchmark numbers reviewed.
- **M3 (end W4+W6+W7):** `v0.2.0-beta.1` — Night Shift on for testers, telemetry opt-in live, Playwright gating.
- **M4:** `v0.2.0` — public release. Show HN draft `[FOUNDER]`: lead with auditable memory / Night Shift demo GIF; benchmarks linked; post-OpenClaw framing implicit, not name-calling.
- W5/W8 merge behind flags at any milestone without gating.

## 12. Explicitly deferred (documented so nobody relitigates casually)

- Messenger bridge (Signal-first, capability-tokened, guard-fronted) — v0.3 candidate; requires spotless v0.2 security record first.
- Privacy Mode 2 (encrypted backup) / Mode 3 (Google skills) — founder ADRs pending.
- Windows/Linux, payments/licensing activation, cloud sync/relay, third LLM provider — unchanged non-goals.
- Memory export/import "passport" beyond existing bootstrap importers — v0.3 candidate (pairs well with the data-sovereignty story).

## 13. Open questions for the founder (answer before the relevant WS starts)

1. Apple Developer enrollment date? (blocks W1.1)
2. W5 "no third provider" interpretation — confirm Apple on-device FM is roster-eligible.
3. Telemetry ingestion choice (W6).
4. Night Shift default state after onboarding: opt-in screen default ON or OFF?
5. Benchmark publication threshold — publish regardless of results, or only if ≥ parity with vec-only published baselines?

— END OF BRIEF —
