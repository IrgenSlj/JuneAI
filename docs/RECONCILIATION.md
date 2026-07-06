# RECONCILIATION.md — JUNE_V02_BRIEF vs. actual repo state

**Produced by:** W0.1 (the brief's mandated first task).
**Date:** 2026-07-06.
**Repo state audited:** `main` @ commit `54d1a5ba` (temporal context layer).
**Method:** direct read of the memory schema/migration/retrieval path + four
parallel subsystem audits (trust/guard/silence; providers/router/scheduler;
tasks/packaging/licensing/telemetry; memory/docs). File paths cited so every
claim is checkable.

This document is the source of truth for **where the brief's assumptions
(`JUNE_V02_BRIEF.md` §2) diverge from reality**, and — more importantly — **how
each divergence changes the downstream workstream specs**. Where the brief's SQL
or design fights the actual code or an accepted ADR, the divergence is recorded
here and the workstream adapts. Nothing downstream should be built against a
brief assumption this document contradicts.

---

## 1. Brief §2 assumptions — audited

Legend: **OK** = assumption holds. **DELTA** = materially different; adapt the
spec. **BETTER** = reality is ahead of the brief (less work than assumed).

| # | Brief §2 assumption | Actual state | Verdict | Impact |
|---|---|---|---|---|
| 1 | Tauri shell + supervised Python sidecar; watchdog + corrupt-DB recovery | Confirmed. Tauri (`apps/desktop/src-tauri`) bundles a PyInstaller-frozen sidecar staged at `Contents/Resources/june-api/`; corrupt-DB recovery + watchdog present (recent commits `82ab5c65`, `1f98c441`). | **OK** | — |
| 2 | Single SQLite; sqlite-vec; graph tables; one memory facade | Confirmed. One `june.db`; sqlite-vec `vec0` index (`memory/vec_index.py`); `graph_nodes`/`graph_edges` tables; `MemoryManager` (`memory/manager.py`) is the facade over `recall.py`/`writers.py`/`vector.py`/`graph.py`/`sqlite.py`. | **OK** | Facade-only rule (W2.3, §10) is enforceable — one seam already. |
| 3 | Memory fact store table = `memories`, integer `id` | **FALSE.** Table is **`semantic_facts`**, composite PK `(user_id, fact_id TEXT)`, content column is **`text`** (not `content`), learned-at column is **`created_at`** (not `observed_at`). Tombstone table `forgotten_facts` already exists (content preserved on forget). | **DELTA** | **All W2/W3/W4 DDL must be rewritten** against `semantic_facts` with a `TEXT fact_id` + `user_id` scoping. `superseded_by INTEGER REFERENCES memories(id)` becomes `superseded_by TEXT` (a `fact_id`). `content` → `text`. `observed_at` ≈ existing `created_at`. See §3. |
| 4 | Trust Ledger: append-only, hash-chained, "Trust" screen | Confirmed and strong. `trust/ledger.py`: blake2b hash chain, `GENESIS_PREV`, process-wide append lock, `LedgerReader.verify_chain()` with tail-truncation check, table `trust_ledger`. UI: `/system` page titled **"Trust"**; ledger renders as **Receipts** (`/system/receipts`); optional Ed25519 signing (`trust/signing.py`). | **OK** (better) | W3.3 events are cheap to add — **but see §3, the `kind` field is a 4-value frozenset, not an extensible enum.** |
| 5 | Guard layer: policy checks on tool/skill paths | Confirmed structurally. `guard/actions.py::evaluate_call()` is the single entry (called in `loop/wiring.py`); action classification, taint tracking (`is_tainted`), approval gates, untrusted-content framing (`guard/framing.py`), secret redaction (`guard/redaction.py`). | **OK** but see #6 | — |
| 6 | (implied by W3.2.4) Guard has injection / instruction-shaped-content heuristics | **FALSE.** There is **no** content-based injection detector — no "ignore previous" scan, no imperative/tool-invocation-phrasing classifier. Defense is purely *structural* (framing + action gates + taint). The red-team strings exist only in tests, asserting they're delivered *as framed data*. | **DELTA** | **W3.2.4 must BUILD the injection heuristic from scratch** (new `guard/injection.py` or similar). It is net-new work, not a reuse. Draft it in the W3 ADR. |
| 7 | Silence Model: notification/proactivity governor | Confirmed. `silence/policy.py::decide()` → `now`/`batch`/`suppress`; thresholds `HIGH_SALIENCE_THRESHOLD=0.7`, `URGENT_WINDOW_HOURS=3.0`, `DISMISSAL_SUPPRESS_THRESHOLD=2`; `surfacing_decisions` table; every decision mirrored to the ledger. "Earned interruption" is real (UI copy + salience gate). Presence = `silence/presence.py` (recency-of-activity). | **OK** | W4.4 morning-report gating reuses `decide()` + a new candidate builder. The "<3 material ops ⇒ suppress" rule maps to a salience computed from op count. |
| 8 | Durable promises: resumable long-running tasks | **PARTIAL.** `tasks` table + per-step trace persisted incrementally (`append_step`/`update_step`); restart *reconciliation* (`reconcile_running_after_restart` → `awaiting_user`, "Interrupted by an app restart"). **But there is no resume-from-checkpoint** — resuming re-runs `task.goal` from scratch with a fresh `SessionState`. Docstring calls this the "vessel-filling" version. | **DELTA** | **W4.2's "each stage resumable via existing durable-promise machinery" is not free.** Real per-stage checkpoint/continue logic must be added (checkpoint table or `plan`-JSON stage cursor). Statuses available: `planning/running/paused/awaiting_user/completed/failed/cancelled`. |
| 9 | Roster: local Ollama Gemma tiers + Gemini cloud; selection logic exists | Confirmed. `providers/providers.toml`: `local-fast=gemma4:e2b`, `local-deep=gemma4:e4b`, `cloud-capable=gemini-2.0-flash`; embeddings `nomic-embed-text`. Live selection = difficulty-based (`router/difficulty.py::tier_for_difficulty`) which **never auto-routes to cloud**; a second policy router (`routing.py::ModelRouter.resolve`, ADR 0009) is wired only into self-tests. | **OK** but see #16 | W5's Apple-FM tier slots *below* `local-fast`. Two parallel roster naming schemes exist (see §3). |
| 10 | 763+ backend tests; Playwright largely absent | **BOTH STALE.** Backend now **908 test functions / 94 files**. Playwright is **present**: `apps/web/playwright.config.ts` + 7 specs (`trust`, `memory`, `silence`, `promises`, `home`, `glass`, `receipts`) with a mocks harness (`e2e/_mocks.ts`). Gated OFF by default in `check.sh` (opt-in `JUNE_E2E=1`). | **BETTER** | **W7 is partly done.** It becomes: extend coverage to the *new* v0.2 flows (provenance review, Night Shift report, update affordance) + wire `JUNE_E2E=1` into the release-tag gate + a flake policy. Not a from-scratch build. |
| 11 | June.app + DMG proven; CI signing degrades gracefully; enrollment pending | Confirmed. `.github/workflows/release.yml` (on `v*` tags) imports Developer ID cert into a temp keychain, injects hardened runtime + entitlements via `--config` override, delegates notarization/stapling to the Tauri bundler; a "check for signing secrets" step selects a signed vs. unsigned build path; DMG uploads regardless. `apps/desktop/RELEASING.md` already exists. | **OK** (better) | W1.1 = wire real secrets once enrolled `[FOUNDER]`. **W1.2's `RELEASING.md` already exists** (refresh, don't create). |
| 12 | Licensing: Ed25519 core present, dormant, empty `PUBLIC_KEYS` | Confirmed exactly. `licensing/verify.py` = 9-step offline verify, never raises, no network; `licensing/keys.py::PUBLIC_KEYS = {}`. Tiers `free/pro/founder`; features `backup_sync/cloud_relay/google_skills/supporter/commercial_use`. | **OK** | Stays dormant (non-goal §3). No action. |
| 13 | README known-stale (ChromaDB/LangGraph refs) | **FALSE.** README (updated 2026-06-28) is accurate: sqlite-vec, ADR 0018 (LangGraph removed), Gemma 4 + Gemini, correct routes. **Zero** stale tokens in README or (future) CURRENT.md. ChromaDB/LangGraph appear only in `docs/experiments/*` (legit historical baselines) and one `docs/product/*` "removed" note. | **BETTER** | **W0.3 is an *enhancement* to the product-surface spec, not a de-stale rewrite.** Add: positioning sentence, hero media, mermaid, hardware tiers, security posture. |
| 14 | `docs/product/` extensive tree | Confirmed (12 files). See §4 inventory. | **OK** | — |

---

## 2. Assumptions the brief *did not state* but that matter

| Discovery | Impact |
|---|---|
| **No FTS5 anywhere.** The current "keyword" recall channel (`recall.py::sqlite_keyword_hits`) is a naive Python `substring-in-lowercase` scan over *structured* tables (goals/loops/prefs/relationships/journal) — **not** BM25, and **not** over `semantic_facts.text`. | W2.2 (FTS5 over `semantic_facts`) is genuinely net-new and additive. The existing keyword scan is not a substitute and should be kept or folded in deliberately. |
| **Retrieval is already multi-store fusion.** `recall.py::gather_hits` fuses vector + graph + structured, dedupes by text, applies feedback multipliers, and salience-reranks (recency × frequency × relevance). | W2.3's "four-signal fusion" is really "add a real BM25 channel + RRF + a temporal signal + a config dataclass on top of the *existing* fusion," not a greenfield build. `s_time` overlaps the existing salience recency term — reconcile, don't double-count. |
| **The retrieval counter W4.5 says to "add if absent" already exists**: `semantic_facts.access_count` + `last_accessed` (migration 5), incremented on the read path (`recall.py::_salience_rerank`). | W4.5 DECAY scoring can read `access_count` directly. No new counter. |
| **Migration mechanism**: versioned registry in `memory/migration.py` (`@MIGRATIONS.register(n, ...)`), tracked in `_schema_migrations`. **Latest applied version = 6; next = 7.** A parallel additive-column migration exists for tasks (`tasks/migration.py`). | All W2/W3 migrations register as **v7, v8, …** in `memory/migration.py`, forward-only + idempotent, per the established pattern (see migration 5's duplicate-column swallow). |
| **Cloud egress is *recorded*, not *gated*.** `providers/provenance.py::record_cloud_call` is the single chokepoint but it's a *passive ledger recorder* (writes `kind="egress"`), invoked only inside `GeminiProvider`. There is no allowlist/enforcement module that inspects/blocks a payload before it leaves. | §10's "centralized cloud egress allowlist through which all cloud-bound payloads pass, with tests" must be **built as an enforcement gate** — but the single chokepoint to build it into already exists. W3/W4 cloud-prohibition tests depend on this. |
| **No OS idle / AC-power / battery / screen-lock signal exists** in the brain. Full-tree grep: no `caffeinate`, no power/idle/lock access. The only "idle" is `silence/presence.py` recency-of-HTTP-activity (`present-active`/`present-idle`/`absent`). No Tauri→brain power/idle bridge. | **W4.1 trigger conditions are not evaluable today.** Either build a Tauri-shell power/idle bridge (new IPC surface) or scope W4's trigger to *June-app idle* (the brief permits this fallback: "June-app-idle is sufficient this phase"). Decide in the W4 ADR. |
| **`proactive/` is dead code** — untracked stale `.pyc` only (the retired 30-min tick engine). Superseded by `silence/`. | Ignore it; do not build Night Shift there. Delete the stale pyc opportunistically. |
| **Version truth is fragmented.** `0.1.0` in tauri.conf/Cargo/all package.json/brain+api pyproject; **`0.0.0`** in workspace `pyproject.toml`; **`0.2.0`** hard-coded in `packages/api/.../app.py` (FastAPI/OpenAPI `version=`) — a drift bug; sidecar runtime reports a **git short SHA**, not semver. No sidecar `--version` flag. | W1.2 (single source of version truth) is genuinely needed and should also **fix the `app.py` `0.2.0` drift** and add a sidecar `--version`. |
| **No in-app update check** of any kind (no `tauri-plugin-updater`, no GitHub Releases API call). | W1.4 is fully net-new. It will be the *first* permitted non-model network call — must land in the Trust screen network disclosure + settings off-switch from day one. |
| **Telemetry is 100% local.** `memory/daos/telemetry.py` + `telemetry.py` façade write only to local `telemetry`/`app_state` tables; no `requests`/`httpx`/`socket`. Nothing phones home. | W6's opt-in health ping is fully net-new and does not collide with the local analytics of the same name. Name the new module distinctly (e.g. `health_ping`) to avoid confusion. |

---

## 3. Per-workstream spec adaptations (binding)

These override the brief where they conflict. Each is expanded in the relevant
workstream's ADR.

### W2 — Retrieval v2
- **Table/columns:** operate on `semantic_facts`. Add columns `valid_from TEXT`,
  `valid_to TEXT`, `observed_at TEXT` (backfill from `created_at`),
  `superseded_by TEXT` (a `fact_id`, **not** an integer FK). Index
  `idx_semantic_facts_validity (valid_to, valid_from)`. Register as migration **v7**.
- **FTS5:** `memories_fts` → name it `semantic_facts_fts`, `content='semantic_facts'`,
  `content_rowid` — but `semantic_facts` has a **composite text PK, no integer rowid
  surrogate exposed**. Either add an `INTEGER PRIMARY KEY`/rowid alias for FTS
  external-content linkage, or use a contentless/standalone FTS table synced by
  triggers keyed on `fact_id`. Decide in the ADR (leaning: standalone FTS keyed by
  `fact_id`, triggers on insert/update/delete).
- **Fusion:** keep the existing vector/graph/structured fusion; add the BM25 channel
  + RRF over the vec & bm25 lists; fold `s_time` into (not on top of) the existing
  salience recency so recency isn't counted twice. One `RetrievalConfig` dataclass.
- **Facade guard:** add the import-linter/test forbidding non-facade FTS/vec access.

### W3 — Provenance & quarantine
- Columns land on `semantic_facts` (v8): `provenance TEXT DEFAULT 'unknown'`,
  `trust_level INTEGER DEFAULT 0`, `quarantined INTEGER DEFAULT 0`, `source_ref TEXT`.
- **Ledger events:** `trust_ledger.kind` is a **frozenset `{egress, action, approval, system}`**,
  not an extensible enum. Do **not** add `MEMORY_WRITE`… as new `kind`s casually.
  ADR decision required: either (a) add one new kind `"memory"` and put the op in
  `payload.op`, or (b) reuse `kind="action"` with `payload.op="MEMORY_WRITE"`.
  Recommendation: (a) — a first-class `"memory"` kind reads best in Receipts and
  keeps the existing filters clean. This changes `VALID_KINDS` + the Receipts filter
  UI (which currently omits even `system`).
- **Injection filter (W3.2.4):** **build it** — no guard heuristic exists. New
  detector invoked before any `web_derived` write; positive → block + ledger event.
- Trust-weighted retrieval multiplier composes with W2's fusion score.

### W4 — Night Shift
- **ADR-0016 tension is real and founder-relevant.** A timer/idle-triggered
  background inference job is the pattern ADR 0016 + CLAUDE.md forbid
  ("do NOT reintroduce heartbeat-as-cron"). The sanctioned pattern in-tree is
  *presence-triggered* consolidation ("reflection on arrival"). **The W4 ADR must
  explicitly reconcile Night Shift with ADR 0016** (the honest framing: Night Shift
  is *offline maintenance*, not proactive *engagement*, and its only user-facing
  output is Silence-Model-gated at the next natural boundary). `[FOUNDER]` sign-off. See §5 Q6.
- **Triggers:** no OS idle/power inputs exist. Either build a Tauri→sidecar
  power/idle IPC bridge, or use the brief-permitted fallback (June-app idle via
  `silence/presence.py` + a `min_interval_hours` cron-style guard). Decide in the ADR.
- **Resumability:** the durable-promise machinery persists state/trace and reconciles
  on restart but does **not** resume mid-plan. Add explicit per-stage checkpointing
  (stage cursor in the run row) — do not assume `execute()` continues where it stopped.
- **Coldstore:** `forgotten_facts` already tombstones with content preserved but is
  **not encrypted**. W4.5's "encrypted-at-rest `memories_cold`" is a new requirement;
  reconcile with the existing tombstone tables (extend vs. add).
- **Cloud prohibition** relies on the egress *gate* that W3/§10 must build (currently
  only recorded).

### W1 — Release engineering
- `RELEASING.md` exists (`apps/desktop/RELEASING.md`) — refresh, don't create.
- Fix version fragmentation incl. the `app.py` `0.2.0` drift; add sidecar `--version`.

### W0.2 — Doc consolidation
- ADR index **already exists** at `docs/decisions/README.md` (a status table through
  0023). **Refresh it in place**; do not create a second `docs/adr/INDEX.md`.
- **CLAUDE.md points to `docs/product/development-plan.md` as the active plan, but the
  project memory + README point to `docs/product/rebuild-plan.md` as canonical.** This
  pointer is inconsistent. CURRENT.md must become the single pointer and CLAUDE.md
  updated to match. (Founder note: confirm rebuild-plan vs development-plan primacy.)

---

## 4. Doc inventory & classification (W0.1)

Classification: **CURRENT** (authoritative, keep at path) · **SUPERSEDED** (was a
plan, now behind this brief + CURRENT.md; archive in W0.2) · **HISTORICAL** (a
record of a moment; keep for provenance, never a live spec).

| Path | Class | Note |
|---|---|---|
| `docs/vision.md` | CURRENT | Durable worldview (four inversions). Untouchable. |
| `docs/product/overview.md` | CURRENT | Product truth. |
| `docs/product/roadmap.md` + root `ROADMAP.md` | CURRENT | Public track sequencing. |
| `docs/architecture/overview.md` | CURRENT | Layered model; accurate. |
| `docs/decisions/0001–0023` + `README.md` | CURRENT | ADRs are append-only; index refreshed in W0.2. |
| `docs/setup/desktop.md`, `docs/setup/environment.md` | CURRENT | Config/setup reference. |
| `JUNE_V02_BRIEF.md` (Downloads) | CURRENT | The active v0.2 plan → copy into repo as the linked plan from CURRENT.md. |
| `docs/product/development-plan.md` | SUPERSEDED | Pre-v0.2 working plan/progress log. Behind this brief. |
| `docs/product/rebuild-plan.md` | SUPERSEDED | The rebuild plan (S0–S13); open items now carried by this brief per its header. |
| `docs/product/ship-to-revenue.md` | SUPERSEDED | Superseded by this brief's milestones §11. |
| `docs/design/master-brief.md` | SUPERSEDED | Superseded by shipped UI + this brief. |
| `docs/product/CLAUDE_HANDOFF_silence_and_trust.md` | HISTORICAL | Handoff note for shipped Silence/Trust work. |
| `docs/product/strategic-review-2026-07-01.md` | HISTORICAL | Point-in-time review. |
| `docs/product/sidecar-spike-findings.md` | HISTORICAL | Spike record (shipped). |
| `docs/product/tauri-build-report.md` | HISTORICAL | Build report (shipped). |
| `docs/product/cold-start-notes.md` | HISTORICAL | Notes; feed W1.3 funnel test, then archive. |
| `docs/product/license-design.md` | HISTORICAL | Licensing core shipped + dormant. |
| `docs/experiments/baseline-2026-06.md`, `docs/experiments/loop-clear.md` | HISTORICAL | LangGraph/Chroma baselines — legit historical; **exempt from the W0.4 stale-token check.** |
| `docs/design/artifact/*` | HISTORICAL | Design prototype (realized in the shipped UI). |
| `docs/LOGO_PROMPT.md` | HISTORICAL | Asset-gen prompt. |

---

## 5. Banned stale-token list (feeds W0.4 CI check)

The doc-hygiene check fails if **README.md** or **docs/CURRENT.md** contain any of:

- `chromadb`, `chroma` (vector backend, superseded by ADR 0019 / sqlite-vec)
- `langgraph`, `langchain` (loop engine, removed by ADR 0018)
- `sentence-transformers`, `sentence_transformers` (dropped with Chroma)
- `Quick Capture`, `capture_items`, `action_intents`, `operating layer` (abandoned, ADR 0015)
- `heartbeat`, `proactive_tick`, `daily orchestration` (abandoned, ADR 0016)
- Old route names to confirm during W0.3 grep (candidates: `/receipts` as a *top-level* route — it's nested under `/system`).

**Scope:** README + CURRENT.md only. `docs/experiments/*` and archived plans are
**exempt** (they legitimately describe removed tech as history).

---

## 6. Founder questions surfaced

From the brief §13 (unchanged, still open):
1. Apple Developer enrollment date? (blocks W1.1)
2. W5 "no third provider" interpretation — is on-device Apple FM roster-eligible?
3. Telemetry ingestion choice (W6).
4. Night Shift onboarding default: ON or OFF?
5. Benchmark publication threshold — publish regardless, or only at ≥ vec-only parity?

New, raised by this reconciliation:
6. **Night Shift vs. ADR 0016.** Confirm the framing that idle-triggered *offline
   maintenance* (surfaced only via the Silence Model) does not violate the
   no-heartbeat inversion — or require presence-triggered-only. This gates W4's ADR.
7. **Ledger event modelling.** Approve adding a first-class `kind="memory"` to the
   Trust Ledger (vs. overloading `kind="action"`), which touches `VALID_KINDS` +
   Receipts filters.
8. **Canonical plan pointer.** `rebuild-plan.md` (memory/README) vs.
   `development-plan.md` (CLAUDE.md) — which is primary before both are archived
   behind CURRENT.md? (Proposed: archive both, CURRENT.md + this brief become the pointer.)

---

*End of reconciliation. Downstream workstreams must adapt to §1–§3 before implementation.*
