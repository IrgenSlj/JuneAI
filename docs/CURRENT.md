# CURRENT.md — state of June

**The single authoritative "what is true right now" page.** When any other
planning doc disagrees with this one, this one wins (and that doc should be
archived). Updated as workstreams land.

- **Last updated:** 2026-08-20 (Stream D complete; B.3/B.5 landed; D.5d measured).
- **Release status:** `v0.1.0` re-cut on 2026-07-25 and **verified working** — a
  45MB Apple Silicon DMG built by CI from the tag, with the frozen sidecar inside
  it, ad-hoc signed. The tag points at the code the artifact was built from. The
  previously published asset (2.7MB, no sidecar, cut before the packaging pipeline
  existed) has been deleted. Active target: **`v0.3.0`**.
- **Active plan:** [`v0.4-development-plan.md`](product/v0.4-development-plan.md)
  — the single plan of record. **Stream D** (correctness and coherence) is the
  current work and displaces the remaining pre-launch items. `v0.3-development-plan.md`,
  `JUNE_V02_BRIEF.md` and `v0.2-execution-plan.md` are superseded.
- **Resume here next session:** **Stream D**, in the
  [plan](product/v0.4-development-plan.md). Work the slices in order; each is
  independently landable (one slice -> `check.sh` green -> one commit -> push),
  and each slice's own **Status** line records where it stopped.
  **All of Stream D is done** — D.1 through D.9 — bar the two fat Svelte admin
  pages noted under D.8.

  **D.5d measured the new surface across four runs**
  ([results](experiments/tool-selection-2026-08.md)). D.5a's argument held both
  ways: near-synonym confusion is gone, and the opposite risk never appeared —
  zero spurious calls and 100% abstention in 288 turns. `tool_aliases.py` never
  fired once.

  It also found the live chat path was **never told when to call a tool**.
  `build_system_prompt` carries those rules and only the scheduler reads it;
  `ContextAssembler` builds its own prompt and said nothing about tools. The
  model chose correctly at the provider level and the loop returned no calls,
  while the reply read "I have remembered that you are vegetarian" — a false
  statement about the user's own data. Fixed; tool turns +7.8 points.

  **Still open, and the top item:** `remember` sits at 60% across all four runs
  and two prompt interventions, so two of every five "remember this" requests
  store nothing — and the reply still claims otherwise. **`gemma4:e4b` is not
  better** (50% on the same corpus), so escalating memory instructions to
  `local-deep` is not the fix; that was measured, not assumed. The remaining gap
  is neither model capacity at this scale nor wording.

  The honesty backstop is structural and has landed: the turn frame reports
  `memories_written`, so a turn that stored nothing looks different from one
  that did. Worth knowing when picking this up — `forget` and `update_promise`
  report their no-match cases truthfully, so **June tells the truth whenever a
  tool actually runs**; the false claim appears only when no tool ran. It is a
  call-rate problem, not a lying-model problem. The unmeasured lever is the
  cloud tier (`--role cloud-capable`), which needs a key and egress and so is
  the founder's call.

  **The D.5b export decision, taken 2026-08-19: leave the tables, remove only
  the code.** `export_memory` enumerates `sqlite_master` rather than a fixed
  list, so every surviving table is already exported; dropping them is
  irreversible with no migration-down; and recall's keyword channel still reads
  the structured rows, so a pre-cleanup user's data keeps reaching them.

  **The tranche 2 decision was taken on 2026-08-19** ([ADR 0032](decisions/0032-model-callable-memory-surface.md)):
  option (b). The seven v1 domain writers are gone and June's model-callable
  memory surface is four deliberate tools — `remember`, `forget`,
  `list_promises`, `update_promise` — on the existing `MemoryManager` and
  `TasksStore` seams. Before this June had *no* model-callable memory tools at
  all, so the product's two headline capabilities were things it could not do on
  purpose when asked. `JUNE_TOOLS` is 54 -> 12 and `JUNE_TOOLS_GEMMA` 24 -> 5,
  with every remaining tool one the product describes. Calendar was **handed off**
  to its MCP skill rather than retired, and `SKILL_OWNED_TOOL_NAMES` plus
  `test_a_handoff_actually_hands_off` keep that distinct from retirement.

  One item is parked, and one closed. Re-running the reliability baseline still
  needs attention (the numbers on file describe the pre-D.4c path no user
  reached). **The tool surface with skills running is no longer untested**: the
  D.5d harness exercises it, which `check.sh` cannot (`JUNE_SKILLS_DISABLED=1`),
  and it is 15 tools on a default install rather than 5. Reading its log is what
  found the skill-contract defect. Still open: the two fat Svelte admin pages
  (`system/+page.svelte` 1515 lines, `skills/+page.svelte` 1335) were never
  decomposed into `packages/ui` the way chat was.
  Launch (Phase 7) remains gated behind D: 7.1, 7.2 and 7.4 are done, and the
  remaining blockers are the 240MB `.git` rewrite (7.0/B.1 — needs a quiet tree,
  it force-pushes `main`) and cutting the release (7.3, blocked on repo workflow
  permissions). The reasoning that gated Phase 7 on SSRF and the packaged binary
  applies to Stream D unchanged: announcing a security-positioned product whose
  live chat path can drop a tool call, and whose Local-only mode does not stop an
  outbound write, would invert the pitch at the moment of maximum scrutiny.
  Phases 0-6 are done.
- **Four decisions taken 2026-07-27** (plan §9): OS geolocation asked once at
  point of use and coarsened to city level; all four launch blockers before
  announcing; rewrite git history before launch; monetization parked for v0.3.
  Five questions stay open — launch channel and order, second platform,
  notarization, the Brave key, and repo workflow permissions.
- **The plan was re-ordered on 2026-07-26.** June has 0 stars and 0 downloads
  after 835 commits, and the local-first-open-source category now has a
  3.2M-user incumbent. The differentiator moves from *local-first* to **the agent
  that can prove what it did** — which is what the guard layer and Trust Ledger
  already are. Reach comes before polish: MCP server, then a checkable security
  claim, then launch. Rationale in
  [`v0.3-development-plan.md`](product/v0.3-development-plan.md) §2.
- **Repo audit:** [`repo-audit-2026-08-18.md`](product/repo-audit-2026-08-18.md)
  — the source of Stream D. Three defects on the live chat path (two proven by
  failing tests), and one structural problem larger than all of them: the v1
  life-coach product was never deleted and still owns 30 of the 54 tools in
  `JUNE_TOOLS` and all 24 in `JUNE_TOOLS_GEMMA`. The
  [2026-07-26 audit](product/repo-audit-2026-07-26.md) still holds on its own
  terms — the 240MB git history and untested first-run paths remain open.
- **Reconciliation (brief vs. reality):** [`RECONCILIATION.md`](RECONCILIATION.md) — historical reference for v0.2.
- **Durable worldview:** [`vision.md`](vision.md) (the four inversions; non-negotiable).
- **Decision log:** [`decisions/`](decisions/) — ADRs 0001–0024 and 0030–0032 accepted; index in [`decisions/README.md`](decisions/README.md).

---

## What June is (one line)

*A personal AI you can audit. June remembers you, forgets gracefully, explains
every action, and never phones home.*

June's center of gravity is the user, not the task. It inverts a coding agent's
four operations: **defers** instead of verifying, **continues** standing
intentions instead of completing and exiting, **forgets** gracefully instead of
accumulating, and **stays quiet** instead of acting fast (the four inversions,
ADR 0015).

---

## Architecture — one paragraph per subsystem

Layering: **Shell → API → Brain → Providers**, with the Brain usable standalone.
Everything lives under one versioned data directory (ADR 0019).

- **Shell (`apps/desktop`).** A Tauri (macOS Apple Silicon) shell supervises a
  PyInstaller-frozen Python sidecar staged at `Contents/Resources/june-api/`.
  Watchdog + corrupt-DB recovery are in place. The web PWA (`apps/web`) is the
  primary shipped surface and the same build every shell wraps.

- **API (`packages/api`).** A thin FastAPI REST + SSE boundary. Pydantic schemas
  are the single source of truth; the TypeScript client is generated from the
  OpenAPI spec (drift is gated in `check.sh`). On startup it reconciles any
  `running` promises orphaned by a restart.

- **Brain (`packages/brain`).** One hand-written harness loop (`loop/handwritten.py`,
  ADR 0018 — no agent framework), model-specific providers, layered context with
  anchored compaction, and June's self-authored character with a fixed honesty +
  safety floor. Loop shape is fixed and never self-modified.

- **Memory (`packages/brain/.../memory`).** One `june.db` over three stores,
  reached through `memory/sqlite.py`. (`MemoryManager` is the highest-level
  facade but not the only door: 18 modules open connections directly. See the
  2026-08-18 audit.) The stores are: structured SQLite rows, a sqlite-vec
  `vec0` semantic index, and an entity graph (`graph_nodes`/`graph_edges`). Facts
  live in **`semantic_facts`** (composite PK `user_id`+`fact_id`, with bi-temporal
  validity columns). Recall (`recall.py::gather_hits`) fuses **four signals** —
  vector similarity, BM25 over the FTS5 mirror `semantic_facts_fts` (kept current
  by insert/update/delete triggers), entity overlap, and temporal validity — via
  Reciprocal Rank Fusion (`rrf_k=60`, tunable through `RetrievalConfig`), then
  reranks by *salience* (recency × frequency × relevance), not similarity alone.
  Missing FTS5 degrades gracefully to the vector channel. Embeddings are served
  locally by Ollama (`nomic-embed-text`). Forgetting is first-class and tombstones
  content into `forgotten_*` tables. Schema is versioned (`memory/migration.py`,
  latest v7). **Measured:** fusion beats vector-only recall@8 by **+29%** on a
  100-case golden corpus at 50k facts (0.760 vs 0.588, MRR 0.702 vs 0.456). p95
  retrieval is 252ms at that scale, against a 120ms target that was not met —
  the vec0 scan is brute-force and grows linearly. Full results, method, and open
  work in [`product/retrieval-benchmark.md`](product/retrieval-benchmark.md).

- **Providers / routing (`packages/brain/.../providers`, `router`).** Three roles:
  `local-fast` (`gemma4:e2b`), `local-deep` (`gemma4:e4b`), `cloud-capable`
  (`gemini-2.0-flash`). The live loop routes by difficulty
  (`router/difficulty.py`) and **never auto-escalates to cloud**; cloud is reached
  only on explicit agentic/skill paths. Every cloud call is bracketed by a single
  chokepoint (`providers/provenance.py::record_cloud_call`) that writes an
  `egress` entry to the Trust Ledger — a skill cannot egress and skip the ledger.
  The chokepoint is no longer passive: in local-only mode the `start` phase raises
  `CloudEgressBlockedError` **before** the request leaves the machine.

- **Trust Ledger (`packages/brain/.../trust`, ADR 0022).** Append-only,
  blake2b-hash-chained local event log (`trust_ledger`), with tail-truncation-aware
  chain verification and optional Ed25519 signing. Kinds today:
  `egress`/`action`/`approval`/`system`/`mcp_access`. Renders in the UI as **Receipts** under
  the **Trust** screen (`/system`), with a verify affordance.

- **Guard layer (`packages/brain/.../guard`, ADR 0021).** A single seam
  (`guard/actions.py::evaluate_call`) classifies each tool call, tracks taint
  (content flowing from untrusted results back into new actions), gates
  `execute`/`write_network`/tainted-network behind approval, frames every tool
  result as untrusted content, and redacts secrets before they hit the ledger.
  The loop's `is_network_tool()` delegates to `classify_action()` (D.3), so the
  Local-only partition and `provenance.egress` cover outbound writes and not only
  the three named read-network tools.
  Defense is primarily **structural** — the gates hold regardless of what the
  content says. A content heuristic (`guard/injection.py`) sits under it as
  defence in depth: it does not block, it *revokes standing approvals*, so an
  "always allow" stops covering a tool once a poisoned result lands
  ([measured](product/injection-benchmark.md): 100% recall, 3% false positives
  on a 62-case corpus). Skill capability contracts declared in the manifest are
  enforced here too, not merely reported. Full picture, gaps first, in
  [`security/threat-model.md`](security/threat-model.md).

- **Silence Model (`packages/brain/.../silence`, ADR 0023).** Governs
  June-*initiated* surfacing only (never the reply path). A pure, clockless,
  model-free rules policy (`decide()` → `now`/`batch`/`suppress`) turns candidates
  into decisions, gated by salience and presence; interruptions must be *earned*.
  Every decision — including staying quiet — is mirrored to the ledger. Presence is
  derived from recency-of-activity (`silence/presence.py`); there is no OS
  idle/power signal.

- **Promises / tasks (`packages/brain/.../tasks`).** Standing intentions (not
  terminating TODOs) persisted in `tasks` with per-step trace, blocked-reason /
  next-action / final-deliverable, retries (cap 5), recurrence, and restart
  reconciliation. Resuming re-runs the goal; there is no mid-plan checkpoint resume
  yet. Exposed at `/tasks`, rendered as **Promises**.

- **Skills (`skills/`, ADR 0005).** Capabilities are standalone MCP servers over
  stdio, one supervised subprocess each, independently toggled, guard-fronted.

- **Scheduler (`packages/brain/.../scheduler`, ADR 0016).** Deterministic,
  *user-requested* jobs only (cron/interval/at). No heartbeat, no timer-driven
  proactivity. A separate event poller drains real-world skill events — the
  sanctioned "world changed" wake.

- **Licensing (`packages/brain/.../licensing`).** A complete offline Ed25519
  entitlement core that is **dormant** (empty `PUBLIC_KEYS`) — everything resolves
  to the `free` tier. No payments, no network.

- **Telemetry.** 100% local. The `telemetry` table records local analytics only;
  nothing phones home. Opt-in health pings are a v0.2 workstream (W6), not yet built.

---

## Privacy posture (enforced in code, not promised)

- No account needed, no signup, no cloud sync by default.
- One automatic network call: a release check, at most daily, carrying no user
  data, ledgered as egress, blocked by local-only, separately disableable
  (ADR 0031). Everything else leaves only when the user asks.
- No silent cloud calls — every cloud-routed model call is surfaced in the UI and
  written to the ledger. A privacy dial can lock June to local-only (provably
  blocks egress).
- No telemetry without explicit opt-in.
- Honesty is a fixed core; personalization shapes tone, never candor.

---

## v0.2 status board (historical — see v0.3 plan for current direction)

| WS | Name | State |
|----|------|-------|
| W0 | Reconciliation, doc consolidation, README | **Done** — pushed (`1acec73d`); doc-hygiene gate live |
| W1 | Release engineering: signing, versioning, update check | Not started (W1.1 blocked: Apple enrollment `[FOUNDER]`) |
| W2 | Retrieval v2: FTS5 + RRF fusion + temporal validity + benchmarks | **In progress** — ADR 0024 accepted (`3702ae33`); S2.1 migration v7 landed (`c0cb1cc9`); S2.3 RRF fusion is the next slice |
| W3 | Memory provenance & quarantine | Not started (needs ADR; injection filter is net-new) |
| W4 | Night Shift: auditable consolidation | Not started (needs ADR + ADR-0016 reconciliation `[FOUNDER]`) |
| W5 | Apple Foundation Models instant tier | Not started (spike-gated, `[FOUNDER]`) |
| W6 | Opt-in health telemetry | Not started (endpoint choice `[FOUNDER]`) |
| W7 | Playwright UI regression | Partly present (7 specs exist; extend to v0.2 flows) |
| W8 | Local voice capture (stretch) | Not started |

See [`RECONCILIATION.md`](RECONCILIATION.md) §3 for the binding spec adaptations
each workstream must follow.
