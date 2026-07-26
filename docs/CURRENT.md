# CURRENT.md — state of June

**The single authoritative "what is true right now" page.** When any other
planning doc disagrees with this one, this one wins (and that doc should be
archived). Updated as workstreams land.

- **Last updated:** 2026-07-26.
- **Release status:** `v0.1.0` re-cut on 2026-07-25 and **verified working** — a
  45MB Apple Silicon DMG built by CI from the tag, with the frozen sidecar inside
  it, ad-hoc signed. The tag points at the code the artifact was built from. The
  previously published asset (2.7MB, no sidecar, cut before the packaging pipeline
  existed) has been deleted. Active target: **`v0.3.0`**.
- **Active plan:** [`v0.3-development-plan.md`](product/v0.3-development-plan.md)
  — the single plan of record: state, competitive position, phases, slices, and
  acceptance criteria. The separate execution plan was merged into it on
  2026-07-26; `JUNE_V02_BRIEF.md` and `v0.2-execution-plan.md` are superseded.
- **Resume here next session:** plan **Phase 4.3 — surface MCP grants in the UI**.
  ADR 0030 and the read-only server (4.1, 4.2) are shipped.
  Phases 0-3 are done: docs reconciled, a working release shipped, retrieval
  measured ([results](product/retrieval-benchmark.md)), and a visual identity
  built (social card, hero, nine architecture diagrams).
- **The plan was re-ordered on 2026-07-26.** June has 0 stars and 0 downloads
  after 835 commits, and the local-first-open-source category now has a
  3.2M-user incumbent. The differentiator moves from *local-first* to **the agent
  that can prove what it did** — which is what the guard layer and Trust Ledger
  already are. Reach comes before polish: MCP server, then a checkable security
  claim, then launch. Rationale in
  [`v0.3-development-plan.md`](product/v0.3-development-plan.md) §2.
- **Repo audit:** [`repo-audit-2026-07-26.md`](product/repo-audit-2026-07-26.md)
  — the codebase is clean; the open items are a 240MB git history carrying v1
  artifacts, and untested first-run paths.
- **Reconciliation (brief vs. reality):** [`RECONCILIATION.md`](RECONCILIATION.md) — historical reference for v0.2.
- **Durable worldview:** [`vision.md`](vision.md) (the four inversions; non-negotiable).
- **Decision log:** [`decisions/`](decisions/) — ADRs 0001–0024 and 0030 accepted; index in [`decisions/README.md`](decisions/README.md).

---

## What June is (one line)

*A personal AI you can audit. June remembers you, forgets gracefully, explains
every action, and never phones home.*

June's center of gravity is the user, not the task. She inverts a coding agent's
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

- **Memory (`packages/brain/.../memory`).** One `june.db` behind one
  `MemoryManager` facade over three stores: structured SQLite rows, a sqlite-vec
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
  Defense is **structural** — there is (as of v0.1) no content-based
  injection-phrase detector.

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

- No account, no signup, no cloud sync by default.
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
