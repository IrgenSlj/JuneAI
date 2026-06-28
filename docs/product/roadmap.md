# Product Roadmap

This is the detailed product roadmap. The short public summary lives at
[`../../ROADMAP.md`](../../ROADMAP.md). The authoritative, decision-by-decision
working plan is [`rebuild-plan.md`](rebuild-plan.md) — this roadmap sequences it and
tracks status; where the two differ, the rebuild plan wins.

## Direction

June is a personal assistant whose center of gravity is the user, not the task.
She remembers what matters, forgets what doesn't, tells the truth, knows when to
stay quiet, and never does anything the user can't see.

The UI should feel simple. The system should be technically rigorous, local-first,
and visibly private. June is tuned for a known model roster (Gemma 4 + Gemini),
not abstracted to be model-agnostic.

The current implementation checklist lives in
[`development-plan.md`](development-plan.md). It consolidates the 2026-06-28
external review into sequenced, shippable slices. Where product worldview differs,
this roadmap and `rebuild-plan.md` remain authoritative; where day-to-day
implementation order differs, the development plan is the working checklist.

This supersedes the earlier "personal operating layer / Quick Capture / Daily
Home" framing (ADR 0013, ADR 0014, the agentic-pivot plan, the v0.1.1
scheduled-development plan), which are retained as historical context only.

## Current Shipped Surface

### Web PWA

The PWA remains the primary shared UI surface: chat, memory, tasks, skills,
system activity, settings, and setup. Installable via browser PWA support.

### Desktop Shell

The Tauri desktop shell builds and produced the v0.1.0 Apple Silicon DMG. It adds
native capabilities the browser cannot: Ollama supervision, native notifications,
system tray, global hotkey, autostart. The current public DMG is ad-hoc signed and
not notarized; signed distribution is deferred until external users justify the
Apple Developer Program cost. The desktop Python sidecar (so the `.app` runs a
brain without a separately-started process) remains open.

### Brain, memory, skills

Three-store memory (one SQLite db: structured rows + a sqlite-vec vector index + a graph) behind one `MemoryManager`; MCP skill
supervisor with bundled skills; scheduler and notification bus. The Tier 1 spine
(below) is built: model-specific providers, salience recall, layered context with
anchored compaction, the honest character block, and per-turn provenance with a
visible cloud boundary. The hand-written loop is the one live engine (ADR 0018);
the LangGraph engine has been removed.

## Active Track — Tier 1: The Spine (built)

Theme: **remember what matters, hold the thread, one honest voice, a visible cloud
boundary — running on local Gemma 4.**

The seven spine modules (C.0-C.6) are implemented, tested, and on `main`; each shipped
with its model-judgment fallback in the same PR. The hand-written loop is the live
chat path — provider layer, layered context, character block, difficulty router, and
capability probe flow through it; the LangGraph engine was removed (ADR 0018). The
active work is the reshape + targeted rewrite tracked in
[rebuild-plan.md](rebuild-plan.md).

The C.0-C.6 acceptance criteria below are all met; they remain as the regression bar:

| Task | What it adds | Done when |
|---|---|---|
| **C.0** Portable data dir + manifest | One documented, versioned folder that *is* June | Round-trip (copy folder → reload → state intact); version mismatch migrates, never crashes; all paths resolve through `layout.py` |
| **C.1** Model-specific provider layer | Gemma 4 + Gemini behind roles; clean seam for a third | All model access goes through a provider; role→model resolution is config-driven; no cloud call without provenance |
| **C.2** Loop behind interface + CLEAR | Hand-written loop measured against LangGraph | Same suite passes on both engines; results in `docs/experiments/loop-clear.md`; default = winner; LangGraph not yet removed |
| **C.3** Layered context + pinned state | Fixed assembly order; anchored compaction | Context never exceeds budget; pinned goal/commitments survive a 100-turn session; compaction merges, never blanks |
| **C.4** Salience-scored recall | `recency × frequency × relevance` | A relevant-but-older memory outranks a similar-but-newer-irrelevant one; access bookkeeping updates on recall; weights config-driven |
| **C.5** Honest character block | Self-authored persona, honesty immutable | Block persists/reloads; `character_update` cannot mutate `fixed`; sycophancy-drift check holds |
| **C.6** Visible cloud boundary + trace | Provenance every turn; difficulty classifier | Cloud call ⇒ provenance `cloud=true`; local-only blocks egress (asserted); a one-line rationale every turn |

**Capability probe (D17)** is plumbed in Tier 1 (`capability/probe.py`) because
the C.3 and C.5 fallbacks read it; its System-page display is Tier 2 (D.7).

**Tier 1 definition of done (the observable demo):** a user chats with June on
local Gemma 4; June recalls a relevant older fact over a merely-similar recent one
(C.4); a long conversation compacts mid-session without losing the user's stated
goal (C.3); June answers in a consistent voice that will gently disagree when
warranted (C.5); and nothing reaches the cloud without a visible provenance line,
with local-only mode provably blocking egress (C.6).

### Suggested first session

1. C.0 fully (the foundation everything writes into).
2. C.1 fully (provider layer with tests).
3. C.2 (interface + hand-written loop + CLEAR harness; LangGraph since removed, ADR 0018).

## Hardening backlog (2026-06-20)

Concrete near-term work surfaced while dogfooding the live local stack. Ordered;
each item is a small, independently shippable slice. Distinct from the Tier 2
differentiators below — this is keeping the spine honest, not adding scope.

1. **[SHIPPED] Local-first egress audit (privacy).** The local embedding model
   (`all-MiniLM-L6-v2`) pinged the HF Hub on every load and could download
   silently. Now loaded with `local_files_only=True` once cached (never contacts
   the Hub), and in Local-only mode an uncached model disables semantic recall
   instead of egressing — structured memory still works. Tested. (Moot since ADR 0019: the in-process sentence-transformer is gone; embeddings now come from local Ollama.)
2. **[SHIPPED] First-token latency UX.** The pre-first-token wait hint now reads
   "Thinking locally…" when the runtime is on-device, surfacing the privacy story
   at the moment of doubt. (Tier-routing of quick factual turns to a faster model
   is deferred — a separate router-judgment change, not just UX.)
3. **[SHIPPED] Build/version surface.** `build_version()` (JUNE_BUILD_SHA override,
   git short-SHA fallback) is exposed as `SystemStatus.version` and shown as a
   quiet "build <sha>" tag in the runtime chip.
4. **[SHIPPED] Loopback API hardening (DNS-rebinding).** Closed via Host-header
   validation (`TrustedHostMiddleware`, allowlist `localhost`/`127.0.0.1`/
   `testserver`, `JUNE_API_ALLOWED_HOSTS` override) rather than a per-session
   token: the project already has opt-in api-key auth (`JUNE_API_AUTH_ENABLED`)
   for exposed deployments, and a token "adds breakage but no security" for the
   single-user loopback case. CORS blocks cross-origin reads; Host validation
   blocks DNS-rebinding. A rebinding request carries the attacker domain in Host
   and is rejected 400.
5. **[SHIPPED] PWA-in-dev verification.** Verified safe by inspecting the
   generated dev service worker (`dev-dist/sw.js`): it precaches only the
   navigation shell (`"/"`), not the client JS/CSS bundles, so Vite serves
   modules fresh and HMR is unaffected. `registerType: "autoUpdate"` runs
   skipWaiting + clientsClaim + cleanupOutdatedCaches, so the one cached shell
   self-updates (worst case: a single full reload of `/` serves the prior shell
   before the new SW takes over). No flag/gate needed; rationale recorded in
   `apps/web/vite.config.ts`.

## Next Track — Tier 2: Differentiators

Trigger: Tier 1 is complete and has been used in real dogfooding. Build simple,
observe, refine; do not over-specify the perfect rule in the abstract.

- **D.1** Temporal context layer — passive time-awareness folded into the
  assembler; no process, no timer.
- **D.2** Event-driven + deferred proactivity + OS-notification scheduler — June
  never cold-starts a session; real-world events may wake her, the clock never
  does; hard deadlines become pre-written OS notifications.
- **D.3** Self-monitor + idle housekeeping + reference-context diffing — hygiene
  (dedup/re-index/decay) when truly idle; idle inference forbidden.
- **D.4** Conservative, reversible forgetting — relevance + decay budget biased
  hard toward retention; reversible and visible in the memory browser.
- **D.5** Durable task ledger built around continuity — append-only log; tasks
  reconstructable after crash; modeled as promises, not terminating TODOs.
- **D.6** Native memory graph — custom HTML5 canvas + ~40-line force simulation,
  opened on demand in the Memory page; no graph library.
- **D.7** System page — self-monitor + capability profile in plain language;
  on-demand and calm, not an always-pulsing dashboard.
- **D.8** Privacy Mode 2 — client-side-encrypted backup of the whole data dir;
  OS keychain day-to-day, passphrase only when moving machines; vetted crypto.
- **D.9** Privacy Mode 3 — Google (Gmail/Calendar/Drive/Maps) as per-service MCP
  skills; granted once, revocable anytime, always visible; reads first, writes
  per-action.

## North Star — Tier 3 (design intent; not built yet)

- Full live brain map (only after D.6 proves the canvas approach).
- Self-improvement Rungs 2–3 — **capability-blocked, not just safety-gated**;
  revisit only when local models are demonstrably strong enough.
- **Rung 4 (core self-modification): permanently excluded.**
- Daily/weekly life loops — obey D.2/D.3 (run when the user shows up, never on a
  timer).
- Page IA rename (Memory / Tasks / Trust) — cosmetic until Tier 1–2 land.
- CLEAR as standing practice — the 70% compaction threshold and salience weights
  are guesses until measured.

## Explicitly Rejected

- **Heartbeat-as-cron** — waking every N minutes to scan and maybe act. (Reverses
  the scheduler-driven proactivity that earlier exploratory work introduced.)
- **Obsidian (or any external app)** as the place to view June's memory.
- **A graph-visualization library** — the force layout is custom canvas code.
- **Hand-rolled cryptography** — the one place to use a vetted library.
- Copying competitor features wholesale — study patterns, write June's own.

## Carried-over Non-Goals

- No account-required modes. No cloud memory as the default. No team workspaces.
- No third model provider. No paid hosting dependency. No always-on audio capture.
