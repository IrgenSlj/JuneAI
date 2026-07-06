# CLAUDE.md — working agreement for June AI

Read this first, then the development plan. This file is the short, durable
orientation for any agent working in this repo. Where it conflicts with
`docs/product/development-plan.md` (the active implementation plan),
`docs/product/overview.md` (what June is), or `docs/vision.md` (the durable
worldview), those win (and fix this file).

## What June is

June is a personal assistant whose center of gravity is the user, not the task:
she remembers what matters, forgets what doesn't, tells the truth, knows when to
stay quiet, and never does anything the user can't see.

The four inversions of a coding agent (ADR 0015) define her:
1. **Defers, not verifies** — validates *with* the user; human-in-the-loop is core.
2. **Continues, not completes** — tasks are standing intentions ("promises"), not terminating TODOs.
3. **Forgets, not accumulates** — forgetting is first-class, conservative, reversible; the user is the source of truth.
4. **Stays quiet, not fast** — acts when the user speaks or the world changes, never on a timer.

## Canonical direction

- `JUNE_V02_BRIEF.md` — the active v0.2 lead plan (workstreams W0–W8), with
  `docs/product/v0.2-execution-plan.md` (sequencing) and `docs/RECONCILIATION.md`
  (brief-vs-reality). `docs/CURRENT.md` is the authoritative state page. (The old
  `docs/product/development-plan.md` and `rebuild-plan.md` are superseded — history only.)
- `docs/product/overview.md` — the product truth: June is a trusted continuity
  engine, not primarily a chat app.
- `docs/vision.md` — the durable product worldview (the four inversions, the non-negotiables).
- ADRs: `docs/decisions/0015` (four inversions), `0016` (event-driven, no heartbeat), `0017` (model-specific providers). ADRs are append-only; supersede by writing a new one.
- `ROADMAP.md` / `docs/product/roadmap.md` sequence the public product tracks.
  The Tier 1 spine is built; current development grows Home continuity,
  Promises, Memory governance, Trust, Skills permissions, and Time.

## Repo layout

Monorepo. `packages/brain` (Python "Brain": loop, providers, memory, context, character, router, scheduler, skills), `packages/api` (FastAPI REST+SSE), `packages/ui` + `apps/web` (SvelteKit PWA), `apps/desktop` (Tauri shell), `skills/` (MCP servers). Stores: one SQLite `june.db` (structured rows + a sqlite-vec vector index + a graph) behind one `MemoryManager`, under `<datadir>/memory/` (ADR 0019); embeddings via local Ollama.

The brain's harness loop lives in `packages/brain/src/june_brain/loop/`. The hand-written loop (`handwritten.py`) is the one engine and the live chat path (ADR 0018; the LangGraph engine and its flags were removed in the rebuild). Loop choices (tier, tools) flow as data through a fixed shape; the shape itself is never self-modified. Tools use June's own abstraction in `tools_base.py` (no LangChain).

Promises live in `packages/brain/src/june_brain/tasks/`, are exposed through
`/tasks`, and are rendered as **Promises** in the UI. A blocked promise must carry
explicit `blocked_reason`, `next_action`, and any `final_deliverable`; do not make
the UI infer user-facing state from trace text.

## Build, test, run

- `./tools/bootstrap.sh` — install Python workspace + pnpm deps.
- `./tools/check.sh` — THE gate: brain+api pytest, frontend `pnpm check`, OpenAPI codegen drift, ruff, and a narrow mypy gate. It exits non-zero on any failure (`set -e`); never read its result through a pipe like `| tail` (that masks the exit code). Must pass before every push.
- `./tools/codegen.sh` — regenerate the OpenAPI client; run after any Pydantic schema or API route change, or the drift check fails.
- Run locally: `packages/brain/.venv/bin/june-api` and `pnpm --filter @june/web dev` (http://localhost:5173).

## Invariants (do not break)

- **Privacy is visible in code.** Every cloud/external call is surfaced in the UI before and after (the per-turn provenance frame). Local-only mode blocks egress. No silent network calls.
- **Honesty is not adjustable.** Personalization shapes tone, never erodes candor into sycophancy.
- **The harness core is fixed** and never self-modified; June evolves character/skills/tuning on top of it.
- **No new dependency that can be implemented customly** (one exception: cryptography — always use vetted libraries, never hand-roll).
- **Graceful degradation ships in the same change** as any model-judgment feature (compaction, salience, shaping, tool dispatch, classification).
- **Behavioral safety floor:** June is not a therapist/doctor/lawyer/financial advisor; responds to distress with care, not diagnosis; no engagement-maximizing metric; sensitive memories are surfaced by the user, not volunteered.

## Do NOT reintroduce (explicitly abandoned directions)

- Heartbeat-as-cron / timer-driven proactivity / daily orchestration (ADR 0016). The scheduler exists only for user-requested, deterministic jobs.
- Quick Capture / personal operating layer / event-ledger / capture->classify->approve pipeline (superseded by ADR 0015).
- Obsidian (or any external app) as the place to view June's memory — replaced by a native on-demand graph.
- Shopping/chores domain features (remnants of the abandoned daily-orchestration framing).

## Conventions

- No emojis in README or documentation.
- Keep PRs focused; add/update tests for behavior changes in `packages/brain` and `packages/api`.
- Document any privacy-boundary change explicitly.
- Work in small validated slices: one slice -> `./tools/check.sh` green -> one commit -> push.
