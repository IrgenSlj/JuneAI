# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

It has two halves on purpose. **Part 1 (Engineering)** is how the repo builds,
tests and is wired; it survives a change of product direction. **Part 2
(Direction)** is what June is *for* right now; it is expected to be replaced
wholesale when direction changes. Keep that boundary when editing this file.

Where this file conflicts with `docs/CURRENT.md` (the authoritative state page),
the active plan named there, `docs/product/overview.md`, or `docs/vision.md`,
those win (and fix this file) — **unless a newer direction brief exists; see
"Changing direction" below.**

---

# Part 1 — Engineering (durable)

## Build, test, run

- `./tools/bootstrap.sh` — install the Python workspace (uv, venv at `packages/brain/.venv`) and pnpm deps.
- `./tools/check.sh` — THE gate. Brain+API pytest, frontend `pnpm check`, OpenAPI codegen drift, ruff, doc hygiene, the one-privacy-predicate grep, and a narrow mypy gate (`operator`/`call-arg` codes only). `set -e`; never read its result through a pipe like `| tail` (that masks the exit code). Must pass before every push.
  - Opt-outs: `JUNE_CHECK_FRONTEND=0`, `JUNE_CHECK_CODEGEN=0`, `JUNE_CHECK_MYPY=0`. Opt-ins: `JUNE_E2E=1` (Playwright), `JUNE_CHECK_RETRIEVAL_BENCH=1` (needs Ollama, report-only), `JUNE_CHECK_SMOKE=1` (frozen sidecar).
- Single test — the gate sets three env vars the tests expect; set them yourself:
  ```
  JUNE_DATA_DIR=.tmp/june-test-data JUNE_SKIP_MODEL_CHECK=1 JUNE_SKILLS_DISABLED=1 \
    packages/brain/.venv/bin/python -m pytest packages/brain/tests/unit_tests/test_invariants.py -q
  ```
  Append `::test_name` or `-k expr` to narrow. API tests live in `packages/api/tests`.
- `./tools/codegen.sh` — regenerate `packages/api/openapi.json` and `packages/ui/src/api/types.ts`; run after any Pydantic schema or API route change, or the drift check fails.
- Run locally: `./tools/run.sh` (starts Ollama, pulls the default Gemma, API, web; Ctrl-C tears down). Or by hand: `packages/brain/.venv/bin/june-api` and `pnpm --filter @june/web dev` (http://localhost:5173). `./tools/preflight.sh` checks health without starting anything (`tools/dev.sh` is a deprecated alias).
- Measurement harnesses (not gated, need a model): `tools/tool_selection_harness.py`, `tools/reliability_harness.py`, `tools/retrieval_bench.py`, `tools/injection_bench.py`. Results are written up under `docs/experiments/`.

## Architecture

Monorepo: Python workspace (uv) + pnpm workspace.

- `packages/brain` — `june_brain`, the Python "Brain". Everything model-facing lives here.
- `packages/api` — `june_api`, FastAPI REST + SSE over the brain. Routes in `june_api/routes/`, Pydantic schemas in `june_api/schemas/` (the source of the OpenAPI contract).
- `packages/ui` — shared Svelte components plus the **generated** TypeScript API client (`src/api/types.ts`; never hand-edit).
- `apps/web` — SvelteKit PWA (routes: chat, memory, tasks, skills, settings, system, setup, help). `apps/desktop` — Tauri shell that runs a frozen `june-api` sidecar. `apps/landing` — marketing site.
- `skills/` — bundled MCP servers (calendar, files, research, telegram), supervised by `june_brain/skills/`.

**A chat turn, end to end:** `routes/chat.py` -> `loop.engine.get_loop()` -> `loop/handwritten.py` (`HandwrittenLoop`, the only engine; ADR 0018). The loop asks `context/assembler.py` (`ContextAssembler`) for the prompt (recall, pinned state, compaction), calls a provider from `providers/` (registry + `providers.toml`; Gemma via local Ollama, Gemini for cloud; provider-native tool calling, ADR 0020), dispatches tools through `tools_base.py` (June's own abstraction, no LangChain) gated by `guard/` (action classification, injection framing, SSRF, redaction), and returns a `TurnResult` with a `TurnProvenance` frame that the route streams as SSE events. Post-turn, the route runs memory extraction in a background task. `router/difficulty.py` picks the model tier per turn. Loop choices (tier, tools) flow as data through a fixed shape.

Note: `ContextAssembler` builds the live chat prompt; `build_system_prompt` is read only by the scheduler. A rule that must reach chat has to go into the assembler.

**Storage:** one SQLite `june.db` under `<datadir>/memory/` holding structured rows, a sqlite-vec vector index, and a graph (ADR 0019); embeddings via local Ollama. `MemoryManager` is the highest-level seam but **not the only one** — many modules open connections directly through `memory/sqlite.py`, so treat that module as the store boundary, and do not write new code that claims the manager mediates everything.

**Cross-cutting modules worth knowing before touching anything:**
- `privacy.py` — the single egress decision (`local_only()` / `egress_permitted()`), fails closed. Only it, `config_store.py`, and the settings/system routes may read `get_privacy_dial` (grep-enforced in `check.sh`).
- `failure.py` — `degrade_quietly` / `fail_closed`; ruff `S110` forbids a silent `pass`, so swallowed failures must use these.
- `trust/` — the signed Trust Ledger (ADR 0022): receipts for memory writes and external reads.
- `silence/` — the silence model (ADR 0023): when June speaks unprompted, event-driven only.
- `tasks/` — promises (standing intentions), exposed at `/tasks`.
- `tools.py` — `JUNE_TOOLS` / `JUNE_TOOLS_GEMMA`, plus `RETIRED_TOOL_NAMES` (denylist the tool merge enforces — add to it when retiring a tool) and `SKILL_OWNED_TOOL_NAMES`.
- `mcp/` — June as an MCP memory server for third-party clients (ADR 0030).

## Engineering invariants (enforced by the gate)

These are mechanics, not product direction; they bind any direction unless an ADR explicitly retires them and the enforcing test is deliberately deleted in the same commit.

- **Privacy is visible in code.** Every cloud/external call is surfaced in the per-turn provenance frame. Local-only mode blocks egress, both directions: `is_network_tool()` delegates to the guard's `classify_action()`, so outbound writes (`send_`, `post_`, `email_`...) are blocked under Local-only and listed in `provenance.egress`. Enforced by `test_invariants.py` and the `get_privacy_dial` grep.
- **No new runtime dependency that could be implemented in-house** (exception: cryptography — always vetted libraries). `test_invariants.py::test_the_runtime_dependency_set_is_deliberate` pins brain and API deps with a reason each; adding one is a deliberate two-place edit.
- **The harness core is fixed and never self-modified.** The only self-edit surface is `character_update`, which can write only `LearnedTraits`. Nothing mechanically stops a future path writing to `loop/`; that stays a review-time rule.
- **Honesty is not adjustable.** Candor, care and the behavioral safety floor (not a therapist/doctor/lawyer/financial advisor; care not diagnosis for distress; no engagement-maximizing metric; sensitive memories surfaced by the user, not volunteered) are `FixedTraits` in `character/block.py`. Pinned by `test_character.py` and `test_invariants.py`.
- **Graceful degradation ships in the same change** as any model-judgment feature (compaction, salience, shaping, tool dispatch, classification). Not mechanically enforceable; the gate only checks the *shape* via `S110`.
- **An invariant is only real if something enforces it.** New invariants land with a test in `packages/brain/tests/unit_tests/test_invariants.py`, or a grep in `check.sh`; otherwise a "known gap" note naming the slice that closes it.

## Conventions

- No emojis in README or documentation.
- June is "June", or "it" — never "she". Enforced on product-voice docs by `tools/check_doc_hygiene.py` (README.md, docs/CURRENT.md); source and tests are out of scope on purpose.
- "No account needed", not "no account".
- Document any privacy-boundary change explicitly.
- ADRs (`docs/decisions/`) are append-only; supersede by writing a new one.
- Work in small validated slices: one slice -> `./tools/check.sh` green -> one commit -> push. Add/update tests for behavior changes in `packages/brain` and `packages/api`.
- Measure before claiming: product claims about reliability cite a harness run in `docs/experiments/`, not an assumption.

---

# Part 2 — Direction (replace when direction changes)

## What June is (current)

A personal assistant whose center of gravity is the user, not the task: it
remembers what matters, forgets what doesn't, tells the truth, knows when to stay
quiet, and never does anything the user can't see. Defined by the four
inversions of a coding agent (ADR 0015): **defers, not verifies**; **continues,
not completes** (tasks are standing "promises"); **forgets, not accumulates**;
**stays quiet, not fast** (event-driven, never on a timer — ADR 0016).

Promises render as **Promises** in the UI. A blocked promise carries explicit
`blocked_reason`, `next_action`, and any `final_deliverable`; the UI never infers
user-facing state from trace text.

## Current plan

- `docs/CURRENT.md` — what is true now, and "resume here".
- `docs/product/v0.4-development-plan.md` — plan of record. Stream D
  (correctness and coherence, from `docs/product/repo-audit-2026-08-18.md`) is
  complete bar two Svelte admin pages; the top open item is `remember` call-rate
  (~73% end to end on the local model).
- `docs/product/overview.md`, `docs/vision.md`, `ROADMAP.md` — product truth and worldview.
- Superseded plans (`JUNE_V02_BRIEF.md`, `v0.2-execution-plan.md`, `v0.3-*`, `rebuild-plan.md`, `development-plan.md`) — do not act on them.

## Do NOT reintroduce (abandoned under the current direction)

- The v1 life-coach tool surface (removed in D.5a). June's model-callable memory surface is the four tools of ADR 0032: `remember`, `forget`, `list_promises`, `update_promise`.
- Heartbeat/cron/timer-driven proactivity or daily orchestration (ADR 0016). The scheduler exists only for user-requested, deterministic jobs.
- Quick Capture / personal operating layer / event-ledger / capture->classify->approve pipeline (superseded by ADR 0015).
- Obsidian or any external app as the memory viewer (replaced by a native on-demand graph).
- Shopping/chores domain features.

## Changing direction

A brief that changes June's direction outranks everything in Part 2 and the
current plan docs from the moment it lands. When it does, land the change in
this order, each as its own slice:

1. **ADR first.** Write a new ADR in `docs/decisions/` that states the new
   direction and names which earlier ADRs it supersedes (likely 0013–0016,
   0023, 0032 — check each). Never edit old ADRs.
2. **Triage the invariants.** For each item in "Engineering invariants", decide
   keep / change / retire, explicitly, in the ADR. Retiring one means deleting
   its enforcing test or `check.sh` grep in the same commit — never leave a test
   pinning a rule the product no longer holds, and never silently weaken one.
3. **Re-point the docs.** New plan of record in `docs/product/`; move the old
   plan into `docs/archive/` with a line in `docs/archive/README.md`; rewrite
   `docs/CURRENT.md`'s header and "resume here"; update `overview.md`,
   `vision.md`, `ROADMAP.md` as the brief requires.
4. **Rewrite Part 2 of this file** to match. Part 1 changes only where the
   architecture actually changes.
5. **Then code.** Removed surfaces go on `RETIRED_TOOL_NAMES` (tools) or the
   "Do NOT reintroduce" list (directions), so the old direction cannot creep back.

Until step 1 lands, if the brief and the current docs disagree, ask rather than
guess which one governs a given change.
