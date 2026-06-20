# Rebuild — session playbook (S2 onward) and tomorrow's actions

Companion to [rebuild-plan.md](rebuild-plan.md) (the what/why) and
[`REBUILD.md`](../../REBUILD.md) (the live checklist). This file is the
*execution* view: what each remaining session actually takes, what it depends
on, the decisions only Irgen can make, and what to do tomorrow. Written at the
end of the S0+S1 session (LangGraph/LangChain fully removed, 19 commits).

## Where we are

- **Done:** S0 (baseline + tracking), **S1 complete** (engine deleted, tool
  abstraction replaced, four deps dropped, repo reshaped, docs corrected,
  ADR 0018). `main` is shippable; the gate is green; the source tree is
  framework-free.
- **Tag:** `v0.2.0-prereshape` marks the before state. Baseline metrics in
  `docs/experiments/baseline-2026-06.md`.
- **What S1 taught us (carry forward):** the original audit sized deletions by
  file count and missed *runtime coupling* — the LangGraph agent was load-bearing
  for task execution, the API lifecycle, and the entire tool layer. So S1 was
  ~10 commits, not "one session." Lesson for every session below: **trace the
  consumers before estimating.** The orchestrator/worker split worked: drive
  design-sensitive slices on the main thread, delegate mechanical bulk (engine
  deletion, 54-tool conversion) to Opus implementers, review + re-gate before
  each commit.

## How to run each session (the operating rhythm that worked)

1. Read the session's slices in `REBUILD.md`; grep the *consumers* of anything
   you plan to change before estimating.
2. Decompose into slices that each leave `main` green. One slice -> `check.sh`
   green -> one commit -> push. Never read the gate through a pipe.
3. Design-sensitive work (new abstractions, data-model changes, anything
   user-facing or security-relevant): do it yourself. Mechanical bulk: delegate
   to an Opus implementer with an exact per-file spec; review the diff and
   re-run the gate yourself before committing.
4. Each session that changes a Pydantic schema/route runs `codegen.sh`. Each
   session that touches config updates `.env.example` + `docs/setup/environment.md`.
5. End every session by updating `REBUILD.md` and writing the ADR if the
   session has one.

---

## Phase 1 remaining (S2-S9): trust + distribution

### S2 — One storage engine: ChromaDB -> sqlite-vec  (ADR 0019)  ·  size: L
The single biggest install win (~575 MB: torch/onnxruntime/transformers leave
with chromadb + sentence-transformers). Real risk lives here.
- **Readiness/risk:** `sqlite-vec` must load into this machine's stdlib
  `sqlite3` on **Python 3.14 / Apple Silicon** — verify the loadable extension
  imports in the brain venv *first slice, before any rewrite*. If it won't load,
  that reshapes S2 and S8 (record in ADR 0019).
- **Embeddings:** Ollama `/api/embed` with a small model (e.g. `embeddinggemma`
  or `nomic-embed-text`). Decide the default model + dimension; `run.sh` and the
  S8 managed-Ollama path must pull it. Keep the SQLite shadow copies so a
  re-embed is safe; keep `chroma.bak` until the user clears it.
- **Graceful degradation (ships same change):** if the embedder is down, recall
  falls back to SQLite keyword hits and the provenance line says so.
- **Migration:** `tools/migrate_chroma_to_sqlitevec.py`, auto-run on first
  start, fixture-tested, logged to the activity stream. Data-dir manifest bumps
  once here (the only breaking-ish migration in the whole rebuild — forward-only).
- **Decision for Irgen:** which embedding model is the default (size vs. recall
  quality). Reasonable default: `nomic-embed-text` (well-supported in Ollama).

### S3 — Decompose `memory/manager.py` (god module)  ·  size: M
Mechanical, behaviour-preserving extraction into `recall.py`, `writers.py`,
`paraphrase.py`, `extractor.py`; `manager.py` becomes a <250-line facade with a
line-count tripwire test. Tests should pass with only import-path edits — strong
delegation candidate. **Do S3 before S5/S6/S7** so those land in clean seams.
(Note: the manager is currently ~1,160 lines and references the provider
registry for extraction — confirm exact line numbers when starting.)

### S4 — Router v2 + language-aware tokens + gated reasoning  ·  size: M
- `context/tokens.py`: calibrated per-script counter (Latin ~4, Greek/Cyrillic
  ~2.5, CJK ~1 chars/tok); measure constants once against the real Gemma
  tokenizer via a `tools/` calibration script; replace every `estimate_tokens`.
- `router/classifier.py`: one constrained `local-fast` call (enum output), LRU
  cache, 300 ms timeout, **heuristic fallback survives** (multilingual greeting
  set EN/NL/EL). Provenance gains the difficulty label + model/fallback flag
  (schema change -> `codegen.sh`).
- Gate `<think>` by difficulty (trivial/standard skip it). Record trivial-turn
  latency vs. baseline.

### S5 — Provider-native structured tool calling + tunable salience  (ADR 0020)  ·  size: L
The reliability fix (recall cv 75.6% -> target <25%). Extend `providers/base.py`
with `tools`/`tool_calls`; Gemma via Ollama structured outputs, Gemini via native
function calling; **prose-JSON parse stays as fallback** (it's the path the new
`tools_base` already feeds). Add a reliability suite (10 runs, cv%). Salience
weights move to the config store (read-only on `/system`, writable in settings;
no auto-tuning — invariant 3). Needs **live Ollama** for the integration test
(skipped in CI).

### S6 — The guard layer  (ADR 0021)  ·  size: XL — most important for positioning
The anti-OpenClaw session. `guard/framing.py` (untrusted-content envelope on
every tool result + red-team regression tests), `guard/actions.py` (action
classes + approval gates via a new `approval_request` SSE event that pauses the
loop; per-conversation allow-list; taint-flagged network writes always ask),
`skill.toml` permission manifests enforced by the loader + supervisor (write
manifests for all six skills incl. telegram), secret-redaction test, and
`docs/security-model.md` (also marketing — link from README + landing). This is
where the `guard/` package and the approval UX get designed carefully — drive it
on the main thread; delegate only the manifest-writing and red-team corpus.

### S7 — Memory bootstrap: day-one value  ·  size: L
`memory/bootstrap/` importers (ChatGPT export, Claude export, Markdown folder,
`.ics`) -> the existing `extractor.py` (so it depends on S3) -> typed memories
tagged `source="import:<kind>"`. Background task with pause/resume + progress
SSE; sensitivity flags identical to live extraction (invariant 8); vector dedupe
(depends on S2). Setup wizard gains an optional "Bring your history" step;
`tools/import.py` CLI for power users. Fixture-based per-importer tests.

### S8 — Desktop distribution  (ADR 0022)  ·  size: XL — needs Irgen + money
A non-developer downloads one DMG and chats locally. **Prototype the PyInstaller
(or python-build-standalone) bundle FIRST** — native extensions (sqlite-vec from
S2) are the risk. Finish `ollama.rs` (detect/install/pull/supervise + RAM-based
model select), first-run experience + 60s glass-box tour, signing + notarization
in `release.sh` + a tag-triggered GH Actions release. Windows deferred; Linux
AppImage best-effort.
- **Needs from Irgen:** Apple Developer Program (~99 USD/yr) — non-optional for
  the trust positioning; the Developer ID cert + notarization credentials.

### S9 — Open cloud role + publish brain  (ADR 0023)  ·  size: M
`providers/openai_compat.py` (extract the shared transport the Gemini provider
already speaks); bind the `cloud-capable` role to a custom endpoint via settings
(base URL + key into keyring + model). Local roles stay Gemma-specific (ADR
0017). Publish `june-brain` 0.3.0 to PyPI via trusted publishing with a ten-line
embedding example. **Needs from Irgen:** PyPI project + trusted-publisher config.

---

## Phase 2 (S10-S13): differentiation — only after Phase 1 ships AND 2 weeks of dogfooding

- **S10 Benchmarks (M):** run the memory stack against LoCoMo, publish honest
  numbers + `tools/bench.sh`; wire the S5 reliability suite into a manual-trigger
  CI job.
- **S11 Promises ledger (L, ADR):** standing-intention data model (open/dormant/
  kept/released), user-confirmed creation (defers), `/tasks` becomes two
  registers, salience-gated recall integration. Strictly event-driven (ADR 0016).
- **S12 Deferred proactivity + temporal context (M, ADR):** passive `now` block;
  a `surface_queue` drained at the *start* of the next user-initiated
  conversation; hard deadlines via OS notifications, never a heartbeat. Sensitive
  memories never queue.
- **S13 Graph + landing + Telegram (L):** native canvas memory-graph (the launch
  hero visual), a real `apps/landing` page, and the Telegram skill hardened to
  single-user binding with a pairing code (it now lives in `skills/telegram` with
  a manifest from S6).

## Recommended order if priorities flex
S3 -> S2 -> S4 -> S5 -> S6 -> S7 -> S9 -> S8. (S3 first because it unblocks clean
seams cheaply; S2 early for the install win and because S5/S7 depend on the
vector layer; S8 last in Phase 1 because it's the most external-dependency-heavy
and benefits from everything else being stable.) The rebuild plan lists S2 before
S3; either works — S3-first only reshuffles two low-risk sessions.

---

## TOMORROW — concrete actions

**For Claude (next session), in order:**
1. Open with a **sqlite-vec load probe** in the brain venv (Python 3.14, Apple
   Silicon). If it loads, proceed with S2; if not, pivot S2's approach and record
   it in ADR 0019 before writing code. (Cheap, decisive, de-risks the biggest
   session.) — *Unless Irgen prefers S3-first per the order above; S3 is lower
   risk and a good warm-up.*
2. Whichever session: grep consumers first, decompose into green slices, delegate
   mechanical bulk to Opus, review + re-gate, push each slice.

**For Irgen — decisions to make (these unblock sessions):**
- **Embedding model default for S2** (recommend `nomic-embed-text`). 
- **Apple Developer Program** enrollment for S8 (~99 USD/yr) — start now; it has
  lead time and S8 is hollow without it.
- **PyPI** project name + trusted publishing for S9 (`june-brain`).
- Confirm the **S3-first vs S2-first** order above.

**For Irgen — when to dogfood:** Phase 2 (promises/proactivity) is gated on you
running the installed app for two weeks. That clock can't start until S8 ships a
DMG you actually use daily. So S8 is on the critical path to *everything* in
Phase 2 — treat it as the real milestone, not a victory lap.

## Entrepreneurial note
The highest-leverage thing this rebuild does is **S6 (the guard layer) + S8
(installable) + S7 (day-one value)** — that trio is the entire product thesis
("trust made visible, installable by a non-developer, valuable on day one").
Everything else (storage, router, decomposition) is necessary plumbing that makes
that trio shippable and credible. If you ever have to cut scope to hit a launch,
cut from Phase 2, never from S6/S7/S8.
