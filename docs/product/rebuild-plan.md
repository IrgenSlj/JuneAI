# June AI — Rebuild & Repo Reshape Plan

> **SUPERSEDED (2026-07-06).** The rebuild described here shipped; its open items
> are now carried by [`JUNE_V02_BRIEF.md`](../../JUNE_V02_BRIEF.md) (the lead
> plan). Current project state is [`docs/CURRENT.md`](../CURRENT.md). The
> "single authoritative working plan" claim below is superseded by those two.
> Retained for history only. (Not physically archived because append-only ADRs link it.)

> **Authority:** This document is the single authoritative working plan. It
> supersedes the earlier `build-spec.md` (retired during repo consolidation;
> recoverable from git history). The durable product worldview lives in
> `docs/vision.md`. Every deliberate decision is recorded as a new ADR (Part 8).
> Live progress is now tracked in `docs/product/development-plan.md`. The old
> root `REBUILD.md` checklist was removed during the 2026-06-28 documentation
> consolidation; use git history if you need the historical checklist.
> **Prepared:** 12 June 2026. **Repo:** `IrgenSlj/JuneAI`. **Owner:** Irgen Salianji.

---

## PART 0 — THE VERDICT THAT SHAPES EVERYTHING (read first)

This is a reshape and targeted rewrite, NOT a greenfield rebuild. The spine
(handwritten loop, layered context, salience recall, character block,
provenance, glass-box trace, test discipline) is good and is June's
differentiator. What is broken is everything around the spine:

1. **Distribution** — only developers can install June.
2. **Day-one value** — June starts with an empty memory; no import path.
3. **Injection-grade trust** — egress visibility but no injection defense.
4. **Dead weight** — LangGraph/LangChain, ChromaDB + sentence-transformers,
   regex difficulty router, a 1,160-line `memory/manager.py` god module.

Do not rewrite the API layer, the SvelteKit app, the Tauri shell, the MCP skill
architecture, the memory schema, or the loop shape. Reshape the repo, delete the
dead weight, rewrite the four weak modules, build the three missing product
layers (install, bootstrap, defense).

**Strategic one-liner:** June wins on trust made visible — installable by a
non-developer, valuable on day one, and provably safe against the failure mode
that burned the rest of the market.

---

## PART 1 — CURRENT STATE AUDIT

### 1.1 STAYS (do not touch except as named in sessions)
Handwritten harness loop, layered context assembler, anchored compaction,
salience recall, character block, glass-box trace + activity terminal,
provenance + privacy dial + egress gating, three-store memory schema behind
MemoryManager, FastAPI surface, SvelteKit PWA + shared ui + design tokens,
Tauri shell + ollama.rs, MCP skills as supervised stdio subprocesses,
scheduler, test suite + check.sh single gate, ADR trail.

### 1.2 GOES (deleted / replaced / relocated)
- langgraph/langchain* + graph.py + loop/langgraph_loop.py + JUNE_CHAT_USE_HARNESS
  / JUNE_LOOP_ENGINE flags — S1.
- chromadb + sentence-transformers -> sqlite-vec + Ollama embeddings — S2.
- Regex difficulty classifier heuristics — S4.
- estimate_tokens 4-char/token heuristic as sole counter — S4.
- packages/june-skill-telegram -> skills/telegram — S1.
- "June AI logo.png" -> assets/logo.png — S1.
- Superseded plan docs removed during documentation consolidation — S1.
- apps/landing/index.html placeholder -> real page — S13.

### 1.3 Defects to fix opportunistically
Reliability variance (S4/S5, metric S10); `<think>` on every turn (S4);
DuckDuckGo fallback flaky (S6 permissioned search + Brave key); dev.sh misnamed
(S1 -> preflight.sh); june-brain unpublished (S9).

---

## PART 2 — TARGET STATE

### 2.3 Architecture deltas
1. One loop engine. JUNE_LOOP_ENGINE removed; trace store is the safety net.
2. One storage engine. SQLite owns facts, structure, graph, AND vectors
   (sqlite-vec vec0). Embeddings from Ollama /api/embed with a hash-cached store.
3. Structured tool calls (Ollama structured outputs for Gemma, native function
   calling for Gemini) normalized behind providers/base.py; prose-parse fallback.
4. A guard layer between model and dispatch: untrusted-content framing,
   consequential-action approval gates, skill permission manifests enforced.
5. Model-based router with heuristic fallback, multilingual by construction.
6. Memory bootstrap importers (chatgpt/claude/markdown/ics) on first run.
7. Desktop app becomes the product: managed Ollama, embedded API sidecar,
   notarized.
8. cloud-capable role accepts any OpenAI-compatible endpoint (ADR 0018/0023);
   local roles remain Gemma-specific per ADR 0017.

(Target repo + brain package shapes: see Part 2.1/2.2 of the source brief and the
current implementation status in `development-plan.md`.)

---

## PART 3 — INVARIANTS (enforced every session)
1. Privacy visible in code; no silent network calls; boundary changes documented.
2. Efficiency = privacy, one axis; prefer local; no unrequested cycles.
3. Harness core fixed, never self-modified.
4. No new dependency implementable customly (crypto excepted). This plan removes
   four heavy deps, adds two small ones (sqlite-vec, defusedxml if needed) — net
   strongly negative. Every add ledgered in its session.
5. Honesty not adjustable; FixedTraits immutable.
6. Graceful degradation ships in the same change as any model-judgment feature.
7. Gate: check.sh green before every push; codegen.sh after schema/route changes;
   small validated slices, one commit each.
8. Behavioral safety floor unchanged; imported sensitive content gets the same
   conservatism flags as extracted memories.
9. Four inversions remain product identity: S6 defers, S11 continues, S5 forgets,
   S12 stays quiet.
10. No emojis in README or docs. ADRs append-only.

---

## PART 4-7 — SESSIONS

The historical per-slice rebuild detail lived in the removed root `REBUILD.md`.
Current work is sequenced in `development-plan.md`. Summary map:

| # | Session | Outcome | New ADR |
|---|---|---|---|
| S0 | Baseline + tracking | Tag, measured baseline | — |
| S1 | Dead weight + reshape | LangGraph gone; repo shape final | 0018 |
| S2 | sqlite-vec storage | One storage engine; huge install drop | 0019 |
| S3 | Memory decomposition | manager.py facade < 250 ln | — |
| S4 | Router v2 + tokens + gated reasoning | Multilingual routing; faster trivial turns | — |
| S5 | Structured tool calls + tunable salience | Reliability cv down; knobs visible | 0020 |
| S6 | Guard layer | Framing, approval gates, manifests, security-model.md | 0021 |
| S7 | Memory bootstrap | Day-one value; importers + wizard step | — |
| S8 | Desktop distribution | Notarized DMG, managed Ollama, sidecar | 0022 |
| S9 | Open cloud role + PyPI | Custom endpoints; june-brain published | 0023 |
| S10 | Benchmarks | LoCoMo numbers + reliability CI | — |
| S11 | Promises ledger | Inversion 2 shipped | (write one) |
| S12 | Deferred proactivity | Inversion 4 shipped; no heartbeat | (write one) |
| S13 | Graph + landing + Telegram | Launch-ready face | — |

## PART 8 — ADR QUEUE
0018 One loop engine · 0019 Single-engine storage + Ollama embeddings ·
0020 Provider-native tool calling · 0021 Guard layer · 0022 Desktop-first
distribution · 0023 Open cloud-capable role. House format, names what it
supersedes, append-only.

## PART 9 — MIGRATION AND COMPATIBILITY
Data dir manifest bumps once at S2 (vector migration). S7/S11 add tables
additively. Never break an existing data dir; forward-only, logged, fixture
tested. Removed env vars detected at startup for one minor version, single log
line, then ignored. .env.example + docs/setup/environment.md updated in every
config-touching session.

## PART 10 — RISKS AND FALLBACKS
PyInstaller vs native ext -> python-build-standalone fallback (prototype bundle
first). Ollama structured outputs weak on small gemma -> prose fallback is the
path; ADR records it. Embedding change degrades recall -> re-embed from shadow
copies, tunable weights, keep chroma.bak. Approval nagginess -> per-conversation
allow-list; taint-flagged network writes non-waivable. Solo bandwidth -> every
session leaves main shippable. Apple notarization -> one-time, 99 USD/yr,
non-optional.

## PART 11 — DEFINITION OF DONE (Phase 1)
A stranger with a MacBook and no terminal skills can: download one DMG, open it
without warnings, optionally import their ChatGPT history, ask June something
personal and get a locally-generated answer citing an imported memory, watch the
activity terminal, flip to local-only and verify nothing leaves, and read
docs/security-model.md to understand why an injected web page cannot silently
empty their disk.
