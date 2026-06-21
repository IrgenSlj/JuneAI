# REBUILD — living checklist

Cross-session continuity anchor for the June AI rebuild & repo reshape.
Authority: `docs/product/rebuild-plan.md` (the working plan) is the single
source of truth; the durable worldview lives in `docs/vision.md`. Check a line
off as its commit lands. Every slice ends with `./tools/check.sh` green and one
commit.

Baseline: tag `v0.2.0-prereshape`, metrics in
`docs/experiments/baseline-2026-06.md`. Execution playbook for S2 onward
(sizing, dependencies, decisions, tomorrow's actions):
`docs/product/rebuild-sessions.md`.

## Out-of-band work (2026-06-21)

Not part of the numbered slices below — user-driven, all on `main`:
- Imported the claude.ai design artifact to `docs/design/artifact/` (+ `docs/design/master-brief.md`). The shipped chat surface already realizes the prototype, so no port was needed.
- Consolidated docs onto this plan: retired `docs/product/build-spec.md` (recoverable from git history), removed `docs/archive/` + `docs/plans/`, purged abandoned-direction copy from live surfaces. `rebuild-plan.md` + this file are the single source of truth; `vision.md` is the durable worldview.
- Shipped trust/transparency UI (no faked data): System glass-box trace browser (`GET /system/traces`), `GET /system/capability` + verdict table, per-skill declared `model_policy`, light Memory polish.
- Known follow-up: a chat→trace deep-link needs a `turn_id` on the chat SSE (`done`/`provenance` carry none today). Details in CHANGELOG `[Unreleased]`.

## Phase 1 — Trust + Distribution

### S0 — Baseline, branch discipline, tracking
- [x] S0.1 Tag `v0.2.0-prereshape`; create REBUILD.md
- [x] S0.2 Baseline metrics snapshot (`docs/experiments/baseline-2026-06.md`)

### S1 — Dead weight removal and repo reshape  (ADR 0018)
- [x] S1.1 Delete LangGraph path (graph.py, langgraph_loop.py, flags); port/rewrite loop integration tests
  - [x] S1.1a Relocate shared graph.py helpers -> loop/agent_helpers.py; repoint wiring.py
  - [x] S1.1b Rewire scheduler off graph.run_agent_sync onto provider stack
  - [x] S1.1c-1 Remove LangGraph chat-route fallback (_iter_events, get_agent, USE_HARNESS flag)
  - [x] S1.1c-2 Drop agent-lifecycle hooks in settings/setup; reconcile supervisor directly in skills route
  - [x] S1.1c-3 Rewire tasks/runtime.py off get_or_create_agent onto provider stack
  - [x] S1.1c-4 Delete graph.py engine + langgraph_loop.py + engine branch + loop/__init__/experiment refs + __init__ exports; repoint/delete ~10 tests
- [x] S1.2 Drop langgraph/langchain* deps; re-lock; record install delta
  - [x] S1.2a Replace langchain @tool/StructuredTool/Command with custom Tool abstraction (tools.py, skills loader/supervisor, models.py)
        DESIGN: new `tools_base.py` — a `Tool` dataclass (name, description, args:dict, func, injected:set)
        with `.invoke(dict)`, plus a `@tool` decorator that introspects the signature (inspect) to build
        name/description(docstring)/args, treating `Annotated[T, Inject]` params (replacing langgraph
        InjectedState) as injected (excluded from advertised args, filled at dispatch). Matches the surface
        wiring.py already uses (.name/.description/.args/.invoke). Command/ToolMessage/InjectedToolCallId
        returns -> plain string returns (UI-state Command is already inert in the handwritten dispatch).
        Skill loader/supervisor: replace StructuredTool wrappers with the same Tool type. models.py:
        replace ChatOpenAI build_chat_model (only _verify_round_trip uses it) with a provider-stack call.
  - [x] S1.2b Drop the four deps; re-lock; record install delta
- [x] S1.3 Move `packages/june-skill-telegram` -> `skills/telegram`
- [x] S1.4 Repo hygiene: logo -> assets/, dev.sh -> preflight.sh (shim), docs/archive/ + INDEX.md, CLAUDE.md update
- [x] S1.5 Update README architecture + tech stack (LangGraph/LangChain removed)
- [x] S1 ADR 0018 — One loop engine

### S2 — One storage engine: ChromaDB -> sqlite-vec  (ADR 0019)
- [x] S2.1 Add sqlite-vec; vec_index.py (vec0 load/upsert/KNN/delete) + extension loading + embedding_cache
- [x] S2.2 EmbeddingService (Ollama /api/embed via GemmaProvider.embed) + hash cache + degrade
- [x] S2.3 Swap VectorStore onto vec0 + EmbeddingService; rewrite test embedder contract
- [x] S2.4 Migration: manifest v1->v2 chroma archival + vec backfill + tools/migrate_chroma_to_sqlitevec.py
- [x] S2.5 Remove chromadb + sentence-transformers; re-lock; venv 1.3G -> 653M (~647M freed)
- [x] S2.6 ADR 0019 — Single-engine storage + Ollama embeddings; purge stale ChromaDB doc claims

### S3 — Decompose the memory god module
- [x] S3.1 Extract paraphrase.py, writers.py, recall.py, extractor.py from manager.py
  - [x] S3.1a paraphrase.py (six row renderers + node/edge formatters)
  - [x] S3.1b recall.py (gather_hits, sqlite keyword scan, salience rerank, feedback helpers)
  - [x] S3.1c writers.py (nine write paths, vector sync, forget/update, WRITE_HANDLERS)
  - [x] S3.1d extractor.py (extract pipeline, prompt render, JSON parse, async bridge)
- [x] S3.2 No behavior change; manager.py 199 ln (< 250 tripwire test)

### S4 — Router v2, language-aware tokens, gated reasoning
- [x] S4.1 context/tokens.py calibrated per-script counter (Latin/Greek/Cyrillic/CJK); replace estimate_tokens
- [x] S4.2 difficulty classifier: LRU cache + 300ms timeout + JSON output + source; multilingual heuristic
- [x] S4.3 Gate `<think>` by difficulty (set_reason per turn); model classifier in loop via injectable seam
- [x] S4.4 Provenance carries difficulty + source (rationale + chip); no codegen drift (dict payload)

### S5 — Structured tool calling + salience tunability  (ADR 0020)
- [x] S5.1 providers/base.py ToolSpec/ToolCall + tools/tool_calls; Gemma + Gemini native
- [x] S5.2 run_turn prefers native tool calls; prose JSON fallback retained (stream_turn = follow-up)
- [x] S5.3 Reliability harness (experiments/reliability + tools/reliability_harness.py); cv math tested, numbers pending live run
- [~] S5.4 Salience weights config-backed + env override (brain done); settings form + /memory feedback view = follow-up
- [x] S5 ADR 0020 — Provider-native structured tool calling (+ scoped follow-ups)

### S6 — Guard layer  (ADR 0021)
- [x] S6.1 guard/framing.py untrusted-content frame (central in dispatch) + standing system rule + red-team tests
- [~] S6.2 guard/actions.py action classes + taint + gate ENFORCED at dispatch + per-conversation allow-list (S6.2a/b done; approval_request SSE event + ConfirmDialog UI pending)
- [ ] S6.3 skill.toml permission manifests; loader + supervisor enforcement; /skills UI
- [x] S6.4 guard/redaction.py secret scrub in TraceStore.write; end-to-end test (key never on disk)
- [ ] S6.5 docs/security-model.md
- [x] S6 ADR 0021 — Guard layer (accepted; framing + redaction shipped, gates/manifests pending)

### S7 — Memory bootstrap: day-one value
- [ ] S7.1 memory/bootstrap/ Importer interface; chatgpt/claude/markdown/ics importers
- [ ] S7.2 Bootstrap pipeline through extractor; progress SSE; pause/resume
- [ ] S7.3 Sensitivity conservatism on imports
- [ ] S7.4 Setup wizard "Bring your history" step
- [ ] S7.5 Honest dedupe via vector layer

### S8 — Desktop distribution  (ADR 0022)
- [ ] S8.1 Sidecar API (PyInstaller / python-build-standalone) supervised by Tauri
- [ ] S8.2 Managed Ollama (detect/install/pull/supervise + RAM-based model select)
- [ ] S8.3 First-run experience + 60s glass-box tour
- [ ] S8.4 Signing + notarization in release.sh + GH Actions
- [ ] S8.5 Auto-update (signed, opt-in, egress-surfaced, local-only suppresses)
- [ ] S8.6 Windows deferred; Linux AppImage best-effort
- [ ] S8 ADR 0022 — Desktop-first distribution

### S9 — Open cloud role + publish brain  (ADR 0023)
- [ ] S9.1 providers/openai_compat.py; cloud-capable role bind to custom endpoint
- [ ] S9.2 Publish june-brain to PyPI (0.3.0, trusted publishing, README example)
- [ ] S9 ADR 0023 — Open cloud-capable role

## Phase 2 — Differentiation (after Phase 1 shipped + 2 weeks dogfood)

### S10 — Public memory benchmark + reliability regression
- [ ] S10 LoCoMo run + docs/experiments/locomo-2026.md + tools/bench.sh + CI reliability job

### S11 — Promises: the continuity ledger
- [ ] S11 Promise data model + /tasks two-register UI + recall integration (ADR: write one)

### S12 — Deferred proactivity + temporal context
- [ ] S12 now block + surface_queue drained at next conversation start (ADR: write one)

### S13 — Graph + landing + Telegram
- [ ] S13 Native memory graph + apps/landing real page + Telegram single-user binding

## Phase 3 — Launch + sustainability (own plan later)
- [ ] Launch sequence, community, first paid product (encrypted sync)
