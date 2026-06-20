# REBUILD — living checklist

Cross-session continuity anchor for the June AI rebuild & repo reshape.
Authority: `docs/product/rebuild-plan.md` (the working plan) wins over the
build spec for the duration of the rebuild. Check a line off as its commit
lands. Every slice ends with `./tools/check.sh` green and one commit.

Baseline: tag `v0.2.0-prereshape`, metrics in
`docs/experiments/baseline-2026-06.md`. Execution playbook for S2 onward
(sizing, dependencies, decisions, tomorrow's actions):
`docs/product/rebuild-sessions.md`.

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
- [ ] S2.1 Add sqlite-vec; implement store_vector.py (same interface, vec0 table)
- [ ] S2.2 Embedding provider via Ollama /api/embed + hash cache + keyword fallback
- [ ] S2.3 Migration tools/migrate_chroma_to_sqlitevec.py (auto on first start)
- [ ] S2.4 Remove chromadb + sentence-transformers; re-lock; record install delta
- [ ] S2.5 ADR 0019 — Single-engine storage + Ollama embeddings

### S3 — Decompose the memory god module
- [ ] S3.1 Extract paraphrase.py, writers.py, recall.py, extractor.py from manager.py
- [ ] S3.2 No behavior change; manager.py < 250 ln tripwire test

### S4 — Router v2, language-aware tokens, gated reasoning
- [ ] S4.1 context/tokens.py calibrated per-script counter; replace estimate_tokens
- [ ] S4.2 router/classifier.py model-based (enum, LRU, 300ms timeout, heuristic fallback)
- [ ] S4.3 Gate `<think>` by difficulty; wire ContextAssembler(reason=...) per turn
- [ ] S4.4 Provenance shows difficulty + model/fallback (codegen)

### S5 — Structured tool calling + salience tunability  (ADR 0020)
- [ ] S5.1 providers/base.py tool-call support; Gemma + Gemini native
- [ ] S5.2 Loop prefers native tool calls; prose JSON fallback retained
- [ ] S5.3 Reliability harness (10 runs, cv%); record vs baseline (target recall cv < 25%)
- [ ] S5.4 Runtime-tunable salience weights; settings field + /memory feedback view
- [ ] S5 ADR 0020 — Provider-native structured tool calling

### S6 — Guard layer  (ADR 0021)
- [ ] S6.1 guard/framing.py untrusted-content frame + red-team tests
- [ ] S6.2 guard/actions.py action classes + approval gates (SSE approval_request) + per-session allow
- [ ] S6.3 skill.toml permission manifests; loader + supervisor enforcement; /skills UI
- [ ] S6.4 Secrets hygiene audit; trace redaction test
- [ ] S6.5 docs/security-model.md
- [ ] S6 ADR 0021 — Guard layer

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
