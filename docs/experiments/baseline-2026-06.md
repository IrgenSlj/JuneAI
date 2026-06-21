# Baseline metrics — pre-reshape (June 2026)

Captured at the start of the rebuild (REBUILD.md S0.2), against the tag
`v0.2.0-prereshape`. Every later session that claims an improvement measures
against this file. Numbers are from a MacBook (Apple Silicon), Python 3.14,
Ollama-backed Gemma.

## Install footprint

| Measure | Size |
|---|---|
| `packages/brain/.venv` (full Python workspace) | 1.3 GB |
| root `node_modules` | 164 MB |
| `apps/web/node_modules` | 1.6 MB |

### Heavy Python deps slated for removal (S1 + S2)

Measured under `packages/brain/.venv/lib/python3.14/site-packages`:

| Package | Size | Removed in |
|---|---|---|
| torch | 408 MB | S2 (via sentence-transformers) |
| transformers | 98 MB | S2 |
| onnxruntime | 69 MB | S2 (via chromadb) |
| tokenizers | 8.3 MB | S2 |
| chromadb | 6.6 MB | S2 |
| huggingface_hub | 6.0 MB | S2 |
| langchain_core | 5.3 MB | S1 |
| sentence_transformers | 4.8 MB | S2 |
| langgraph | 2.6 MB | S1 |
| safetensors | 1.1 MB | S2 |
| langchain | 1.1 MB | S1 |
| langchain_openai | 872 KB | S1 |
| **Combined footprint** | **~612 MB** | — |

The combined removable footprint is ~612 MB — roughly half the venv. S1 removes
the LangGraph/LangChain stack (~9.9 MB direct, no transitive heavyweights); S2
removes chromadb + sentence-transformers, which is where the torch/onnxruntime/
transformers mass (~575 MB) actually lives. Record the post-session `du -sh`
delta in each session.

## Cold start

| Measure | Value |
|---|---|
| `june-api` cold start (process launch to first HTTP response on `/`) | 3.1 s |

Method: launched `june-api` on a free port, polled `GET /` at 100 ms intervals
until a response (404 is a response — the app is up), recorded elapsed.

## Test suite

| Measure | Value |
|---|---|
| Test functions (`packages/brain/tests` + `packages/api/tests`) | 577 |

## CLEAR loop-engine numbers (copied from `loop-clear.md`)

Host: MacBook (Apple Silicon), Ollama 0.20.2, gemma4:e2b via local
OpenAI-compatible API. 5 runs per engine per task.

| Task | Engine | Cost (tok) | Latency (ms) | Efficacy | Assurance | Reliability (cv%) |
|---|---|---|---|---|---|---|
| recall_question | handwritten | 360 | 2 237 | 100% | 100% | 75.6% |
| multi_step_tool | handwritten | 518 | 6 955 | 100% | 100% | 19.2% |
| long_conversation_compaction | handwritten | 825 | 1 166 | 100% | 100% | 54.0% |
| cloud_escalation | handwritten | 499 | 6 309 | 100% | 100% | 16.7% |
| stay_quiet | handwritten | 465 | 5 650 | 100% | 100% | 20.1% |
| recall_question | langgraph | 167 | 21 753 | 100% | 100% | 15.2% |
| multi_step_tool | langgraph | 348 | 36 670 | 100% | 100% | 31.7% |
| long_conversation_compaction | langgraph | 765 | 33 001 | 100% | 100% | 8.2% |
| cloud_escalation | langgraph | 474 | 29 580 | 100% | 100% | 6.6% |
| stay_quiet | langgraph | 269 | 23 409 | 100% | 100% | 1.8% |

Conclusion (closing evidence for ADR 0018): handwritten wins on latency 3-17x
at equal efficacy. Worst handwritten reliability is recall_question at cv 75.6%
— the regression target for S5 (structured tool calls) is recall cv < 25%.

## Realized deltas (updated as sessions land)

- **S1.2b (drop LangGraph/LangChain):** removed `langgraph`, `langchain`,
  `langchain-core`, `langchain-openai` and their transitive tree (langgraph-*,
  tiktoken, langsmith, ...). Brain `site-packages` 1301 MB -> 1282 MB (~19 MB
  freed in the existing venv; a fresh install avoids the whole tree). The bulk
  of the venv mass (torch/onnxruntime/transformers ~575 MB) is pulled by
  chromadb + sentence-transformers and drops in S2. The gate passes with
  langchain physically uninstalled — the source tree is LangChain-free.

- **S2 (ChromaDB -> sqlite-vec):** removed `chromadb` + `sentence-transformers`
  and their heavy tree (`torch` 408 MB, `transformers` 98 MB, `onnxruntime`
  69 MB, `tokenizers`, `safetensors`). Added `sqlite-vec` (~0.5 MB loadable C
  extension). Brain `.venv` **1.3 GB -> 653 MB (~647 MB freed)** — roughly half
  the venv, the single largest drop in the rebuild. Embeddings now come from a
  local Ollama model (default `nomic-embed-text`); vectors live in a `vec0`
  virtual table inside the same `june.db`, so the data dir is one copyable
  SQLite file. The full gate (583 tests) passes with the whole torch/chroma
  tree physically uninstalled — the source tree is ChromaDB-free. The sqlite-vec
  load probe confirmed the extension loads into stdlib `sqlite3` on Python 3.14
  / Apple Silicon, so no platform pivot was needed. **Cold start: `create_app`
  now ~0.53 s vs the 3.1 s baseline (~6x faster)** — the torch/transformers
  import tree no longer loads. Full brain+api suite: 610 tests in ~9 s.

## Reliability harness (S5.3)

`tools/reliability_harness.py` runs representative tasks (recall, multi-step
tool use, compaction) N times against the live local model and reports the
coefficient of variation (cv%) per metric, via the pure, unit-tested
`june_brain.experiments.reliability` module. Run it once with Ollama up and
record the cv numbers here next to the CLEAR baseline (recall cv 75.6%); the
S5 target is recall cv < 25%, now that provider-native tool calling (S5.1/S5.2)
replaces the prose-JSON parse on the run_turn path. Numbers pending a local run.

## Targets derived from this baseline

- S2: install footprint drops by the bulk of ~575 MB (torch/onnxruntime/transformers).
  Realized: ~647 MB freed (see Realized deltas).
- S4: trivial-turn latency measurably below the handwritten baseline above.
- S5: recall cv% from 75.6% toward < 25% (measure with the reliability harness).
- S5: recall_question reliability cv from 75.6% to < 25%.
