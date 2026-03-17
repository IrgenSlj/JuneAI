# JuneAI Makeover Plan

## Goal

Move JuneAI toward a real personal assistant architecture that can run in two modes:

- Local mode with a small Mistral-class model for privacy and offline-friendly use
- Claude mode for best reasoning quality and stronger tool reliability

The assistant should keep the same product promise in both modes: memory, planning, proactive capture, and visible tool execution.

## Baseline Review

Before this pass, the codebase had three issues that blocked the migration:

1. The graph was hard-wired to `ChatOpenAI` and one OpenAI-compatible endpoint path.
2. Tool reliability was not measured directly, so local-model failures could be mistaken for success.
3. Integration coverage depended on live network behavior instead of deterministic offline tests.

## What Changed

### Runtime abstraction

- Added runtime presets in [`src/agent/config.py`](/Users/admin/JuneAI/JuneAI-app/src/agent/config.py)
- Added provider-specific model construction in [`src/agent/models.py`](/Users/admin/JuneAI/JuneAI-app/src/agent/models.py)
- The agent now supports:
  - `local_mistral_3b`
  - `local_mistral_8b`
  - `claude_high`

### Tool-call diagnostics

- The graph now records tool-call request/success/failure counts in [`src/agent/graph.py`](/Users/admin/JuneAI/JuneAI-app/src/agent/graph.py)
- The UI surfaces runtime and tool results in [`app.py`](/Users/admin/JuneAI/JuneAI-app/app.py)
- This is the main verification loop for local-model readiness

### Testability

- Added deterministic provider/runtime tests in [`tests/unit_tests/test_models.py`](/Users/admin/JuneAI/JuneAI-app/tests/unit_tests/test_models.py)
- Reworked the graph integration test to run without live model or LangSmith dependency in [`tests/integration_tests/test_graph.py`](/Users/admin/JuneAI/JuneAI-app/tests/integration_tests/test_graph.py)
- Fixed test discovery by adding `src` to the test path in [`tests/conftest.py`](/Users/admin/JuneAI/JuneAI-app/tests/conftest.py)

## Recommended Next Build Phases

1. Add model-specific tool-evaluation transcripts.
   Use fixed user prompts and assert exact tool outcomes for local 3B, local 8B, and Claude.

2. Split the current prompt into:
   - shared assistant behavior
   - local-small tool policy
   - Claude quality policy

3. Build a first-class evaluation screen in the UI.
   Show:
   - tool success rate
   - capture rate by chapter
   - missed-tool examples
   - per-runtime comparison

4. Tighten tool schemas for smaller local models.
   Prefer short field names, fewer optional arguments, and fewer multi-tool chains.

5. Replace broad “capture everything” prompting with targeted extraction passes when needed.
   Small local models perform better when the tool decision is narrow and explicit.

## Verification Commands

```bash
.venv/bin/python -m pytest tests/unit_tests -q
.venv/bin/python -m pytest tests/integration_tests/test_graph.py -q
PYTHONPATH=src .venv/bin/streamlit run app.py --server.headless true --server.port 8501
```

## Runtime Setup Examples

### Local Mistral

```env
MODEL_PRESET=local_mistral_8b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LOCAL_LARGE_MODEL_NAME=mistral-nemo
```

### Claude

```env
MODEL_PRESET=claude_high
ANTHROPIC_API_KEY=...
CLAUDE_MODEL_NAME=claude-3-5-sonnet-latest
```
