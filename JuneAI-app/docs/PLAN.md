# JuneAI Development Plan

## Goal

A local-first personal AI that runs in two modes:

- Local mode (Gemma 4 / Mistral) — private, offline-friendly, on-device inference
- Claude mode — maximum reasoning quality via Anthropic API

Both modes share the same memory, tool layer, and UX. The product promise: one place to talk, remember, plan, and stay organized.

---

## Architecture

```
Streamlit (app.py)
  └─ LangGraph agent (src/agent/graph.py)
      ├─ chat node — LLM call with memory context + tools bound
      └─ run_tools node — executes tool calls, updates tool_stats
          └─ Tools (src/agent/tools.py) → Memory (src/agent/memory.py, SQLite)
```

UI is split-column: chat (left, hideable) + active panel (right, tabbed).

---

## Runtime Presets

| Key | Model | Notes |
|-----|-------|-------|
| `local_gemma_4` | gemma4:e4b | temperature 0.4, native tool calling |
| `local_mistral_7b` | mistral:7b-instruct-v0.3 | temperature 0.3, native tool calling |
| `local_mistral_3b` | mistral | temperature 0.2, recovery strategy |
| `local_mistral_8b` | mistral-nemo | temperature 0.2, recovery strategy |
| `claude_high` | claude-sonnet-4-6 | temperature 0.35, API |

---

## Completed Build Phases

### Session 1 — Runtime Abstraction
- Runtime presets in `src/agent/config.py`
- Provider-specific model construction in `src/agent/models.py`
- Tool-call diagnostics (requested / succeeded / failed) in `graph.py`
- Deterministic offline integration tests

### Session 2 — UI Foundation
- Split-column layout: chat left, panel right
- Tabbed nav: Today / Agenda / Plans / Gym & Food / Health / Calendar
- Dark mode toggle, model badge, settings dialog
- Right-rail panel builders in `src/agent_ui/panels.py`

### Session 3 — Memory Surfaces
- 14 memory chapters (calendar, goals, habits, body, workouts, gym plan, nutrition, food program, mood, journal, relationships, preferences, favourites, open loops)
- Chapter completeness tracking, setup progress card
- Context intelligence: recovery readiness, active commitments summary
- Pattern detection: workout gaps, low energy streaks, stale goals, broken habits, upcoming birthdays

### Session 4 — Chat & Input Polish
- Sticky chat column; transcript scrolls independently
- Enter = send, Cmd+Enter = newline
- Hint ticker (rotating tips, 15s interval)
- Quick-log buttons in Today panel (Sleep / Workout / Mood / Weight / Water +1)
- Journal & Mood surface in Agenda panel
- Message window capped at 24 turns with compressed summary prefix

### Session 5 — Stability & Reliability
- gemma4 temperature lowered 1.0 to 0.4 (tool-call reliability)
- Empty-response notice surfaces instead of silent failure
- Generation exception handling with model-not-found cache invalidation
- Removed stale `src/agent/prompts.py` compatibility shim

### Session 6 — Input UI Redesign
- Logo: PNG replaced with inline Syne bold text matching nav tab weight
- Input area redesigned as unified card: [+] [textarea] [send icon]
- Send button changed to compact icon, textarea borderless inside card
- Pattern insights fixed: category tag no longer rendered as body text

---

## Next Build Phases

### High priority

1. **LLM reliability — model name verification**
   - `LOCAL_GEMMA_MODEL_NAME` must match the actual tag in `ollama list`
   - Add a UI hint when the model name looks wrong (not just "404 not found")

2. **Streaming response rendering**
   - Wire `live_response` streaming into the transcript placeholder progressively
   - Currently no partial output is shown during generation

3. **Tool-call evaluation screen**
   - Per-turn: tool success rate, which tools fired, what they saved
   - Accessible from settings panel or a dedicated tab

### Medium priority

4. **Model-specific prompt policies**
   - Split system prompt: shared behavior + local-small policy + Claude policy
   - Smaller prompts mean faster, more reliable tool decisions for 4B models

5. **Tighten tool schemas for local models**
   - Fewer optional fields, shorter names, no multi-tool chains per turn for 3B models

6. **Onboarding flow improvement**
   - Guide new users to fill the empty chapters via conversation starters
   - Surface contextual "teach June about X" prompts

### Lower priority

7. Model-specific tool-evaluation transcripts (fixed prompts, assert exact tool outcomes)
8. Export improvements (CSV, filtered by chapter)
9. Docker volume persistence verification test

---

## Verification Commands

```bash
cd JuneAI-app
.venv/bin/python -m pytest tests/unit_tests -q
.venv/bin/python -m pytest tests/integration_tests -q
PYTHONPATH=src .venv/bin/streamlit run app.py --server.headless true --server.port 8501
make check-ollama
make smoke
```

---

## Environment Setup

### Local Gemma 4

```env
MODEL_PRESET=local_gemma_4
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LOCAL_GEMMA_MODEL_NAME=gemma4:e4b
```

Verify with `ollama list` that the tag matches `LOCAL_GEMMA_MODEL_NAME`.

### Claude

```env
MODEL_PRESET=claude_high
ANTHROPIC_API_KEY=sk-ant-...
```
