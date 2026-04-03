# Next Session — Session 2: Mistral Tool Calling

Pick up here. Read this file first, then start coding.

## Context

Full plan is in `docs/PLAN.md`. Session 1 is complete.

Repo: /Users/admin/JuneAI/  
App dir: /Users/admin/JuneAI/JuneAI-app/  
Venv: /Users/admin/JuneAI/JuneAI-app/.venv  
Run tests: `cd JuneAI-app && .venv/bin/python -m pytest tests/unit_tests/ -q`  
Current tests: 88 passing  

## Session 1 — What Was Done (complete)

- `graph.py`: Startup guard — `june_agent` creation wrapped in try/except, `startup_error` exported
- `app.py`: Shows friendly setup screen if `startup_error` is set
- `scripts/check_ollama.py`: New script to verify Ollama is running and model is available
- `Makefile`: `check-ollama` target added
- `graph.py`: Removed `ast.literal_eval` — replaced with safe JSON-only extraction strategies
- `config.py`: Fixed dead code in API key resolution — preset default now reachable as fallback, raises `ValueError` if key is still missing
- `graph.py`: Memory context cache — 30s TTL per user_id, invalidated after tool writes
- `memory.py`: JSON corruption recovery now logs a `WARNING` before falling back to default
- 10 new tests in `test_reliability.py`, all passing

## Session 2 Goal

Make tool calling 95%+ reliable with Mistral running locally.

Mistral 3B/8B produces tool calls as JSON text in the message content field instead of the
`tool_calls` field that LangChain expects. The current recovery system works but is fragile.
Mistral 7B Instruct v0.3 supports native OpenAI-style function calling — it eliminates the
recovery-from-text hack entirely for the default model.

## Five Tasks — Do in Order

### Task 1 — Upgrade default model preset to mistral:7b-instruct-v0.3

**In `config.py`**:
- Add a new `local_mistral_7b` preset:
  - `model_env_var`: `"LOCAL_LARGE_MODEL_NAME"`
  - `default_model`: `"mistral:7b-instruct-v0.3"`
  - `tool_strategy`: `"native"`
  - `temperature`: `0.3`
  - `max_tokens`: `4096`
- Make `local_mistral_7b` the default preset (used when `MODEL_PRESET` env var is not set)
- Keep `local_mistral_3b` and `local_mistral_8b` presets as-is

**In `.env`**:
- Update default to `MODEL_PRESET=local_mistral_7b`

### Task 2 — Model capability detection

**In `config.py`**:
- Add `detect_tool_strategy(model_name: str) -> str`:
  - Returns `"native"` for: any model name containing `"7b-instruct-v0.3"`, `"mistral-nemo"`,
    `"mistral-small"`, `"mixtral"`, `"claude"`, `"gpt-4"`, `"gpt-3.5-turbo"`
  - Returns `"recovery"` for: `"mistral:3b"`, `"llama"`, and anything else
- `RuntimeConfig.tool_strategy` is already a field — ensure it is populated from either
  the preset definition or the result of `detect_tool_strategy(model_name)`

### Task 3 — Separate native vs recovery code paths in graph.py

Currently `graph.py` always runs the recovery pipeline. Add branching based on `tool_strategy`.

**In `graph.py`**:
- After getting an `AIMessage` from the LLM, check `runtime_config.tool_strategy`:
  - If `"native"`: trust `message.tool_calls` directly, skip `_recover_tool_call` and
    `_normalize_tool_call` entirely (unless `message.tool_calls` is empty, in which case
    treat as a direct text reply)
  - If `"recovery"`: run the existing recovery + normalization pipeline as-is
- This means the recovery code stays — it is still needed for 3B models and fallbacks

### Task 4 — Reduced tool set for 3B models

48 tools is a long context for Mistral 3B. Small models often drop tool calls when the
function list is too long.

**In `tools.py`**:
- Define `JUNE_TOOLS_CORE` — a list of the 20 most important tools:
  - `log_mood`, `save_journal_entry`, `track_goal`, `update_goal_status`,
    `save_open_loop`, `update_open_loop_status`, `save_calendar_item`,
    `list_calendar_items`, `update_calendar_item_status`, `save_user_preference`,
    `log_body_metrics`, `log_workout_session`, `log_habit_completion`,
    `create_or_update_habit`, `log_nutrition`, `log_water`,
    `set_ui_chapter`, `set_ui_focus`, `set_ui_checklist`, `clear_ui_workspace`
- Keep `JUNE_TOOLS` (full 48-tool list) unchanged

**In `graph.py`**:
- When binding tools to the LLM:
  - If preset is `local_mistral_3b`: use `JUNE_TOOLS_CORE`
  - Otherwise: use `JUNE_TOOLS`
- Import `JUNE_TOOLS_CORE` from tools alongside `JUNE_TOOLS`

### Task 5 — Tool success rate badge in UI

The right-rail debug panel already shows tool logs. Surface the success rate more visibly.

**In `app.py`** (or wherever the right-rail debug tab is rendered):
- After each turn, compute: `success = tool_stats.get("success_count", 0)`,
  `total = tool_stats.get("total_count", 0)`
- Show a compact badge above the tool log: `Tools: {success}/{total} saved` 
- If any tools dropped (`success < total`): show in amber; if all saved: show in green
- Use `st.markdown` with inline HTML or Streamlit's native `st.metric`
- `tool_stats` is already tracked in `AgentState` from Session 1 / prior work — read
  it from `st.session_state` after the agent runs

## Files to Touch

| File | What changes |
|------|-------------|
| `src/agent/config.py` | Tasks 1, 2 — new preset, detect_tool_strategy |
| `src/agent/graph.py` | Tasks 3, 4 — branching on tool_strategy, tool set selection |
| `src/agent/tools.py` | Task 4 — JUNE_TOOLS_CORE list |
| `app.py` | Task 5 — tool success rate badge |
| `.env` | Task 1 — update default MODEL_PRESET |

## Success Criteria

- [ ] `local_mistral_7b` preset exists with `tool_strategy="native"`
- [ ] `detect_tool_strategy("mistral:7b-instruct-v0.3")` returns `"native"`
- [ ] `detect_tool_strategy("mistral:3b")` returns `"recovery"`
- [ ] Native path skips recovery when `tool_strategy == "native"` and `tool_calls` is populated
- [ ] `JUNE_TOOLS_CORE` has exactly 20 tools
- [ ] 3B preset uses `JUNE_TOOLS_CORE`, all others use `JUNE_TOOLS`
- [ ] Tool success badge appears in the right rail
- [ ] All 88 existing tests still pass
- [ ] At least 5 new tests: preset resolution, tool strategy detection, native/recovery branching,
     core tool list length, tool binding selection by preset

## Workflow

Use `isolation: "worktree"` — keeps main stable, easy to review diff before merging.
Run tests after each task, not at the end.

## After Session 2

Update this file: change heading to "Next Session — Session 3: SQLite Memory Upgrade"
and fill in the Session 3 briefing from docs/PLAN.md.
