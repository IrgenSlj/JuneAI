# Next Session — Session 1: Reliability Foundation

Pick up here. Read this file first, then start coding.

## Context

Full plan is in `docs/PLAN.md`. This file is the Session 1 briefing only.

Repo: /Users/admin/JuneAI/  
App dir: /Users/admin/JuneAI/JuneAI-app/  
Venv: /Users/admin/JuneAI/JuneAI-app/.venv  
Run tests: `cd JuneAI-app && .venv/bin/python -m pytest tests/unit_tests/ -v`  
Run app: `make run` (needs Ollama running)  
Current tests: 78 passing  

## Session 1 Goal

Eliminate every crash, silent failure, and risky code path.
No new features. Just make the foundation solid.

## Five Tasks — Do in Order

### Task 1 — Startup health check
**Problem**: `june_agent = create_june_agent()` runs at module import time in `graph.py:682`.
If Ollama is down, the app crashes to a blank Streamlit error screen with no guidance.

**Fix**:
- Wrap `create_june_agent()` in try/except in `graph.py`
- Store a `startup_error: str | None` alongside `june_agent`
- In `app.py`, check `startup_error` before the main UI and show a setup screen:
  "June could not start. Is Ollama running? Try: `ollama serve` then `ollama pull mistral`"
- Add `scripts/check_ollama.py`: pings `http://localhost:11434/api/tags`, prints model list
- Add `make check-ollama` target to Makefile

### Task 2 — Remove ast.literal_eval
**Problem**: `graph.py` around line 180 uses `ast.literal_eval()` as a JSON parsing fallback
on raw LLM output. This is a code execution risk on untrusted input.

**Fix**:
- Replace with a strict JSON-only parser
- If all JSON strategies fail: log the raw text (first 120 chars) and return `None` (no-op)
- No `ast` import anywhere in the codebase after this

### Task 3 — Fix config dead code
**Problem**: `config.py:144-147` — the API key fallback to `preset.default_api_key` is
unreachable. The env var path always fires first, so missing env vars silently produce `None`.

**Fix**:
- Correct the resolution order: env var first, then preset default, then raise `ValueError`
  with a clear message naming the missing variable

### Task 4 — Memory context caching
**Problem**: `_build_memory_context()` in `graph.py` reads multiple memory files from disk
on every single turn. Rebuilds today_summary, recovery_readiness, and commitments each time.

**Fix**:
- Cache the result per `user_id` with a 30-second TTL (use a simple dict + timestamp)
- Invalidate the cache entry whenever a tool writes to memory (pass a flag back via AgentState
  or use a module-level dict keyed by user_id)
- Expected: ~150-200ms saved per turn on populated memory

### Task 5 — Log JSON corruption recovery
**Problem**: `memory.py` `_recover_json()` silently returns defaults when a file is corrupted.
No log entry, no way to know data was lost.

**Fix**:
- Add a `warnings.warn()` or `logging.warning()` call with: file path + first 80 chars of raw string
- Keep the fallback behavior (return default) — just make it audible

## Files to Touch

| File | What changes |
|------|-------------|
| `JuneAI-app/src/agent/graph.py` | Tasks 1, 2, 4 |
| `JuneAI-app/src/agent/config.py` | Task 3 |
| `JuneAI-app/src/agent/memory.py` | Task 5 |
| `JuneAI-app/app.py` | Task 1 (startup error UI) |
| `JuneAI-app/scripts/check_ollama.py` | Task 1 (new file) |
| `JuneAI-app/Makefile` | Task 1 (new target) |

## Success Criteria

- [ ] App starts and shows a friendly setup screen if Ollama is not running
- [ ] No `ast.literal_eval` or `import ast` anywhere in the codebase
- [ ] Missing API key raises a clear `ValueError` naming the env var
- [ ] `_build_memory_context()` result is cached — confirm by logging cache hit/miss
- [ ] JSON corruption recovery emits a warning log
- [ ] All 78 existing tests still pass
- [ ] At least 3 new tests: startup error handling, JSON parser safety, cache invalidation

## Workflow

Use `isolation: "worktree"` — keeps main stable, easy to review diff before merging.
Run tests after each task, not at the end.

## After Session 1

Update this file: change heading to "Next Session — Session 2: Mistral Tool Calling"
and fill in the Session 2 briefing from docs/PLAN.md.
