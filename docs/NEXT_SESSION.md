# Next Session — Session 6: Intelligence Layer + Ship

Pick up here. Read this file first, then start coding.

## Context

Full plan is in `docs/PLAN.md`. Sessions 1-5 are complete.

Repo: /Users/admin/JuneAI/  
App dir: /Users/admin/JuneAI/JuneAI-app/  
Venv: /Users/admin/JuneAI/JuneAI-app/.venv  
Run tests: `cd JuneAI-app && .venv/bin/python -m pytest tests/unit_tests/ -q`  
Current tests: 116 passing  
Run app: `cd JuneAI-app && make run`  

## What Was Completed in Sessions 1-5

Session 1 — Reliability foundation (startup guard, safe JSON parsing, memory cache)  
Session 2 — Mistral tool calling (native/recovery branching, capability detection, core tool set)  
Session 3 — SQLite memory (drop-in replacement, cross-chapter queries, migration script)  
Session 4 — June personality (proactive patterns in `src/agent/patterns.py`, dynamic system prompt, daily chapter focus)  
Session 5 — UI overhaul (three-panel layout, custom header with logo + date + panel toggles, Ollama download management with progress tracking and version error detection, calendar dialog, settings dialog, health panel, chat bubbles, typing indicator)  

## Session 6 Goal

Make June proactively useful and get the product ready to share. Six tasks.

---

## Task 1 — Weekly Summary Tool

Add a new agent tool `generate_weekly_summary` that produces a personal weekly review.

**File**: `src/agent/tools.py` — add to the tool list alongside the existing tools.

**What it does**: reads the last 7 days of memory across chapters and returns a structured summary:
- Workouts completed (count, types, avg energy rating)
- Habits maintained (per-habit completion rate from `get_habit_consistency(days=7)`)
- Goals progressed (any goal with `updated_at` in the last 7 days)
- Body averages (avg sleep, avg energy, avg stress from `get_body_metrics(days=7)`)
- Notable calendar events (from `get_calendar_items` with dates in the last 7 days)
- Notable mood log (from `get_mood_history(limit=7)`)

The tool saves the result as a journal entry and returns a markdown-formatted string.

**Trigger**: user says anything like "give me my week summary", "weekly review", "how was my week".

**Test**: `tests/unit_tests/test_weekly_summary.py` — 3 tests:
1. Tool returns a non-empty string for a memory with 7 days of data
2. Tool saves a journal entry (visible in `get_journal()`)
3. Tool returns a graceful "not enough data" string for an empty memory

---

## Task 2 — Smart Daily Suggestion

One proactive suggestion injected into June's opening message once per day.

**File**: `src/agent/patterns.py` — add a new function:

```python
def get_daily_suggestion(memory: Memory) -> str | None:
    """Return one actionable suggestion or None if nothing obvious to suggest."""
```

Logic (pick the first matching condition):
1. Calendar has a gap tomorrow (no items) + there is an overdue or stalled goal → "Your calendar is clear tomorrow — good slot to move '{goal_title}' forward"
2. Energy logged below 3 for 3 days + no wind-down reminder in calendar → "You have had low energy for 3 days — want to add a wind-down reminder tonight?"
3. Active goal with no `next_step` + open for 14+ days → "'{goal_title}' has been open for N days with no next step — want to break it down?"
4. Habit completion rate under 50% for any habit this week → "'{habit_name}' is only N% this week — what is getting in the way?"

**Injection**: in `src/agent/skills.py` inside `build_system_prompt`, after `patterns_context`, add:
```python
suggestion_context = ""
if memory is not None:
    from .patterns import get_daily_suggestion
    suggestion = get_daily_suggestion(memory)
    if suggestion:
        suggestion_context = f"\nJUNE'S SUGGESTION FOR TODAY: {suggestion}\n(Offer this naturally once if the conversation allows — do not force it.)\n"
```

**Test**: 2 tests in `tests/unit_tests/test_patterns.py`:
1. `test_daily_suggestion_stalled_goal` — goal with no next_step, 15+ days old → suggestion returned
2. `test_daily_suggestion_none_for_empty_memory` — fresh memory → returns None

---

## Task 3 — Memory Export

Add `make export-memory` that writes a readable markdown file of everything June knows.

**File**: `scripts/export_memory.py`

```python
#!/usr/bin/env python3
"""Export JuneAI memory to a readable markdown file.

Usage: python scripts/export_memory.py [--user-id USER] [--output report.md]
"""
```

The output should be organised by chapter with item counts and key data. Example structure:
```markdown
# June Memory Export — USER_ID — DATE

## Goals (3 active)
- Run a 5k — next step: Buy shoes — deadline: 2026-06-01
...

## Calendar (8 items)
...
```

Add to Makefile:
```makefile
export-memory:
    $(VENV_PYTHON) scripts/export_memory.py
```

---

## Task 4 — Docker Setup

**File 1**: `JuneAI-app/Dockerfile`
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[all]"
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501"]
```

**File 2**: `docker-compose.yml` (in repo root `/Users/admin/JuneAI/`)
```yaml
version: "3.9"
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  juneai:
    build: ./JuneAI-app
    ports:
      - "8501:8501"
    environment:
      - LLM_BASE_URL=http://ollama:11434/v1
      - LLM_API_KEY=ollama
      - MODEL_NAME=gemma4:e4b
    depends_on:
      - ollama
    volumes:
      - june_memory:/app/.june_memory

volumes:
  ollama_data:
  june_memory:
```

Add to Makefile:
```makefile
docker-run:
    docker compose up --build
```

Smoke test: `docker compose up -d && sleep 10 && curl -f http://localhost:8501` must return 200.

---

## Task 5 — Gemma 4 as Primary Model

Once Ollama is updated (`brew upgrade ollama`), Gemma 4 becomes the default model.

Changes needed:
1. In `src/agent/config.py`, add Gemma 4 to capability detection:
   ```python
   # In detect_tool_strategy():
   if any(m in name for m in ("gemma4", "gemma3", "gemma 4", "gemma 3")):
       return "gemma"
   ```
2. Verify `gemma4:e4b` tool calling works with the existing `gemma` prompt style
3. The default preset in `.env` is already set to `local_gemma4` — no change needed
4. Test: add one test to `test_configuration.py` verifying `detect_tool_strategy("gemma4:e4b") == "gemma"`

---

## Task 6 — README Rewrite + Tests to 100+

**README.md** (in `/Users/admin/JuneAI/`):

Write a clean, honest README covering:
- What JuneAI is (one paragraph — "June is a local AI personal assistant...")
- Stack: Python, LangGraph, Gemma 4 via Ollama, Streamlit
- One-command setup: `make bootstrap && make run`
- Prerequisites: Python 3.9+, Ollama, brew
- Screenshot placeholder: `[screenshot]`
- Vision: one paragraph on the "friend not a tool" principle
- No emojis

**Tests**: get to 100+ total. At the end of this session, `pytest tests/unit_tests/ -q` should show 100+ passing. Needed:
- 3 tests for weekly summary tool (Task 1)
- 2 tests for daily suggestion (Task 2)
- 1 test for memory export script (Task 3 — test that the script runs and produces output)
- 1 test for Docker smoke (optional, skip if Docker not available in CI)
- 1 test for Gemma 4 capability detection (Task 5)

That puts us at 116 + 8 = 124 tests minimum.

---

## Known Issues Going Into Session 6

1. **Gemma 4 not downloadable** — Ollama version too old. Run `brew upgrade ollama` first thing. The app now shows an "Upgrade Ollama" button that does this for you.

2. **Logo PNG path** — The PNG logo requires Streamlit static file serving, enabled in `.streamlit/config.toml`. If the logo does not appear, restart the app once after a fresh pull.

3. **Header on mobile** — The panel toggle buttons (H, M) may overlap on narrow screens. The media query at 768px hides both panels. Verify on Safari mobile.

---

## Success Criteria for Session 6

- [ ] `make bootstrap && make run` works on a clean machine with Ollama installed
- [ ] `docker compose up` brings the full stack up in under 3 minutes
- [ ] Weekly summary generates correctly for a user with 7 days of data
- [ ] Daily suggestion appears naturally in June's first message of the day (when conditions match)
- [ ] `make export-memory` generates a readable markdown file
- [ ] 100+ tests, all green
- [ ] Gemma 4 downloads and tool calling works after Ollama upgrade
- [ ] README is accurate and self-contained

## Commit Message to Use at End

```
Session 6: intelligence layer, Docker, README, 100+ tests
```
