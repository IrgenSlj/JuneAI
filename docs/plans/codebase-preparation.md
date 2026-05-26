# Codebase Preparation Plan

> **Status:** Historical/code-health backlog. Python 3.13 is now the runtime
> baseline. Active feature preparation is tracked in
> [v0.1.1 Scheduled Development](v0.1.1-scheduled-development.md).

Before any feature work begins, fix the architectural debt that blocks safe, rapid development.

## Session P1.1 — Python 3.13 Upgrade

**Files to modify:**
- `pyproject.toml` — bump `requires-python`, `python_version`, remove ruff ignores
- `.github/workflows/checks.yml` — simplify CI matrix
- All `.py` files — automated annotation migration

**Commands:**
```bash
# Update pyproject.toml
sed -i '' 's/>=3.10/>=3.13/' pyproject.toml
# Run ruff migration
ruff check --fix --unsafe-fixes packages/brain/src packages/api/src JuneAI-app/src
# Manual review of remaining issues
mypy packages/brain/src packages/api/src
```

**Checklist:**
- [ ] `requires-python = ">=3.13"` in `pyproject.toml`
- [ ] `tool.mypy.python_version = "3.13"`
- [ ] Ruff ignores `UP006`, `UP007`, `UP035`, `UP045` removed
- [ ] CI matrix reduced to `3.13` only
- [ ] `Optional[X]` → `X | None` across all files
- [ ] `Union[X, Y]` → `X | Y`
- [ ] `Dict[str, Any]` → `dict[str, Any]`
- [ ] `List[str]` → `list[str]`
- [ ] `Tuple[X, Y]` → `tuple[X, Y]`
- [ ] `Type[X]` → `type[X]`
- [ ] `Set[X]` → `set[X]`
- [ ] `from typing import Optional` removed where no longer needed
- [ ] Docs updated to reference Python 3.13

## Session P1.2 — Fix ActivityLog Singleton Race

**Problem:** `ActivityLog.__new__` locks but `__init__` doesn't. Two threads can race through `__new__`, both see `_initialised=False`, both run `executescript`.

**Solution:** Move initialization entirely into `__new__` block or use a module-level `_init_lock` + state check.

**Files:** `packages/brain/src/june_brain/activity.py`

**Changes:**
```python
# Before: __new__ with lock + __init__ without
# After: __new__ handles all init, __init__ is a no-op
```

**Checklist:**
- [ ] `__init__` becomes no-op if already initialized
- [ ] Thread-safety verified with concurrent access test
- [ ] `reset_for_tests()` still works

## Session P1.3 — Replace Broad Exception Handlers

**Problem:** ~50 `except Exception:  # noqa: BLE001` across the codebase. Hides real bugs.

**Solution:** Each catch gets a specific exception type or at minimum a `logger.exception()` call.

**Priority files (most critical first):**
1. `packages/brain/src/june_brain/graph.py` — agent runtime, must not silently swallow
2. `packages/brain/src/june_brain/memory/manager.py` — memory writes, must surface errors
3. `packages/brain/src/june_brain/patterns.py` — pattern detection, safe to be defensive
4. `packages/api/src/june_api/app.py` — activity middleware, best-effort is okay
5. `packages/brain/src/june_brain/skills/` — skill lifecycle, needs specific error types

**Checklist:**
- [ ] `graph.py:664` — recall block: narrow to specific exception types
- [ ] `graph.py:699` — skill tools load: narrow
- [ ] `graph.py:932` — agent build: log exception
- [ ] `graph.py:947` — agent build: log exception
- [ ] `graph.py:979` — agent reload: log exception
- [ ] `manager.py:84` — vector search: log exception trace
- [ ] `manager.py:109` — graph lookup: log exception trace
- [ ] `manager.py:117` — sqlite keyword: log exception trace
- [ ] `manager.py:137` — feedback lookup: log exception trace
- [ ] `manager.py:303` — write handler: log exception trace
- [ ] `manager.py:564` — extract LLM call: log exception trace
- [ ] `app.py:108` — activity log: keep as is (best-effort)
- [ ] `patterns.py:249,250,260,266,277` — keep broad but ensure `logger.exception` calls
- [ ] `supervisor.py` — skill lifecycle: keep broad, all paths log
- [ ] All other `# noqa: BLE001` locations audited

## Session P1.4 — Add Schema Migration System

**Problem:** `CREATE TABLE IF NOT EXISTS` means schema changes break existing databases.

**Solution:** Add Alembic migration support with a version table.

**Files:**
- `packages/brain/pyproject.toml` — add `alembic` dependency
- `packages/brain/src/june_brain/memory/alembic/` — migration directory
- `packages/brain/src/june_brain/memory/alembic.ini` — alembic config
- `packages/brain/src/june_brain/memory/sqlite.py` — add migration check on init

**Implementation:**
```python
# On database init (per-connection establishment):
# 1. Create alembic_version table if not exists
# 2. Check current version
# 3. Run any pending migrations
# 4. On migration failure: log error, raise, do NOT auto-continue
```

**Checklist:**
- [ ] Alembic installed and configured
- [ ] Initial migration captures all ~30 existing tables
- [ ] Migration runs automatically on `Memory.__init__`
- [ ] Migration failure is loud (doesn't silently continue with stale schema)
- [ ] `reset_for_tests()` also resets migration state

## Session P1.5 — Split sqlite.py into Per-Domain DAOs

**Problem:** `memory/sqlite.py` is 1768 lines with ~30 tables and ~50 query methods all in one file.

**Solution:** Split into domain-specific modules under `memory/`.

**New structure:**
```
memory/
├── __init__.py
├── sqlite.py          # Connection pool, base class, schema management
├── dao_goals.py       # goals, open_loops tables + queries
├── dao_habits.py      # habits, habit_completions
├── dao_body.py        # body_metrics, water, nutrition
├── dao_calendar.py    # calendar_items
├── dao_relationships.py # relationships
├── dao_journal.py     # journal, moods
├── dao_preferences.py # preferences, favorites
├── dao_tasks.py       # tasks, task_steps (move from tasks/store.py?)
├── dao_feedback.py    # memory_feedback
├── dao_chat.py        # chat_messages
└── dao_chores.py      # chores (future)
```

**Migration strategy:**
1. Create new DAO classes in separate files
2. Each DAO takes `conn` (or `db_path`) in constructor
3. `Memory` class becomes a facade that delegates to DAOs
4. Old methods deprecated, not removed (for backward compat during transition)
5. Once all callers use DAOs, remove deprecated methods

**Checklist:**
- [ ] Connection pool stays in `sqlite.py`
- [ ] Schema DDL stays in `sqlite.py` (single source of truth)
- [ ] Each DAO class has clean constructor `def __init__(self, user_id: str)`
- [ ] `Memory` class delegates to DAOs with same method signatures
- [ ] All existing tests pass without modification
- [ ] No circular imports between DAOs

## Session P1.6 — Replace Tool Alias Chain with Data-Driven Table

**Problem:** `graph.py:277-476` has a 200-line if/elif chain for parameter aliasing. Grows with every tool.

**Solution:** Extract aliases and parameter normalizers into a data-driven table.

**New file:** `packages/brain/src/june_brain/tool_aliases.py`

```python
# Data structure:
ToolAlias = {
    "aliases": list[str],           # old names → canonical name
    "param_map": dict[str, str],   # old param → canonical param
    "normalizer": Callable | None, # optional parameter reshaprer
}

TOOL_ALIASES: dict[str, ToolAlias] = {
    "track_goal": {
        "aliases": ["save_goal", "create_goal", "add_goal"],
        "param_map": {
            "goal": "title", "name": "title", "area": "category",
            "deadline": "target_date", "next": "next_step",
        },
    },
    # ... one entry per tool
}
```

**Checklist:**
- [ ] All aliases from `_normalize_tool_call` moved to table
- [ ] All parameter name mappings moved to `param_map`
- [ ] Complex tools (save_calendar_item, save_journal_entry) keep normalizer functions
- [ ] `_normalize_tool_call` becomes a simple table lookup + dispatch
- [ ] Tests pass: same inputs produce same outputs

## Session P1.7 — Add API Key Authentication

**Implementation plan from ADR 0012.**

**Files:**
- `packages/brain/src/june_brain/auth.py` — key generation + validation
- `packages/api/src/june_api/middleware/auth.py` — FastAPI middleware
- `packages/api/src/june_api/app.py` — register middleware
- `packages/api/src/june_api/routes/setup.py` — exempt setup routes
- `packages/api/tests/` — auth tests

**Checklist:**
- [ ] Key generated on first-run, stored in config store
- [ ] Middleware validates `X-June-Api-Key` header on all routes except:
  - `/healthz`
  - `/setup/status`
  - `/setup/apply`
  - `/openapi.json`, `/docs`, `/redoc`
- [ ] 401 response with `{"detail": "Invalid or missing API key"}`
- [ ] CORS still restricted to localhost
- [ ] Frontend updated to read and send the key
- [ ] Tests cover: valid key, missing key, wrong key, key rotation

## Session P1.8 — Add Data Portability

**Problem:** No way to export/import the full knowledge graph. The Obsidian export tool only covers journal entries.

**Solution:** Add bulk export/import endpoints.

**Files:**
- `packages/api/src/june_api/routes/export.py` — export route
- `packages/api/src/june_api/routes/import.py` — import route
- `packages/brain/src/june_brain/memory/export.py` — export logic
- `packages/brain/src/june_brain/memory/import_.py` — import logic

**Export format (JSON):**
```json
{
  "version": 1,
  "exported_at": "2026-06-01T12:00:00Z",
  "user_id": "...",
  "stores": {
    "sqlite": { /* all rows from all tables */ },
    "vector": { /* all entries with text+metadata */ },
    "graph": { /* all nodes and edges */ }
  }
}
```

**Checklist:**
- [ ] `GET /memory/{user_id}/export` — downloads full memory as JSON
- [ ] `POST /memory/{user_id}/import` — imports JSON, merges or replaces
- [ ] CLI tool `python -m june_brain.memory.export --user <id> --output path`
- [ ] CLI tool `python -m june_brain.memory.import_ --user <id> --input path`
- [ ] Import validates schema version before writing
- [ ] Import is transactional (all-or-nothing per store)

## Session P1.9 — Reactive Agent Rebuild on Skill Toggle

**Problem:** After toggling a skill, `reload_agent()` must be called manually. The API already exposes `/skills/{key}/toggle`, but doesn't trigger the rebuild.

**Solution:** Make `reload_agent()` automatic after any skill mutation.

**Files:**
- `packages/api/src/june_api/routes/skills.py` — add reload after toggle/install/uninstall
- `packages/brain/src/june_brain/graph.py` — ensure `reload_agent()` is thread-safe

**Checklist:**
- [ ] `POST /skills/{key}/toggle` calls `reload_agent()` after mutation
- [ ] `POST /skills/registry/install` calls `reload_agent()`
- [ ] `DELETE /skills/registry/{key}` calls `reload_agent()`
- [ ] Concurrent requests don't race during reload
- [ ] If reload fails, old agent stays in place (not replaced with `None`)
