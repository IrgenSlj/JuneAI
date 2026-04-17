# JuneAI v1 (Legacy Streamlit App)

This directory contains the v1 Streamlit prototype of June. It is no longer being developed. It will be deleted at the end of v2 Week 1 as described in [`../docs/product/plan.md`](../docs/product/plan.md).

## Why It Is Still Here

- Historical reference during the v2 migration.
- The brain modules under `src/agent/` are migrating to `packages/brain/` — see [ADR 0001](../docs/decisions/0001-monorepo-structure.md).
- User data in `.june_memory/` migrates to `~/Library/Application Support/June/` during v2 Week 1.

## Running v1

```bash
cp .env.example .env
make bootstrap
make check-ollama
make run
```

Open `http://127.0.0.1:8501`.

## What Transfers to v2

The brain and memory logic transfer. The UI does not. See the migration table in [ADR 0003](../docs/decisions/0003-streamlit-to-sveltekit.md).

| Path | Fate |
|---|---|
| `app.py` | Deleted |
| `src/agent_ui/*` | Deleted |
| `src/agent/graph.py` | Moves to `packages/brain/src/june_brain/agent/` |
| `src/agent/memory.py` | Splits into four files under `packages/brain/src/june_brain/memory/` |
| `src/agent/tools.py` | Splits into MCP servers under `skills/` |
| `src/agent/patterns.py` | Moves to `packages/brain/src/june_brain/patterns/` |
| `src/agent/skills.py` | Moves to `packages/brain/src/june_brain/skills/loader.py` |
| `src/agent/runtime_privacy.py`, `telemetry.py`, `config.py` | Move, with presets trimmed to Gemma 4 + Gemini |
| SQLite schema in `.june_memory/june.db` | Preserved, migrated to platform data dir |
| Tests under `tests/unit_tests/` | ~80% migrate; UI tests deleted |

## After Week 1

This directory will be gone from `main`. A `legacy/streamlit` branch will preserve the final v1 state for historical reference.
