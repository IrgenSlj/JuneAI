# Architecture

JuneAI is a local-first Streamlit application wrapped around a LangGraph agent and a SQLite-backed memory layer.

## System Shape

1. The user interacts with the Streamlit shell in [app.py](/Users/admin/JuneAI/JuneAI-app/app.py).
2. `app.py` resolves the active runtime preset and streams turns through the compiled LangGraph agent in [graph.py](/Users/admin/JuneAI/JuneAI-app/src/agent/graph.py).
3. The agent chooses whether to answer directly or call tools from [tools.py](/Users/admin/JuneAI/JuneAI-app/src/agent/tools.py).
4. Tools read and write structured memory through [memory.py](/Users/admin/JuneAI/JuneAI-app/src/agent/memory.py).
5. The UI refreshes today surfaces, memory chapters, workspace content, and activity telemetry in real time.

## Runtime Model

Supported runtime families:

- Local OpenAI-compatible endpoints via Ollama or similar
- Anthropic Claude via API

Runtime resolution lives in [config.py](/Users/admin/JuneAI/JuneAI-app/src/agent/config.py). Model construction lives in [models.py](/Users/admin/JuneAI/JuneAI-app/src/agent/models.py). Ollama process and download helpers live in [ollama_manager.py](/Users/admin/JuneAI/JuneAI-app/src/agent/ollama_manager.py).

## Persistence Model

JuneAI now stores user memory in one SQLite database under `MEMORY_DIR/june.db`.

Key domains include:

- chat messages
- moods and journal
- goals and open loops
- calendar items
- relationships and preferences
- workouts, body metrics, nutrition, water, habits
- telemetry and app state

The architecture page at [architecture.html](/Users/admin/JuneAI/docs/architecture.html) reflects the current SQLite model and should be treated as the visual companion to this file.

## UI Module Map

The UI is gradually being decomposed out of `app.py`.

Current extracted modules include:

- shell state: [state.py](/Users/admin/JuneAI/JuneAI-app/src/agent_ui/state.py)
- shell runtime helpers: [shell_runtime.py](/Users/admin/JuneAI/JuneAI-app/src/agent_ui/shell_runtime.py)
- transcript rendering: [transcript.py](/Users/admin/JuneAI/JuneAI-app/src/agent_ui/transcript.py)
- panel models: [panels.py](/Users/admin/JuneAI/JuneAI-app/src/agent_ui/panels.py)
- onboarding: [onboarding.py](/Users/admin/JuneAI/JuneAI-app/src/agent_ui/onboarding.py)

## Current Architectural Priorities

- continue extracting dialog and startup/runtime flows from `app.py`
- add clearer runtime validation and preset diagnostics
- expose telemetry and tool outcomes more explicitly in the product UI
- keep documentation aligned with actual runtime behavior and environment expectations
