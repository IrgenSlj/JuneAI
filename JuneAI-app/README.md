# JuneAI App

This directory contains the runnable JuneAI application: Streamlit shell, LangGraph agent, runtime configuration, memory layer, tests, and developer scripts.

## Start

```bash
cp .env.example .env
make bootstrap
make check-ollama
make run
```

Open `http://127.0.0.1:8501`.

## What Lives Here

- [app.py](/Users/admin/JuneAI/JuneAI-app/app.py): Streamlit entrypoint
- [src/agent](/Users/admin/JuneAI/JuneAI-app/src/agent): runtime config, graph, tools, memory, models
- [src/agent_ui](/Users/admin/JuneAI/JuneAI-app/src/agent_ui): shell state, rendering, panels, onboarding, transcript, extracted runtime helpers
- [tests](/Users/admin/JuneAI/JuneAI-app/tests): unit and integration suites
- [scripts](/Users/admin/JuneAI/JuneAI-app/scripts): bootstrap, smoke, export, environment checks

## Commands

```bash
make bootstrap
make verify-env
make check-ollama
make run
make smoke
make lint
make test
make integration_tests
make export-memory USER_ID=admin
```

## Runtime Setup

The canonical runtime/environment documentation lives at:

- [../docs/setup/environment.md](/Users/admin/JuneAI/docs/setup/environment.md)

Use `.env.example` as the starting point. Avoid setting shared override variables unless you need them.

## App Docs

- product/repo docs index: [../docs/README.md](/Users/admin/JuneAI/docs/README.md)
- architecture summary: [../docs/architecture/README.md](/Users/admin/JuneAI/docs/architecture/README.md)
- interactive architecture diagrams: [../docs/architecture.html](/Users/admin/JuneAI/docs/architecture.html)
- roadmap: [../docs/product/roadmap.md](/Users/admin/JuneAI/docs/product/roadmap.md)
- next sessions: [../docs/product/next-sessions.md](/Users/admin/JuneAI/docs/product/next-sessions.md)
