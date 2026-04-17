# JuneAI

JuneAI is a local-first personal AI system built around one product: June, a memory-aware assistant that helps you keep continuity across plans, health, habits, relationships, and daily execution.

This repository is split into a top-level product/docs layer and the runnable application in `JuneAI-app/`.

## Repo Map

```text
JuneAI/
├── README.md
├── docs/
│   ├── README.md
│   ├── architecture/
│   │   └── README.md
│   ├── architecture.html
│   ├── product/
│   │   ├── roadmap.md
│   │   └── next-sessions.md
│   └── setup/
│       └── environment.md
├── JuneAI-app/
│   ├── README.md
│   ├── app.py
│   ├── src/
│   ├── tests/
│   ├── scripts/
│   └── .env.example
└── docs/PLAN.md, docs/NEXT_SESSION.md
```

## Start Here

- Product and repo overview: [docs/README.md](/Users/admin/JuneAI/docs/README.md)
- Application setup and developer commands: [JuneAI-app/README.md](/Users/admin/JuneAI/JuneAI-app/README.md)
- Architecture overview: [docs/architecture/README.md](/Users/admin/JuneAI/docs/architecture/README.md)
- Interactive architecture diagrams: [docs/architecture.html](/Users/admin/JuneAI/docs/architecture.html)
- Runtime and environment variables: [docs/setup/environment.md](/Users/admin/JuneAI/docs/setup/environment.md)
- Roadmap and next sessions: [docs/product/roadmap.md](/Users/admin/JuneAI/docs/product/roadmap.md), [docs/product/next-sessions.md](/Users/admin/JuneAI/docs/product/next-sessions.md)

## Quick Start

```bash
cd JuneAI/JuneAI-app
cp .env.example .env
make bootstrap
make check-ollama
make run
```

Open `http://127.0.0.1:8501`.

## Current State

- Runtime profiles support local Ollama models and Anthropic Claude
- Gemma 4 is the default local preset
- Memory is stored in SQLite under `MEMORY_DIR`
- The Streamlit shell is active and production-usable
- Source-tree lint and type checks are green through `make lint`
- Unit and integration tests are green

## Notes

- `JuneAI-app/.env` is a local machine file. Review it against [docs/setup/environment.md](/Users/admin/JuneAI/docs/setup/environment.md) because it may still contain older runtime selections.
- The old docs entry points `docs/PLAN.md` and `docs/NEXT_SESSION.md` remain as compatibility files and now point to the new docs structure.
