# June

The open personal AI that remembers you.

June runs privately on your laptop via Gemma 4 and scales to the cloud via Gemini. It works identically in your browser, on your Mac, and on your iPhone. Everything is open source. Everything is free.

## Status

June is mid-migration from v1 (Streamlit prototype) to v2 (multi-platform product). The v2 architecture is documented and the 8-week plan is active. See [`docs/product/plan.md`](docs/product/plan.md) for the current week and exit criteria.

The v1 Streamlit app still runs in `JuneAI-app/` during the transition but is no longer being developed.

## Read These First

1. [Vision](docs/vision.md) — what June is and the three non-negotiables
2. [Architecture overview](docs/architecture/overview.md) — the layered model
3. [Architecture decisions](docs/decisions/README.md) — the six ADRs that justify the design
4. [8-week plan](docs/product/plan.md) — what we're building and when
5. [Environment](docs/setup/environment.md) — configuration reference

## Repository Layout

```
JuneAI/
├── apps/              # end-user apps: web, desktop (Tauri), mobile (Capacitor)
├── packages/          # internal libraries: brain, api, ui, design
├── skills/            # MCP skill servers: calendar, health, research, files, daily
├── docs/              # vision, architecture, decisions, product plan, setup
├── tools/             # developer tooling
├── JuneAI-app/        # v1 Streamlit app (legacy, retiring during Week 1)
└── README.md
```

For the rationale behind this layout see [ADR 0001](docs/decisions/0001-monorepo-structure.md).

## Running v1 (Legacy)

While the v2 migration is in progress, the v1 Streamlit app is still runnable:

```bash
cd JuneAI-app
cp .env.example .env
make bootstrap
make check-ollama
make run
```

Open `http://127.0.0.1:8501`.

v1 will be deleted at the end of Week 1. See [`docs/product/plan.md`](docs/product/plan.md).

## Running v2

Not yet available. Week 1 produces the foundation. Week 3 produces the first usable browser build.

## License

Open source under a permissive license. See [`LICENSE`](JuneAI-app/LICENSE).

## Contributing

Contribution guidelines will be published in Week 8. Until then, discussion happens in GitHub issues.
