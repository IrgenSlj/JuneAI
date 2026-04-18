# June

The open personal AI that remembers you.

June runs privately on your laptop via Gemma 4 and scales to the cloud via Gemini. It works identically in your browser, on your Mac, and on your iPhone. Everything is open source. Everything is free.

## Status

June is in early development. Week 1 is complete: the monorepo is in place, the brain is in `packages/brain/`, and the two supported model runtimes (Gemma 4 and Gemini) are wired through one code path. See [`docs/product/plan.md`](docs/product/plan.md) for the current week and exit criteria.

The original v1 Streamlit prototype is preserved on the `legacy/streamlit` branch for historical reference.

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
├── tools/             # developer tooling (dev.sh, migrate_v1_data.py)
└── README.md
```

For the rationale behind this layout see [ADR 0001](docs/decisions/0001-monorepo-structure.md).

## Quickstart

```bash
cp .env.example .env
./tools/dev.sh
```

`dev.sh` verifies Ollama is running with Gemma 4 pulled (or that a `GEMINI_API_KEY` is set when `MODEL_PROVIDER=gemini`), creates a Python venv at `packages/brain/.venv`, and runs the brain tests.

An end-to-end user experience lands in Week 3 when the SvelteKit app is wired up.

## Migrating from v1

If you ran the v1 Streamlit app locally and want to keep your conversation history:

```bash
python tools/migrate_v1_data.py
```

The script copies `JuneAI-app/.june_memory/june.db` to the platform-appropriate location (`~/Library/Application Support/June/` on macOS) and archives the source.

## License

MIT. See [`LICENSE`](LICENSE).

## Contributing

Contribution guidelines will be published in Week 8. Until then, discussion happens in GitHub issues.
