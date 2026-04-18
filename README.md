# June

The open personal AI that remembers you.

June runs privately on your laptop via Gemma 4 and scales to the cloud via Gemini. It works identically in your browser, on your Mac, and on your iPhone. Everything is open source. Everything is free.

## Status

June is in early development. Weeks 1 through 4 are complete:

- **Week 1** — monorepo reshaped; `packages/brain/` holds the intelligence layer; Gemma 4 and Gemini wired through one code path.
- **Week 2** — FastAPI boundary at `packages/api/` with SSE streaming on `POST /chat`; TypeScript types generated from Pydantic schemas.
- **Week 3** — SvelteKit PWA at `apps/web/` streaming live tokens from Gemma; shared components in `packages/ui/`.
- **Week 4** — three-store memory shipped (SQLite + ChromaDB + knowledge graph) behind a single `MemoryManager`; recall on every turn, extract runs post-stream; `/memory` browser with per-fact delete.

Week 5 (skills as MCP servers) is next. See [`docs/product/plan.md`](docs/product/plan.md) for the current exit criteria.

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

To run the full stack locally:

```bash
# terminal 1 — API
packages/brain/.venv/bin/uvicorn june_api.app:app --reload --port 8000

# terminal 2 — web
pnpm --filter @june/web dev
```

Open http://localhost:5173 for the chat surface and http://localhost:5173/memory for the memory browser.

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
