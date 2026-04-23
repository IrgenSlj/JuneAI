# June

The open personal AI that remembers you.

June runs privately on your laptop via Gemma 4 and scales to the cloud via Gemini. It works identically in your browser, on your Mac, and on your iPhone. Everything is open source. Everything is free.

## Status

The web prototype is **shipped**. The prototype checklist is complete:

- **Intelligence** — `packages/brain/` runs Gemma 4 (local, via Ollama) or Gemini (cloud) behind one code path.
- **API** — `packages/api/` exposes a FastAPI surface with SSE streaming on `POST /chat`; Pydantic schemas generate the TypeScript client.
- **Memory** — three stores behind one `MemoryManager`: SQLite for structured facts, ChromaDB for semantic recall, a graph for entities and relationships. Recall runs before every turn; extract runs after.
- **Skills** — each skill is a standalone MCP server launched by a supervisor in the brain; the `/skills` page toggles them on and off at runtime.
- **Web shell** — SvelteKit PWA at `apps/web/` with installable manifest, service worker, first-run setup, settings, memory browser, skills registry, offline states, keyboard shortcuts, and an accessibility pass.
- **Branding** — June "J" wordmark with light mode default and dark mode toggle.

See [`docs/product/roadmap.md`](docs/product/roadmap.md) for the item-by-item breakdown and what comes next (gated on user traction, not calendar).

The original v1 Streamlit prototype is preserved on the `legacy/streamlit` branch for historical reference.

## Read These First

1. [Vision](docs/vision.md) — what June is and the three non-negotiables
2. [Product overview](docs/product/overview.md) — the surfaces and the product boundary
3. [Roadmap](docs/product/roadmap.md) — what ships next and what triggers the next surface
4. [Architecture overview](docs/architecture/overview.md) — the layered model
5. [Architecture decisions](docs/decisions/README.md) — the ADRs that justify the design
6. [Environment](docs/setup/environment.md) — configuration reference

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

Open http://localhost:5173 for the chat surface. The app defaults to **light mode** — click the moon icon in the header to switch to dark mode. From there you can reach:

- `/setup` — first-run provider selection and key verification
- `/memory` — browse, search, and forget anything June has learned
- `/skills` — toggle MCP skills on and off at runtime
- `/settings` — switch providers, update your Gemini key, or toggle theme
- `/help/ollama` — troubleshooting for local Gemma via Ollama

## Migrating from v1

If you ran the v1 Streamlit app locally and want to keep your conversation history:

```bash
python tools/migrate_v1_data.py
```

The script copies `JuneAI-app/.june_memory/june.db` to the platform-appropriate location (`~/Library/Application Support/June/` on macOS) and archives the source.

## License

MIT. See [`LICENSE`](LICENSE).

## Contributing

Formal contribution guidelines are not yet published — the project is still hardening its first surface. Discussion happens in GitHub issues; pull requests are welcome for bug fixes and for items listed in [`docs/product/roadmap.md`](docs/product/roadmap.md).
