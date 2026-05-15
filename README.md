# June

An open personal AI that remembers you.

June is a local-first assistant with long-term memory. It runs on your laptop via Ollama/Gemma 4, or can use Gemini when you choose cloud inference. Memory is stored on your machine; in Gemini mode the current prompt and relevant recalled context are sent to Google's API.

## Status

June is **alpha software**. The web PWA is the primary working surface, but
the project is still in an open-source readiness pass before it should be
presented as broadly download-and-use software:

- **Intelligence** — `packages/brain/` runs Gemma 4 (local, via Ollama) or Gemini (cloud) behind one code path.
- **API** — `packages/api/` exposes a FastAPI surface with SSE streaming on `POST /chat`; Pydantic schemas generate the TypeScript client.
- **Memory** — three stores behind one `MemoryManager`: SQLite for structured facts, ChromaDB for semantic recall, a graph for entities and relationships. Recall runs before every turn; extract runs after.
- **Skills** — each skill is a standalone MCP server launched by a supervisor in the brain; the `/skills` page toggles them on and off at runtime.
- **Web shell** — SvelteKit PWA at `apps/web/` with installable manifest, service worker, first-run setup, settings, memory browser, skills registry, offline states, keyboard shortcuts, and an accessibility pass.
- **Branding** — June "J" wordmark with light mode default and dark mode toggle.

The current hardening priorities are provider correctness, conversation
continuity, memory delete/edit correctness across all stores, fresh-clone setup,
local API safety, and desktop build CI. See the
[open-source readiness plan](docs/product/open-source-readiness-plan.md) for
the detailed development plan.

The **desktop shell** is experimental. Phases 1–4 of the [desktop-shell plan](docs/product/desktop-shell-plan.md) are implemented in source, but Rust compilation, packaging, signing, and distribution CI are still part of the hardening track. Mobile is planned, not shipped.

See [`docs/product/roadmap.md`](docs/product/roadmap.md) for the item-by-item breakdown and what comes next (gated on user traction, not calendar).

The original v1 Streamlit prototype is preserved on the `legacy/streamlit` branch for historical reference.

## Read These First

1. [Vision](docs/vision.md) — what June is and the three non-negotiables
2. [Product overview](docs/product/overview.md) — the surfaces and the product boundary
3. [Open-source readiness plan](docs/product/open-source-readiness-plan.md) — the hardening plan before public alpha
4. [Roadmap](docs/product/roadmap.md) — what ships next and what triggers the next surface
5. [Desktop shell plan](docs/product/desktop-shell-plan.md) — the active development plan
6. [Responsive and touch plan](docs/product/responsive-plan.md) — how the UI works on every screen
7. [Architecture overview](docs/architecture/overview.md) — the layered model
8. [Architecture decisions](docs/decisions/README.md) — the ADRs that justify the design
9. [Environment](docs/setup/environment.md) — configuration reference

## Repository Layout

```
JuneAI/
├── apps/              # end-user apps: web and desktop (Tauri); mobile is planned
├── packages/          # internal libraries: brain, api, ui, design
├── skills/            # MCP skill servers: calendar, health, research, files, daily
├── docs/              # vision, architecture, decisions, product plan, setup
├── tools/             # developer tooling (dev.sh, migrate_v1_data.py)
└── README.md
```

For the rationale behind this layout see [ADR 0001](docs/decisions/0001-monorepo-structure.md).

## Quickstart

Install prerequisites:

- Node.js 20+ with `pnpm`
- Python 3.10+ (CI covers 3.10, 3.11, and 3.12)
- Ollama with `gemma4:e4b` pulled, or a Gemini API key

```bash
cp .env.example .env
./tools/bootstrap.sh
./tools/check.sh
```

`bootstrap.sh` creates a Python venv at `packages/brain/.venv`, installs the brain/API/skill packages editable, and runs `pnpm install` when needed. `check.sh` runs backend tests, frontend checks, and the OpenAPI type drift check.

To also verify the selected model provider before running backend tests:

```bash
./tools/dev.sh
```

Contributors who only want to run tests without installing Ollama can skip provider checks:

```bash
JUNE_SKIP_MODEL_CHECK=1 ./tools/dev.sh
```

To run the full stack locally:

```bash
# terminal 1 — API
packages/brain/.venv/bin/june-api

# terminal 2 — web
pnpm --filter @june/web dev
```

Open http://localhost:5173 for the chat surface. The app defaults to **light mode** — click the moon icon in the header to switch to dark mode. From there you can reach:

- `/setup` — first-run provider selection and key verification
- `/memory` — browse, search, and forget anything June has learned
- `/skills` — toggle MCP skills on and off at runtime
- `/settings` — switch providers, update your Gemini key, or toggle theme
- `/help/ollama` — troubleshooting for local Gemma via Ollama

To inspect June's memory outside the app, export an Obsidian vault:

```bash
python tools/export_obsidian.py --user local --vault ~/JuneMemory
```

The export writes Markdown notes for memory and skills plus an Obsidian Canvas
map of the system architecture. The same payload is available from
`GET /obsidian/{user_id}` for custom tooling.

## Migrating from v1

If you ran the v1 Streamlit app locally and want to keep your conversation history:

```bash
python tools/migrate_v1_data.py
```

The script copies `JuneAI-app/.june_memory/june.db` to the platform-appropriate location (`~/Library/Application Support/June/` on macOS) and archives the source.

## License

MIT. See [`LICENSE`](LICENSE).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), and
the [`open-source readiness plan`](docs/product/open-source-readiness-plan.md).
Discussion happens in GitHub issues; pull requests are welcome for bug fixes and
for items listed in the hardening plan and [`docs/product/roadmap.md`](docs/product/roadmap.md).
