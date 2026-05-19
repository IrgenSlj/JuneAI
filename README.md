# June

An open personal agent that remembers you.

June is a private-by-default personal agent with long-term memory. Chat and recall run locally on Gemma 4 via Ollama; agentic capability reaches Gemini when you allow it, with per-call visibility in the UI. Memory stays on your machine. No account, no signup.

## Status

June is **alpha software** in active development. As of 2026-05-19 the project is mid-way through the [agentic pivot](docs/product/agentic-pivot-plan.md) — a twelve-week reframing from "chat-with-memory" to "personal agent with memory" anchored by [ADR 0009](docs/decisions/0009-private-by-default-and-model-routing.md) and [ADR 0010](docs/decisions/0010-agentic-core-tasks-oauth-computer-use.md).

What's shipped on `main`:

- **Intelligence** — `packages/brain/` runs Gemma 4 (local) or Gemini (cloud) behind one code path. A three-tier model router resolves `SkillModelPolicy × UserPrivacyDial → ResolvedTier` per call (ADR 0009), and the user holds a `/settings` dial that controls cloud access (local-only, private-by-default, cloud-first).
- **Tasks** — first-class long-running units of work in their own sqlite table. `TaskRuntime` pipes a goal through the existing LangGraph agent and records every tool call as a step. `/tasks` page shows active and recent tasks with full traces; Start/Pause/Cancel controls work end-to-end.
- **Memory** — three stores behind one `MemoryManager`: SQLite for structured facts, ChromaDB for semantic recall, a graph for entities. `/memory` now opens with a stats card (per-store totals, last write, recent learnings) above the editable browser.
- **Skills** — each skill is a standalone MCP server launched by a supervisor. The `/skills` page toggles them, runs them in a per-tool playground (form-generated from input_schema), and browses a curated MCP registry of third-party servers (filesystem, github, notion, postgres, brave-search, sqlite).
- **System** — `/system` shows an at-a-glance architecture overview plus a rolling activity log of every API request and tool call (status, latency, label). The trust primitive: "what did June just do?"
- **API** — `packages/api/` exposes a FastAPI surface with SSE streaming on `POST /chat` and CRUD on `/tasks`, `/memory`, `/skills`, `/skills/registry`, `/system`. Pydantic schemas generate the TypeScript client.
- **Web shell** — SvelteKit PWA at `apps/web/` with installable manifest, service worker, first-run setup, settings, memory browser, skills registry, tasks page, system dashboard, offline states, keyboard shortcuts, light/dark, and an accessibility pass.
- **Branding** — June "J" wordmark with light mode default and dark mode toggle.

What's still pending in Sprint 1: chat-event provenance through the LangGraph stream (slice 1.1b), Gmail/Calendar OAuth skills (gated on Google verified-app review), Playwright browser skill, and the desktop-shell first compile (needs `rustup`).

The **desktop shell** is experimental. Phases 1–4 of the [desktop-shell plan](docs/product/desktop-shell-plan.md) are implemented in source; Phase 4.5 (first compile) is Sprint 1.7 of the pivot. Mobile is planned, not shipped.

The original v1 Streamlit prototype is preserved on the `legacy/streamlit` branch for historical reference.

## Read These First

1. [Vision](docs/vision.md) — what June is and the three non-negotiables
2. [Product overview](docs/product/overview.md) — the surfaces and the product boundary
3. [Agentic pivot plan](docs/product/agentic-pivot-plan.md) — the active twelve-week execution plan with sprint status
4. [Roadmap](docs/product/roadmap.md) — what ships next and what triggers the next surface
5. [ADR 0009 — Private-by-default with three-tier model routing](docs/decisions/0009-private-by-default-and-model-routing.md)
6. [ADR 0010 — Agentic core: tasks, OAuth, computer use, MCP](docs/decisions/0010-agentic-core-tasks-oauth-computer-use.md)
7. [Desktop shell plan](docs/product/desktop-shell-plan.md) — Phase 4.5 first compile still pending
8. [Architecture overview](docs/architecture/overview.md) — the layered model
9. [Architecture decisions](docs/decisions/README.md) — the ADRs that justify the design
10. [Environment](docs/setup/environment.md) — configuration reference

The [open-source readiness plan](docs/product/open-source-readiness-plan.md) is paused for the duration of the pivot and resumes after Sprint 4.

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

- `/tasks` — give June work to do; watch the trace as she does it
- `/memory` — browse, search, and forget anything June has learned; stats card at the top
- `/skills` — toggle MCP skills, try a tool in the playground, browse the registry of third-party servers
- `/system` — runtime status plus a rolling activity log of every recent request and tool call
- `/settings` — privacy dial, provider switch, Gemini key, theme toggle
- `/setup` — first-run provider selection and key verification
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

See [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), and the
[agentic pivot plan](docs/product/agentic-pivot-plan.md) for the current sprint
backlog. Discussion happens in GitHub issues; pull requests are welcome for bug
fixes and for items listed in the pivot plan and [`docs/product/roadmap.md`](docs/product/roadmap.md).
