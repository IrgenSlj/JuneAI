<p align="center">
  <img src="June%20AI%20logo.png" alt="June" width="96" />
</p>

<h1 align="center">June</h1>

<p align="center">
  <strong>The open personal agent that remembers you.</strong><br />
  Private by default. Local-first. Yours to run, read, and extend.
</p>

<p align="center">
  <a href="https://github.com/IrgenSlj/JuneAI/actions/workflows/checks.yml"><img src="https://github.com/IrgenSlj/JuneAI/actions/workflows/checks.yml/badge.svg" alt="CI status" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license" /></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg" alt="Python 3.10-3.12" />
  <img src="https://img.shields.io/badge/node-20%2B-339933.svg" alt="Node 20+" />
  <img src="https://img.shields.io/badge/status-alpha-orange.svg" alt="Alpha" />
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs welcome" />
</p>

---

June is a personal AI agent that runs on your machine, remembers what matters to
you, and does real work across your files and services — without an account and
without sending your data anywhere you didn't approve.

Chat and recall run locally on **Gemma 4** via [Ollama](https://ollama.com).
Agentic work reaches **Gemini** only when you allow it, and every cloud call is
visible in the UI before and after it happens. Your conversations, memories, and
embeddings live in a local SQLite database and an on-disk vector store. There is
no signup, no telemetry without consent, and one button to export everything.

> **Status:** June is alpha software under active development. It is usable today
> as a web app; the desktop shell and OAuth-backed service skills are in flight.
> See the [roadmap](docs/product/roadmap.md) for what ships next.

## Why June

Most assistants ask you to trade privacy for capability. June is a bet that you
shouldn't have to. Five principles are enforced in code, not just promised:

- **No account.** June is installed, not subscribed to. No signup, no login, no
  cloud sync by default.
- **No silent cloud calls.** Every cloud-routed model call and every external
  service call is surfaced in the UI. A privacy dial in settings can lock June
  to local-only.
- **No telemetry without consent.** Nothing leaves your device unless you opt in.
- **One brain, every surface.** Browser, desktop, and (planned) mobile share the
  same memory, the same agent, and the same UI build. No shell-specific business
  logic.
- **Your data is portable.** Export is one command; delete is one button.

## Highlights

What is on `main` and working today:

- **Memory that compounds.** Three stores behind one `MemoryManager` facade —
  SQLite for structured facts, [ChromaDB](https://www.trychroma.com) for semantic
  recall, and a graph for entities and relationships. Every turn recalls relevant
  context before responding and extracts new facts afterward. The `/memory`
  browser lets you inspect, search, edit, copy, and forget anything June knows.
- **Three-tier model routing.** A router resolves `SkillModelPolicy × UserPrivacyDial → ResolvedTier`
  on every call, so a single turn can mix local recall, local planning, and one
  cloud-required tool call. Each assistant message carries a provenance chip
  showing which tier and model produced it.
- **Tasks as first-class work.** Long-running, observable units of work that
  survive the conversation that spawned them. The runtime pipes a goal through
  the agent and records every tool call as a step; the trace streams live over
  Server-Sent Events as the task runs. Start and cancel work end-to-end.
- **Skills as MCP servers.** Each capability (calendar, health, research,
  sandboxed files, journaling) is a standalone [Model Context Protocol](https://modelcontextprotocol.io)
  server, independently toggled and supervised. Any third-party MCP server is
  installable in one click from an in-app registry, and there is a per-tool
  playground for trying any tool by hand.
- **A trust dashboard.** `/system` shows the live runtime status and a rolling
  activity log of every API request and tool call — status, latency, label — so
  you can always answer "what did June just do?"

## Architecture

June is layered. Each layer calls only into the one below it.

```
┌───────────────────────────────────────────────────────────────┐
│  SHELLS     Tauri (desktop)   Capacitor (mobile)   PWA (web)   │
├───────────────────────────────────────────────────────────────┤
│  UI         SvelteKit app + shared TypeScript components        │
├───────────────────────────────────────────────────────────────┤
│  API        FastAPI · REST + SSE streaming                      │
├───────────────────────────────────────────────────────────────┤
│  BRAIN      LangGraph agent · memory · skills supervisor        │
├───────────────────────────────────────────────────────────────┤
│  PROVIDERS  Ollama / Gemma 4 (local)      Gemini (cloud)        │
└───────────────────────────────────────────────────────────────┘
                              ↑
                    ┌─────────┴──────────┐
                    │  SKILLS (MCP)      │
                    │  calendar, health, │
                    │  research, files,  │
                    │  daily             │
                    └────────────────────┘
```

The **brain** is the intelligence and is usable on its own — a Python developer
can depend on `june-brain` and embed June without the HTTP layer. The **API** is
a deliberately thin FastAPI boundary that streams the compiled agent. The **UI**
is a single SvelteKit build that every shell wraps. See
[docs/architecture/overview.md](docs/architecture/overview.md) for the full
picture and [docs/decisions/](docs/decisions/) for the ADRs behind each choice.

## Quickstart

**Prerequisites**

- Node.js 20+ with [`pnpm`](https://pnpm.io)
- Python 3.10+ (CI covers 3.10, 3.11, and 3.12)
- One model provider:
  - [Ollama](https://ollama.com) with `gemma4:e4b` pulled (fully local), **or**
  - a [Gemini API key](https://aistudio.google.com) (cloud)

**Install and verify**

```bash
git clone https://github.com/IrgenSlj/JuneAI.git
cd JuneAI
cp .env.example .env
./tools/bootstrap.sh   # creates packages/brain/.venv, installs Python + pnpm deps
./tools/check.sh       # backend tests, frontend checks, OpenAPI drift check
```

`bootstrap.sh` is idempotent and safe to re-run. If you only want to run the
backend tests without a model installed, `check.sh` already skips provider
probes; use `JUNE_SKIP_MODEL_CHECK=1 ./tools/dev.sh` when you want the provider
sanity check too.

**Run the stack**

```bash
# terminal 1 — API
packages/brain/.venv/bin/june-api

# terminal 2 — web app
pnpm dev
```

Open <http://localhost:5173>. First run walks you through choosing a provider and
verifying it with a single round-trip. From there:

| Route       | What it does                                                        |
| ----------- | ------------------------------------------------------------------- |
| `/`         | Chat with streaming responses, inline tool calls, provenance chips  |
| `/tasks`    | Give June work; watch the step trace stream in live                 |
| `/memory`   | Browse, search, edit, and forget what June remembers                |
| `/skills`   | Toggle MCP skills, try tools in a playground, browse the registry   |
| `/system`   | Runtime status and a live activity log of every request and tool    |
| `/settings` | Privacy dial, provider switch, Gemini key, theme                    |

## Repository layout

```
JuneAI/
├── apps/
│   ├── web/          SvelteKit PWA — the primary shipped surface
│   └── desktop/      Tauri shell (in development)
├── packages/
│   ├── brain/        Python: LangGraph agent, memory, skills supervisor, routing
│   ├── api/          Python: FastAPI surface (REST + SSE)
│   ├── ui/           Shared Svelte components + the generated typed API client
│   └── design/       Design tokens
├── skills/           MCP skill servers: calendar, health, research, files, daily
├── docs/             Vision, architecture, ADRs, product plans, setup guides
└── tools/            bootstrap.sh, check.sh, dev.sh, codegen.sh
```

## Tech stack

- **Brain:** Python, [LangGraph](https://langchain-ai.github.io/langgraph/),
  LangChain, ChromaDB, SQLite, an OpenAI-compatible client for both providers.
- **API:** [FastAPI](https://fastapi.tiangolo.com) with SSE streaming. Pydantic
  schemas are the single source of truth; the TypeScript client is generated
  from the OpenAPI spec.
- **UI:** [SvelteKit](https://svelte.dev) 5 (runes), installable PWA, light/dark.
- **Skills:** [Model Context Protocol](https://modelcontextprotocol.io) over
  stdio, one supervised subprocess per skill.
- **Shells:** [Tauri](https://tauri.app) (desktop), [Capacitor](https://capacitorjs.com)
  (mobile, planned).

## Development

The project gate is one command and is exactly what CI runs:

```bash
./tools/check.sh
```

It runs the backend test suite (`pytest`), the frontend type/lint checks
(`svelte-check`), and an OpenAPI codegen drift check that fails if the generated
TypeScript client is out of sync with the Pydantic schemas. CI runs the same gate
across Python 3.10/3.11/3.12 plus a frontend build.

When you change a Pydantic schema or an API route, regenerate the client:

```bash
./tools/codegen.sh
```

To export June's memory as an [Obsidian](https://obsidian.md) vault (Markdown
notes plus an architecture Canvas):

```bash
python tools/export_obsidian.py --user local --vault ~/JuneMemory
```

## Roadmap

June is mid-way through an [agentic pivot](docs/product/agentic-pivot-plan.md) —
a transformation from "chat with memory" to "personal agent with memory." The
agentic core (router, tasks, files skill, MCP registry, live trace, provenance)
is shipped. Next up are OAuth-backed Gmail and Calendar skills, a browser
automation skill, and signed desktop installers. The plan is trigger-gated, not
date-driven — read [docs/product/roadmap.md](docs/product/roadmap.md) for what
unlocks each surface.

## Contributing

Contributions are welcome. The bar for a change is simple: it keeps the
local-first privacy model understandable and does not make first-run setup harder
for a newcomer.

- Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup and the PR checklist, and
  [SECURITY.md](SECURITY.md) for responsible disclosure.
- Good first contributions: a new MCP skill, a memory-browser improvement, a
  provider edge case, or anything on the [roadmap](docs/product/roadmap.md).
- Keep PRs focused; add or update tests for behavior changes in `packages/brain`
  or `packages/api`; run `./tools/check.sh` before pushing.
- Any change that touches the privacy boundary must make cloud-mode behavior
  visible in the UI and the docs.

Discussion happens in [GitHub issues](https://github.com/IrgenSlj/JuneAI/issues).

## Documentation

- [Vision](docs/vision.md) — what June is and the non-negotiables
- [Product overview](docs/product/overview.md) — the surfaces and the boundary
- [Architecture overview](docs/architecture/overview.md) — the layered model
- [Architecture Decision Records](docs/decisions/) — why the design is the way it is
- [Agentic pivot plan](docs/product/agentic-pivot-plan.md) — the active execution plan
- [Roadmap](docs/product/roadmap.md) — what ships next
- [Environment reference](docs/setup/environment.md) — configuration options

## License

[MIT](LICENSE).
