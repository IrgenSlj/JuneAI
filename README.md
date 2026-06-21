<p align="center">
  <img src="assets/logo.png" alt="June" width="96" />
</p>

<h1 align="center">June</h1>

<p align="center">
  <strong>The open personal agent that remembers you.</strong><br />
  Private by default. Local-first. Yours to run, read, and extend.
</p>

<p align="center">
  <a href="https://github.com/IrgenSlj/JuneAI/actions/workflows/checks.yml"><img src="https://github.com/IrgenSlj/JuneAI/actions/workflows/checks.yml/badge.svg" alt="CI status" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license" /></a>
  <img src="https://img.shields.io/badge/python-3.13-3776AB.svg" alt="Python 3.13" />
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

> **Status:** June is alpha software under active development. It is usable as a
> web app and has a v0.1.0 Apple Silicon macOS DMG on GitHub Releases (ad-hoc
> signed, not notarized, so macOS may show a first-launch warning). The **Tier 1
> spine** of the canonical [build specification](docs/product/build-spec.md) is
> built — portable data directory, model-specific provider layer, measured harness
> loop, layered context with anchored compaction, salience recall, an honest
> character, and a visible cloud boundary. Current focus: running the loop
> experiment and making that measured loop the live path. See the
> [roadmap](ROADMAP.md).

June's center of gravity is the user, not the task. She borrows a coding agent's
skeleton but inverts its four operations: she **defers** to the user instead of
verifying against ground truth, **continues** standing intentions instead of
completing and exiting, **forgets** gracefully instead of accumulating, and knows
when to **stay quiet** instead of acting fast.

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

On `main` today:

- **Memory that remembers what matters.** Three stores behind one `MemoryManager`
  — SQLite facts, a sqlite-vec semantic index, and an
  entity graph. Recall is ranked by *salience* (recency × frequency × relevance),
  not similarity alone. Browse, search, edit, and forget anything at `/memory`.
- **A visible cloud boundary.** Every turn carries a provenance line — which model
  ran, whether anything left the device, and a plain-English rationale. A privacy
  dial can lock June to local-only, which provably blocks egress.
- **One honest voice.** June's character is a self-authored block with honesty and
  a behavioral safety floor as a fixed, non-editable core — personalization can
  shape tone, never erode candor into flattery.
- **Portable by design.** Everything June is lives under one documented, versioned
  data directory — copy the folder to move machines.
- **Tasks, skills, and a trust surface.** Long-running tasks with a live SSE step
  trace; capabilities as standalone [MCP](https://modelcontextprotocol.io) servers,
  independently toggled; `/system` shows a rolling log of every request and tool call.

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
│  BRAIN      agent loop · context · memory · character · skills  │
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
- Python 3.13
- One model provider:
  - [Ollama](https://ollama.com) with `gemma4:e2b` pulled (fully local; `run.sh` pulls it for you), **or**
  - a [Gemini API key](https://aistudio.google.com) (cloud)

**Install and verify**

```bash
git clone https://github.com/IrgenSlj/JuneAI.git
cd JuneAI
cp .env.example .env
./tools/bootstrap.sh   # creates packages/brain/.venv, installs Python + pnpm deps
./tools/check.sh       # backend tests, frontend checks, OpenAPI drift check
```

`bootstrap.sh` is idempotent and safe to re-run. `dev.sh` is a check-only script
(it verifies Ollama/Gemini readiness and runs the gate); it does not start the
app.

**Run the stack**

```bash
./tools/run.sh   # starts Ollama if needed, pulls the model, runs API + web; Ctrl-C stops all
```

Or run the pieces yourself in two terminals:

```bash
packages/brain/.venv/bin/june-api   # API
pnpm dev                            # web app
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
│   ├── brain/        Python: provider layer, harness loop, context, memory, character, skills
│   ├── api/          Python: FastAPI surface (REST + SSE)
│   ├── ui/           Shared Svelte components + the generated typed API client
│   └── design/       Design tokens
├── skills/           MCP skill servers: calendar, health, research, files, daily
├── docs/             Vision, architecture, ADRs, product plans, setup guides
└── tools/            run.sh (launch), bootstrap.sh, check.sh, preflight.sh, codegen.sh
```

## Tech stack

- **Brain:** Python with a hand-written harness loop (one engine, no agent
  framework — see ADR 0018), three-store memory (one SQLite db: structured rows + a sqlite-vec vector index + a graph),
  and a model-specific provider layer (Gemma 4 + Gemini).
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
on Python 3.13 plus a frontend build.

When you change a Pydantic schema or an API route, regenerate the client:

```bash
./tools/codegen.sh
```

## Roadmap

The **Tier 1 spine** is built — portable data directory, model-specific provider
layer, measured harness loop, layered context with anchored compaction, salience
recall, honest character, and a visible cloud boundary — on top of the shipped
foundation (three-store memory, tasks, MCP skills, desktop DMG). Current focus is
finishing Tier 1: run the loop experiment and make the measured loop the live path.
Tier 2 (proactivity, native memory graph, encrypted backup, Google skills) starts
only after Tier 1 is used. See [ROADMAP.md](ROADMAP.md) and the
[build specification](docs/product/build-spec.md).

## Contributing

Contributions are welcome. The bar for a change is simple: it keeps the
local-first privacy model understandable and does not make first-run setup harder
for a newcomer.

- Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup and the PR checklist, and
  [SECURITY.md](SECURITY.md) for responsible disclosure.
- Good first contributions: a memory-browser improvement, a provider edge case, a
  salience-weight or character-shaping refinement, a small MCP skill, or anything
  on the [roadmap](ROADMAP.md).
- Keep PRs focused; add or update tests for behavior changes in `packages/brain`
  or `packages/api`; run `./tools/check.sh` before pushing.
- Any change that touches the privacy boundary must make cloud-mode behavior
  visible in the UI and the docs.

Discussion happens in [GitHub issues](https://github.com/IrgenSlj/JuneAI/issues).

## Documentation

- [Build specification](docs/product/build-spec.md) — the authoritative, decision-by-decision plan
- [Vision](docs/vision.md) — what June is and the non-negotiables
- [Product overview](docs/product/overview.md) — the surfaces and the boundary
- [Architecture overview](docs/architecture/overview.md) — the layered model
- [Architecture Decision Records](docs/decisions/) — why the design is the way it is
- [Roadmap](docs/product/roadmap.md) — what ships next
- [Environment reference](docs/setup/environment.md) — configuration options

## License

[MIT](LICENSE).
