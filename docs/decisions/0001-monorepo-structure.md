# ADR 0001: Monorepo Structure with apps/packages/skills Separation

**Status:** Accepted
**Date:** 2026-04-17

## Context

The v1 repository assumes a single application. Everything — the LangGraph agent, the memory layer, the Streamlit UI, the tools, user data — lives inside `JuneAI-app/`. The `app.py` file is 3,726 lines. Memory and tools files each exceed 1,400 lines.

The v2 vision requires four distinct surfaces: a browser app, a Mac desktop app, an iPhone app, and a reusable Python brain that developers could embed. A single-app layout cannot support this. Without structural separation, every new surface would copy logic, diverge, and accumulate inconsistency.

## Decision

The repository becomes a monorepo with three top-level code directories:

- `apps/` contains end-user applications: `web`, `desktop`, `mobile`. These are what users download.
- `packages/` contains internal libraries: `brain` (Python agent core), `api` (FastAPI surface), `ui` (shared TypeScript components), `design` (tokens and icons).
- `skills/` contains Model Context Protocol servers, one per domain: `calendar`, `health`, `research`, `files`, `daily`.

Documentation lives in `docs/`. Developer tooling lives in `tools/`. User data never lives in the repo; it moves to platform-appropriate locations (`~/Library/Application Support/June/` on macOS).

JavaScript workspaces are managed with pnpm. Python workspaces are managed with uv. Each package has its own dependency graph and can be tested in isolation.

## Consequences

**Positive:**

- Each layer has one reason to change, one owner, one dependency graph.
- The brain can be published to PyPI; the UI package can be published to npm. Distribution becomes a moat.
- Skills can be developed, versioned, and released independently, including by community contributors.
- Tests run only for packages affected by a change.
- A new surface (e.g., a CLI) adds a folder under `apps/` without touching existing code.

**Negative:**

- Initial setup is more complex than a single-app layout. One-time cost, paid once.
- Developers must understand the boundary between `apps` and `packages`. This is documented here and in `docs/architecture/overview.md`.
- Cross-package refactors require coordinated changes. Mitigated by the monorepo keeping everything in one commit.

## Alternatives Considered

**Multi-repo.** One repo per surface. Rejected because the brain, the API, and the UI evolve together for most features; splitting them would force cross-repo PRs for every change.

**Keep the single-app layout.** Grow `JuneAI-app/` by adding subfolders. Rejected because `app.py` already shows the limits of this approach at one surface; adding desktop and mobile would collapse the structure entirely.

**Nx or Turborepo.** Heavier monorepo tooling. Rejected for now — pnpm workspaces plus uv workspaces plus a small `tools/dev.sh` script is enough. Revisit if build coordination becomes a real bottleneck.
