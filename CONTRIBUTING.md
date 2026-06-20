# Contributing

June is alpha software. Contributions are welcome, but the bar for changes is that they keep the local-first privacy model understandable and do not make setup harder for a first-time user.

## Development Setup

Prerequisites:

- Node.js 20+
- `pnpm`
- Python 3.13
- Ollama with `gemma4:e2b`, or a Gemini API key if you want to run the app end to end

Bootstrap the repo:

```bash
cp .env.example .env
./tools/bootstrap.sh
```

Run the main checks:

```bash
./tools/check.sh
```

`check.sh` runs backend tests, frontend checks, the OpenAPI codegen drift check, Ruff lint, and a narrow mypy gate (the `operator` and `call-arg` real-bug classes must stay at zero). Run it before every push.

When you want the provider sanity checks as well, use:

```bash
./tools/preflight.sh
```

`JUNE_SKIP_MODEL_CHECK=1` is useful for contributors who only need backend tests through `dev.sh`. Remove it when you want `dev.sh` to verify Ollama/Gemini readiness.

Run the app:

```bash
# terminal 1
packages/brain/.venv/bin/june-api

# terminal 2
pnpm --filter @june/web dev
```

Open http://localhost:5173.

## Pull Requests

- Keep changes focused. Avoid mixing feature work, formatting, and docs cleanup in one PR.
- Add or update tests for behavior changes in `packages/brain` or `packages/api`.
- Regenerate API types with `./tools/codegen.sh` when changing Pydantic schemas or API routes.
- Do not commit local data, generated caches, `.env`, virtualenvs, or build output.
- Document any privacy boundary change explicitly. Cloud-mode behavior must be visible in UI/docs.

## Project Priorities

The authoritative direction is the [build specification](docs/product/build-spec.md)
and [ADRs 0015-0017](docs/decisions/). June is built in tiers; the Tier 1 spine
(C.0-C.6) is implemented and on `main`.

Current priorities (finishing Tier 1):

- Wire the hand-written loop as the live chat path — route the provider layer,
  layered context, character block, difficulty router, and capability probe
  through it, replacing the LangGraph agent as the live path (kept as a fallback).
- Browser-verify the visible cloud boundary and per-turn rationale in the chat UI.
- Then use the spine before starting Tier 2 differentiators.

Non-negotiable throughout: privacy is visible in code (no silent egress; local-only
blocks the cloud), honesty is not adjustable, and the harness core is fixed and
never self-modified. June acts on user input or real-world events, never on a timer.
