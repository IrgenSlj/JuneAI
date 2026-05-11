# Contributing

June is alpha software. Contributions are welcome, but the bar for changes is that they keep the local-first privacy model understandable and do not make setup harder for a first-time user.

## Development Setup

Prerequisites:

- Node.js 20+
- `pnpm`
- Python 3.10+
- Ollama with `gemma4:e4b`, or a Gemini API key if you want to run the app end to end

Bootstrap the repo:

```bash
cp .env.example .env
pnpm install
JUNE_SKIP_MODEL_CHECK=1 ./tools/dev.sh
```

`JUNE_SKIP_MODEL_CHECK=1` is useful for contributors who only need tests. Remove it when you want `dev.sh` to verify Ollama/Gemini readiness.

Run the main checks:

```bash
pnpm check
pnpm build
packages/brain/.venv/bin/python -m pytest packages/brain/tests packages/api/tests
```

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

Current hardening priorities:

- Reliable fresh-clone setup.
- Honest privacy and security boundaries.
- Memory correctness and user control.
- Desktop shell compilation, packaging, and release automation.
