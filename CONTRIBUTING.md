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

`check.sh` runs backend tests, frontend checks, and the OpenAPI codegen drift check. The broader Ruff/mypy policy is tracked in the open-source readiness plan and is not enforced yet.

When you want the provider sanity checks as well, use:

```bash
./tools/dev.sh
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

Current v0.1.1 priorities:

- Quick capture: turn messy input into structured candidates.
- Durable event ledger: record what June saw, proposed, and did.
- Action preview and approvals: protect calendar writes, notifications,
  messages, deletions, and cloud-required actions.
- Daily Home: simple first screen over the serious backend.
- Memory correctness and user control remain non-negotiable.
