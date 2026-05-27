# Environment Reference

This file is the canonical reference for June's runtime configuration. Per [ADR 0002](../decisions/0002-gemma-gemini-only.md), only Gemma 4 and Gemini are supported.

## Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `MODEL_PROVIDER` | Yes | `gemma` | One of `gemma` (local, via Ollama) or `gemini` (cloud, via Google AI Studio) |
| `GEMMA_MODEL` | No | `gemma4:e2b` | Ollama tag for the local Gemma model. Must match `ollama list` |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434/v1` | Base URL for the local Ollama server |
| `GEMINI_API_KEY` | For `gemini` | — | API key from https://aistudio.google.com |
| `GEMINI_MODEL` | No | `gemini-2.0-flash` | Gemini model identifier |
| `JUNE_DATA_DIR` | No | platform default | Directory for `june.db` and ChromaDB index |
| `JUNE_LOG_DIR` | No | platform default | Directory for structured logs |
| `MODEL_TEMPERATURE` | No | `0.4` | Applied to both providers |
| `MODEL_MAX_TOKENS` | No | `4096` | Applied to both providers |
| `JUNE_SKIP_MODEL_CHECK` | No | — | Set to `1` when running developer tests without Ollama/Gemini configured |
| `PYTHON_BIN` | No | `python3` for bootstrap, venv Python for checks | Python executable used by developer scripts |
| `JUNE_VENV` | No | `packages/brain/.venv` | Virtualenv path used by developer scripts |
| `JUNE_SKIP_PNPM_INSTALL` | No | — | Set to `1` to skip `pnpm install` in `tools/bootstrap.sh` |
| `JUNE_CHECK_FRONTEND` | No | `1` | Set to `0` to skip frontend checks in `tools/check.sh` |
| `JUNE_CHECK_CODEGEN` | No | `1` | Set to `0` to skip OpenAPI codegen drift checks in `tools/check.sh` |

## Default Data Directory

| Platform | Path |
|---|---|
| macOS | `~/Library/Application Support/June/` |
| Linux | `~/.local/share/June/` |
| Windows | `%APPDATA%/June/` |
| iOS | app sandbox container |

## Recommended Local `.env`

```env
# Local inference (default)
MODEL_PROVIDER=gemma
GEMMA_MODEL=gemma4:e2b
OLLAMA_BASE_URL=http://localhost:11434/v1

# Optional cloud escape valve
# MODEL_PROVIDER=gemini
# GEMINI_API_KEY=your_key_here
# GEMINI_MODEL=gemini-2.0-flash

# Optional overrides
# JUNE_DATA_DIR=
# MODEL_TEMPERATURE=
# MODEL_MAX_TOKENS=
```

## Switching Between Local and Cloud

Toggle `MODEL_PROVIDER` between `gemma` and `gemini`. Both can be configured at the same time; the active one is selected by `MODEL_PROVIDER`. This is deliberate: users who want both can switch with one line.

Privacy boundary: memory files stay in `JUNE_DATA_DIR` for both providers. With `MODEL_PROVIDER=gemini`, June sends the current prompt plus relevant recalled memory context to Google's API for inference. Use `MODEL_PROVIDER=gemma` when inference must remain local.

## Persistent User Choices

The `/setup` and `/settings` screens write the user's provider pick to `JUNE_DATA_DIR/config.json` (mode 0600). The brain overlays that file onto `os.environ` at startup; any value already set in the environment (including `.env`) wins, so developers can still force a provider from `.env`.

The Gemini API key goes through a separate path:

- **macOS Keychain, Linux Secret Service, Windows Credential Manager** — used automatically via the `keyring` package when a backend is available. The key never lands in `config.json` in this case.
- **Mode-0600 JSON fallback** — headless Linux, Docker, CI, or any environment without a keyring backend. `config.json` still holds the key with restrictive permissions.

The `/settings` screen shows which of the two is active. "Forget key" removes the value from whichever store holds it.

## First-Run Setup

1. Install Ollama: `brew install ollama` on macOS.
2. Pull Gemma 4: `ollama pull gemma4:e2b`.
3. Start Ollama: `ollama serve` (runs in the background on first `ollama` command).
4. Set `MODEL_PROVIDER=gemma` in `.env`.

Cloud-only users skip Ollama entirely and set `MODEL_PROVIDER=gemini` with a `GEMINI_API_KEY`.

Developers who only need to run the automated test suite can skip provider verification:

```bash
JUNE_SKIP_MODEL_CHECK=1 ./tools/dev.sh
```

For fresh clones, prefer the split developer commands:

```bash
./tools/bootstrap.sh  # install Python workspace and pnpm deps when needed
./tools/check.sh      # backend tests, frontend checks, OpenAPI type drift check
```

The desktop shell will perform steps 1 through 3 on the user's behalf with one click — see [ADR 0008](../decisions/0008-ollama-supervision.md). The web shell continues to require manual setup as documented above.

## Skill Subprocess Variables

Set by the brain's skill supervisor when spawning an MCP skill server. Never set these manually in a shell.

| Variable | Set By | Purpose |
|---|---|---|
| `JUNE_IS_SKILL_SUBPROCESS` | Supervisor | Signals that the process is a skill child. The brain's graph module skips agent construction under this flag to prevent a recursive fork bomb. |
| `JUNE_SKILLS_DISABLED` | Supervisor | Defense in depth — prevents `get_supervisor()` from auto-starting skill subprocesses in children. |

See [ADR 0005](../decisions/0005-skills-as-mcp.md) for the full rationale.
