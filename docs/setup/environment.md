# Environment Reference

This file is the canonical reference for June's runtime configuration. Per [ADR 0002](../decisions/0002-gemma-gemini-only.md), only Gemma 4 and Gemini are supported.

## Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `MODEL_PROVIDER` | Yes | `gemma` | One of `gemma` (local, via Ollama) or `gemini` (cloud, via Google AI Studio) |
| `GEMMA_MODEL` | No | `gemma4:e4b` | Ollama tag for the local Gemma model. Must match `ollama list` |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434/v1` | Base URL for the local Ollama server |
| `GEMINI_API_KEY` | For `gemini` | — | API key from https://aistudio.google.com |
| `GEMINI_MODEL` | No | `gemini-2.0-flash` | Gemini model identifier |
| `JUNE_DATA_DIR` | No | platform default | Directory for `june.db` and ChromaDB index |
| `JUNE_LOG_DIR` | No | platform default | Directory for structured logs |
| `MODEL_TEMPERATURE` | No | `0.4` | Applied to both providers |
| `MODEL_MAX_TOKENS` | No | `4096` | Applied to both providers |

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
GEMMA_MODEL=gemma4:e4b
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

## First-Run Setup

1. Install Ollama: `brew install ollama` on macOS.
2. Pull Gemma 4: `ollama pull gemma4:e4b`.
3. Start Ollama: `ollama serve` (runs in the background on first `ollama` command).
4. Set `MODEL_PROVIDER=gemma` in `.env`.

Cloud-only users skip Ollama entirely and set `MODEL_PROVIDER=gemini` with a `GEMINI_API_KEY`.

## Skill Subprocess Variables

Set by the brain's skill supervisor when spawning an MCP skill server. Never set these manually in a shell.

| Variable | Set By | Purpose |
|---|---|---|
| `JUNE_IS_SKILL_SUBPROCESS` | Supervisor | Signals that the process is a skill child. The brain's graph module skips agent construction under this flag to prevent a recursive fork bomb. |
| `JUNE_SKILLS_DISABLED` | Supervisor | Defense in depth — prevents `get_supervisor()` from auto-starting skill subprocesses in children. |

See [ADR 0005](../decisions/0005-skills-as-mcp.md) for the full rationale.
