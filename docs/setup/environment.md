# Environment Variables

This file is the canonical reference for JuneAI runtime configuration.

## Core Variables

| Variable | Required | Typical value | Purpose | Notes |
|---|---|---|---|---|
| `MODEL_PRESET` | Yes | `local_gemma_4` | Selects the named runtime preset | Recommended primary switch |
| `LLM_BASE_URL` | Local/OpenAI-compatible | `http://localhost:11434/v1` | Base URL for Ollama or another compatible endpoint | Ignored for default Claude use |
| `LLM_API_KEY` | Local/OpenAI-compatible | `ollama` | API key for OpenAI-compatible endpoints | Usually `ollama` for local use |
| `MEMORY_DIR` | No | `.june_memory` | Directory that contains `june.db` | Relative to `JuneAI-app/` unless absolute |

## Provider and Model Overrides

| Variable | Required | Typical value | Purpose | Notes |
|---|---|---|---|---|
| `MODEL_PROVIDER` | No | `openai_compatible` or `anthropic` | Forces provider selection | Advanced override only |
| `MODEL_NAME` | No | blank | Overrides the resolved preset model directly | Use sparingly |
| `MODEL_TEMPERATURE` | No | blank | Overrides preset temperature | Numeric string |
| `MODEL_MAX_TOKENS` | No | blank | Overrides preset max tokens | Integer string |
| `MODEL_TOOL_STRATEGY` | No | blank | Overrides tool strategy | Advanced/runtime debugging |

## Preset-Specific Model Tag Overrides

| Variable | Required | Typical value | Used by | Notes |
|---|---|---|---|---|
| `LOCAL_LLAMA_MODEL_NAME` | No | `llama3.2:3b` | `local_llama3_2` | Optional |
| `LOCAL_SMALL_MODEL_NAME` | No | `mistral` | `local_mistral_3b` | Optional |
| `LOCAL_LARGE_MODEL_NAME` | No | blank | `local_mistral_7b`, `local_mistral_8b` | Shared by both presets; leave blank unless intentionally overriding both |
| `LOCAL_GEMMA_MODEL_NAME` | No | `gemma4:e4b` | `local_gemma_4` | Must match `ollama list` if set |
| `CLAUDE_MODEL_NAME` | No | `claude-sonnet-4-6` | `claude_high` | Optional override |

## API Credentials

| Variable | Required | Typical value | Purpose | Notes |
|---|---|---|---|---|
| `ANTHROPIC_API_KEY` | For `claude_high` | blank | Auth for Anthropic runtime | Required only for Claude |

## Recommended Local `.env`

```env
MODEL_PRESET=local_gemma_4
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
MEMORY_DIR=.june_memory

# Optional model tag overrides
LOCAL_GEMMA_MODEL_NAME=gemma4:e4b
LOCAL_LLAMA_MODEL_NAME=llama3.2:3b

# Leave blank unless intentionally overriding preset defaults
LOCAL_LARGE_MODEL_NAME=
LOCAL_SMALL_MODEL_NAME=
MODEL_NAME=
MODEL_PROVIDER=
MODEL_TEMPERATURE=
MODEL_MAX_TOKENS=
MODEL_TOOL_STRATEGY=
CLAUDE_MODEL_NAME=
ANTHROPIC_API_KEY=
```

## Important Reminders

- `LOCAL_LARGE_MODEL_NAME` is shared by both large Mistral presets in the current config implementation.
- If you switch to Claude, make sure `ANTHROPIC_API_KEY` is set.
- If your app behavior does not match the docs, inspect your local `JuneAI-app/.env` first.
- `make check-ollama` should be run after changing local runtime settings.
