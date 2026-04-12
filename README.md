# JuneAI

A local-first personal AI that feels like a friend who actually remembers you.

June is not a generic chatbot. She is a private life console: one place to talk, remember,
plan, and keep continuity across your daily life. Every conversation makes her more useful.
Everything stays on your machine.

---

## What June Does

- Captures appointments, reminders, birthdays, trips, and commitments from natural conversation
- Tracks goals and open follow-ups, flags when something has gone stale
- Logs workouts, body metrics, nutrition, and habits
- Remembers people, relationships, and personal context
- Notices patterns: low energy streaks, broken habit chains, upcoming deadlines
- Opens each session with context-aware observations, not a blank slate
- Generates weekly summaries and contextual suggestions
- Updates the right-rail dashboard in real time as the conversation progresses

---

## Tech Stack

| Layer | Technology | Notes |
|-------|------------|-------|
| UI | Streamlit | Single-page, split layout |
| Agent | LangGraph | Two-node graph: chat + tools |
| LLM (primary) | Mistral 7B Instruct v0.3 via Ollama | Local, fully private |
| LLM (optional) | Claude Sonnet/Opus via Anthropic API | Cloud, max capability |
| LLM (any) | Any OpenAI-compatible endpoint | LM Studio, OpenRouter, etc. |
| Memory | SQLite (`june.db`, per MEMORY_DIR) | Single file, fast, inspectable |
| Language | Python 3.9+ | |

Memory lives in a single SQLite file (`june.db`) in the configured `MEMORY_DIR`. No cloud sync. No accounts.

---

## Quick Start

You need [Ollama](https://ollama.com) installed and running.

```bash
ollama pull mistral
```

Then:

```bash
git clone https://github.com/IrgenSlj/JuneAI.git
cd JuneAI/JuneAI-app
make bootstrap
make run
```

Open `http://localhost:8501`.

On first run, June will introduce herself and begin learning about you through conversation.

To verify your environment before running:

```bash
make check-ollama    # confirms Ollama is running and model is available
make verify-env      # confirms Python environment is correct
make smoke           # confirms the Streamlit app serves HTTP 200
```

### Docker

```bash
cp .env.example .env   # fill in MODEL_PRESET and any API keys
make docker-build
make docker-up
```

Open `http://localhost:8501`. Memory persists in a named Docker volume (`june_memory`).

### Export your memory

```bash
make export-memory USER_ID=admin
# writes june_export_admin_<timestamp>.json
```

---

## Runtime Profiles

JuneAI supports three named profiles and any custom OpenAI-compatible endpoint.

| Profile | Model | Where inference runs | Best for |
|---------|-------|----------------------|----------|
| `local_mistral_7b` | mistral:7b-instruct-v0.3 | Your machine | Default local runtime, strong tool reliability |
| `local_mistral_3b` | mistral | Your machine | Low-resource machines |
| `local_mistral_8b` | mistral-nemo | Your machine | Better reasoning than 3B, still local |
| `local_gemma_4` | gemma4:e4b | Your machine | Multimodal assistant use (requires Ollama >= 0.6) |
| `claude_high` | claude-sonnet-4-6 | Anthropic API | Maximum reasoning quality |

Set in `.env`:

```env
MODEL_PRESET=local_mistral_7b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
```

For Claude:

```env
MODEL_PRESET=claude_high
ANTHROPIC_API_KEY=your_key_here
```

The active profile is shown as a badge in the UI. You can switch profiles mid-session
without restarting.

---

## Memory Chapters

June organises everything you share into chapters. Each chapter is a domain of your life.

| Chapter | What June stores |
|---------|-----------------|
| Calendar | Events, appointments, reminders, trips, birthdays |
| Goals | Active goals with deadlines and status |
| Open Loops | Unresolved threads and follow-ups |
| Habits | Daily habits, streaks, and completions |
| Body | Weight, sleep, energy, HRV, steps — logged daily |
| Workouts | Sessions, exercises, duration, energy before/after |
| Gym Plan | Active training programme |
| Nutrition | Meals, calories, protein |
| Food Program | Nutrition approach and daily structure |
| Mood | Daily mood and note |
| Journal | Free-form entries |
| Relationships | People, context, communication notes, birthdays |
| Preferences | How you like things |
| Favourites | Books, films, places, recommendations |

Chapters are filled through conversation — no forms, no setup wizard.

---

## Project Structure

```
JuneAI-app/
|-- app.py                     # Streamlit entry point
|-- src/
|   |-- agent/
|   |   |-- graph.py           # LangGraph agent (chat + tools nodes)
|   |   |-- tools.py           # Tools the LLM can call
|   |   |-- memory.py          # SQLite memory layer (june.db)
|   |   |-- skills.py          # Role-based system prompts
|   |   |-- config.py          # Runtime profile resolution
|   |   |-- patterns.py        # Proactive pattern detection
|   |   |-- context_intelligence.py  # Recovery readiness + commitment summaries
|   |   |-- telemetry.py       # Tool call event logging
|   |   `-- runtime_privacy.py # Privacy status and preset switching
|   `-- agent_ui/
|       |-- panels.py          # Right-rail panel builders
|       |-- rendering.py       # HTML rendering helpers
|       |-- chapters.py        # Chapter metadata
|       |-- chapter_surface.py # Chapter status and freshness
|       |-- onboarding.py      # First-run flow
|       |-- transcript.py      # Chat message rendering
|       `-- state.py           # Session state management
|-- tests/
|   |-- unit_tests/
|   `-- integration_tests/
|-- scripts/
|   |-- bootstrap_env.py
|   |-- verify_env.py
|   |-- check_ollama.py
|   `-- export_memory.py
|-- Dockerfile
|-- docker-compose.yml
|-- docs/
|   |-- PLAN.md                # Development roadmap
|   `-- architecture.html      # System diagrams
|-- pyproject.toml
`-- Makefile
```

---

## Development

```bash
make bootstrap          # create .venv and install deps
make check-ollama       # verify Ollama is running and model available
make run                # start the app
make test               # run unit tests
make integration_tests  # run integration tests
make lint               # run ruff and mypy
make smoke              # HTTP smoke test
make export-memory      # export memory to JSON (USER_ID=admin)
make docker-build       # build the Docker image
make docker-up          # start with docker compose (detached)
make docker-down        # stop the docker compose stack
```

---

## How It Works

On each message turn:

1. The user sends a message in the chat.
2. June selects an internal skill (assistant, planner, wellness, curator) based on context.
3. Memory context is injected into the system prompt: today's summary, recovery readiness,
   active commitments, and any active pattern observations.
4. The LangGraph agent invokes the configured model.
5. The model decides to call tools, reply directly, or both.
6. Tools read and write the local memory store in `.june_memory/`.
7. The right-rail dashboard updates to reflect the new memory state.
8. June's reply streams back to the user.

The chat history is capped at 20 recent messages in full fidelity. Older messages are
summarized and injected as compressed context so nothing is truly forgotten.

---

## Privacy

The default configuration runs entirely on your machine. No messages, no memory, no
model calls leave your device when using a local Ollama model.

When using Claude (`claude_high`), messages are sent to Anthropic's API. The privacy
badge in the UI shows a green dot for local and an amber dot for API-assisted inference.

Memory is stored in a single SQLite file (`june.db`) in `MEMORY_DIR` (default: `~/.june_memory`).
It is fully readable with any SQLite tool. You own your data. Use `make export-memory` to
dump everything to a portable JSON file.

---

## Design Principles

The interface should feel like a quiet command center, not a productivity app full of noise.

- Minimal and spacious with strong readability
- Chat remains central; surrounding memory surfaces stay visible without competing
- Every panel earns its place through utility
- No promotional copy, overdesigned gradients, or decorative elements inside the app shell

The assistant should feel calm, direct, competent, private, and structured.

---

## Assistant Behavior

June behaves like a high-agency personal assistant with memory.

- Listens for explicit and implicit structure in conversation
- Saves useful information proactively when confidence is high
- Prefers concise, actionable responses
- Uses tools when they improve continuity, recall, or execution
- Does not spam memory with weak inferences
- Does not default to relationship advice or therapist behavior unless the conversation
  is actually about those things

---

## Architecture Notes

The current architecture is intentionally simple:

1. Streamlit gathers user input.
2. The app sends state to the LangGraph agent.
3. The model decides whether to answer directly or use tools.
4. Tools update memory and optionally update the workspace UI state.
5. Streamed events are shown in the UI.

This simplicity is a feature. Avoid overengineering.

When adding features:

- They must map to a real persistent surface or meaningful assistant behavior
- They should strengthen the "personal operating layer" concept
- They should not turn the app into a generic dashboard

Tool verification requirements:

- Every tool-using turn emits structured diagnostics for requested, succeeded, and failed calls
- UI logs make tool behavior inspectable without reading raw LangGraph traces
- Tests must cover tool success accounting without requiring a live model endpoint

---

## North Star

JuneAI should feel like a private, local, intelligent life console:

- one place to talk
- one place to remember
- one place to plan
- one place to stay organized and healthy

If a future change does not strengthen that direction, it should probably not be added.

---

## License

MIT. See [LICENSE](JuneAI-app/LICENSE).
