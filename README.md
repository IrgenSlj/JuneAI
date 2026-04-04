# JuneAI

A local-first personal AI that feels like a friend who actually remembers you.

June is not a generic chatbot. She is a private life console: one place to talk, remember,
plan, and keep continuity across your daily life. Every conversation makes her more useful.
Everything stays on your machine.

---

## The Idea

Most AI tools answer questions. June builds a picture of you over time.

She tracks your goals, your gym programme, your calendar, your moods, your sleep, the
people you care about, and your open threads. She cross-references all of it and uses it
to give you context-aware responses that feel less like a search engine and more like
talking to someone who knows you well.

The more you tell June, the more useful she becomes. That is the entire product.

---

## What June Does

- Captures appointments, reminders, birthdays, trips, and commitments from natural conversation
- Tracks goals and open follow-ups, flags when something has gone stale
- Logs workouts, body metrics, nutrition, water, and habits
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
| Memory | Local JSON files (per user, per chapter) | Simple, inspectable, migration path planned |
| Language | Python 3.9+ | |

Memory lives in `.june_memory/` as local JSON files. No cloud sync. No accounts.

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

---

## Runtime Profiles

JuneAI supports three named profiles and any custom OpenAI-compatible endpoint.

| Profile | Model | Where inference runs | Best for |
|---------|-------|----------------------|----------|
| `local_gemma_4` | gemma4 | Your machine | Default local runtime, native tools, long-context personal assistant use |
| `local_mistral_7b` | mistral:7b-instruct-v0.3 | Your machine | Alternative local runtime, strong tool reliability |
| `local_mistral_3b` | mistral:3b | Your machine | Low-resource machines |
| `claude_high` | claude-sonnet-4-6 | Anthropic API | Maximum reasoning quality |

Set in `.env`:

```env
MODEL_PRESET=local_gemma_4
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LOCAL_GEMMA_MODEL_NAME=gemma4
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
| Water | Daily glass count |
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
|   |   |-- tools.py           # 48 tools the LLM can call
|   |   |-- memory.py          # Local JSON memory layer
|   |   |-- skills.py          # Role-based system prompts
|   |   |-- config.py          # Runtime profile resolution
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
|   |-- check_ollama.py        # Ollama health check
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
```

---

## How It Works

On each message turn:

1. The user sends a message in the chat.
2. June selects an internal skill (assistant, planner, wellness, curator) based on context.
3. Memory context is injected into the system prompt: today's summary, recovery readiness,
   active commitments, and any active pattern observations.
4. The LangGraph agent invokes Mistral (or the configured model).
5. Mistral decides to call tools, reply directly, or both.
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

Memory files are stored at `.june_memory/june.db` and are fully readable with any
SQLite viewer. You own your data.

---

## License

MIT. See [LICENSE](JuneAI-app/LICENSE).
