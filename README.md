# JuneAI

> An offline-first personal assistant with memory, routines, reminders, and chaptered life context.

JuneAI is a local-first assistant designed to help a user stay organized, active, healthy, and consistent over time. It is not a generic chatbot and it is no longer centered on relationship coaching. The product direction is a private life console: one place to talk, remember, plan, and keep continuity across daily life.

June is built with Streamlit, LangGraph, and local JSON memory. It now supports two execution modes:

- Local-first mode with a small Mistral-class model behind an OpenAI-compatible endpoint
- High-performance mode with Claude for stronger reasoning and higher tool reliability

## What June Does

- Captures appointments, reminders, birthdays, trips, and other agenda items from conversation
- Stores goals, open loops, and pinned workspace notes
- Saves gym schedules and food programs
- Tracks preferences, favorites, and recommendations
- Remembers relationship and family context when relevant
- Shows chapter-based stored memory for categories like `Calendar`, `Plans`, `Birthdays`, `Family`, and `Dating/Love`
- Starts the day proactively with a local daily check-in and upcoming reminders
- Stays aware of local date, time, weekday, part of day, and day of year

## Current Interface

The current UI is intentionally minimal:

- Left side: conversation
- Right side: reminders, chapter buttons, stored chapter content, workspace, and tool logs

Chapter buttons currently include:

- `Calendar`
- `Gym Schedule`
- `Food Schedule`
- `Trips`
- `Plans`
- `Dating/Love`
- `Family`
- `Birthdays`

Each chapter opens inline and shows what June has actually stored from previous chats.

## How It Works

On each message:

1. The user sends a prompt in Streamlit.
2. June auto-selects an internal skill route from the prompt.
3. A LangGraph agent decides whether to answer directly or use tools.
4. Tools update local memory and, when useful, the workspace panel.
5. The UI shows streamed responses plus tool activity logs.

Memory is persisted as plain JSON files on disk. There is no database and no required cloud sync layer.

## Core Memory Types

June currently stores:

- Chat history
- Mood logs
- Journal entries
- Relationship profiles
- Goals
- Open loops
- Preferences
- Calendar items
- Favorites
- Gym plans
- Food programs
- App state for daily check-ins and rotating quote timing

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI | [Streamlit](https://streamlit.io) |
| Agent | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM Client | [LangChain](https://github.com/langchain-ai/langchain) + `langchain-openai` |
| Model Backend | OpenAI-compatible APIs or local Ollama |
| Storage | Local JSON files |
| Language | Python |

## Run Locally

```bash
git clone https://github.com/IrgenSlj/JuneAI.git
cd JuneAI/JuneAI-app
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Set your model configuration:

```env
MODEL_PRESET=local_mistral_8b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LOCAL_SMALL_MODEL_NAME=mistral
LOCAL_LARGE_MODEL_NAME=mistral-nemo
```

For Claude:

```env
MODEL_PRESET=claude_high
ANTHROPIC_API_KEY=your_key_here
CLAUDE_MODEL_NAME=claude-3-5-sonnet-latest
```

Then run:

```bash
PYTHONPATH=src .venv/bin/streamlit run app.py --server.headless true --server.port 8501
```

Open:

- `http://localhost:8501`

## Project Structure

```text
JuneAI-app/
|-- app.py
|-- src/agent/
|   |-- graph.py
|   |-- tools.py
|   |-- memory.py
|   |-- skills.py
|   `-- config.py
|-- tests/
|   |-- unit_tests/
|   `-- integration_tests/
|-- requirements.txt
|-- pyproject.toml
`-- langgraph.json
```

## Runtime Profiles

- `local_mistral_3b`
  - Small local profile for fast offline use
  - Tuned for short turns and conservative tool calling
- `local_mistral_8b`
  - Default local profile
  - Better tool-call reliability and still local/offline-friendly
- `claude_high`
  - Cloud profile for stronger reasoning and best overall assistant quality

The active profile is shown in the sidebar and logged for each turn.

## Tool Reliability

June now tracks tool-call diagnostics in the graph state:

- Requested tool calls
- Successful tool executions
- Failed tool executions
- Per-tool preview logs in the UI activity panel

This makes it possible to validate whether a local model is actually calling tools correctly instead of only producing plausible text.

## Development

```bash
.venv/bin/python -m pytest tests/unit_tests -q
.venv/bin/python -m pytest tests/integration_tests/test_graph.py -q
```

## Notes

- June is designed to work offline-friendly, but live model responses still depend on the configured LLM endpoint.
- Memory files are local and human-inspectable.
- The UI includes tool logs, tool success counters, and capture-health counts so you can verify whether the model is actually storing information.

## License

MIT. See [LICENSE](JuneAI-app/LICENSE).
