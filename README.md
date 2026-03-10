# JuneAI

> Your AI companion for love, life & growth.

JuneAI is a conversational AI companion designed to feel like talking to a thoughtful, emotionally attuned friend. She listens without judgment, remembers what matters, helps users understand their emotions, and supports relationship conversations in one place.

June is built as a LangGraph-powered agent with local memory and tool use. The app works with any model provider that exposes an OpenAI-compatible API, and it defaults to a local Ollama setup for simple offline development.

---

## What June Can Do

### Friend & Therapist Mode
June listens first, validates feelings, asks useful follow-up questions, and offers perspective without rushing to solve everything. When something important comes up, she can save it as a journal entry for future reflection.

### Dating Coach Mode
June helps users think clearly about compatibility, attraction, communication style, and what they actually want in a relationship. She can also generate more natural conversation starters based on real context.

### Mood Tracker Mode
June can log moods during conversation, show recent emotional patterns, and connect current feelings to prior journal entries or mood history.

---

## Current Capabilities

| Capability | Description |
|------------|-------------|
| **Persistent Memory** | Saves conversation history, mood logs, and journal entries locally per user |
| **Mood Logging** | Tracks emotions with timestamps and optional notes |
| **Mood History** | Retrieves recent mood patterns for reflection |
| **Journal Entries** | Saves meaningful reflections as personal notes |
| **Compatibility Analysis** | Structures relationship analysis around values, personality, and communication |
| **Conversation Starters** | Generates tailored openers for dating or friendship contexts |
| **Multi-user** | Keeps each user's local memory isolated by name |

---

## How It Works

June is implemented as a [LangGraph](https://github.com/langchain-ai/langgraph) ReAct agent. On each message, the model decides whether to answer directly or call one of June's tools first. That keeps the behavior flexible while preserving a simple architecture.

```text
User Message
  |
  v
LangGraph ReAct Agent
  |
  |- Tool: log_mood
  |- Tool: get_mood_history
  |- Tool: save_journal_entry
  |- Tool: get_journal
  |- Tool: analyze_compatibility
  `- Tool: generate_conversation_starters
  |
  v
Streamlit UI + updated memory
```

Memory is stored as plain JSON files on disk. There is no database or cloud sync layer in the current version.

---

## Model Provider Support

JuneAI currently uses `langchain-openai` with configurable `LLM_BASE_URL`, `LLM_API_KEY`, and `MODEL_NAME` settings. In practice, that means it works with:

- Local Ollama models exposed through its OpenAI-compatible endpoint
- OpenRouter
- OpenAI-compatible gateways
- Any AI provider that offers an OpenAI-style chat completions API

The default local configuration is:

```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
MODEL_NAME=phi3:mini
```

Example cloud configuration using a Llama-family model through OpenRouter:

```env
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=your_api_key_here
MODEL_NAME=meta-llama/llama-3.1-8b-instruct:free
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI | [Streamlit](https://streamlit.io) |
| Agent | [LangGraph](https://github.com/langchain-ai/langgraph) ReAct |
| LLM Client | [LangChain](https://github.com/langchain-ai/langchain) + `langchain-openai` |
| Model Backend | Any OpenAI-compatible provider or local Ollama |
| Memory | Local JSON files |
| Language | Python 3.9+ |

---

## Getting Started

**1. Clone and enter the app directory**

```bash
git clone https://github.com/IrgenSlj/JuneAI.git
cd JuneAI/JuneAI-app
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure your model provider**

```bash
cp .env.example .env
```

Then edit `.env` and set:

- `LLM_BASE_URL`
- `LLM_API_KEY`
- `MODEL_NAME`

You can use a local Ollama model or any hosted provider with an OpenAI-compatible API.

**4. Run**

```bash
streamlit run app.py
```

June will be available at `http://localhost:8501`.

---

## Project Structure

```text
JuneAI-app/
|-- app.py                    # Streamlit UI entry point
|-- src/agent/
|   |-- graph.py              # LangGraph agent definition
|   |-- tools.py              # Tool implementations
|   |-- memory.py             # Local JSON memory system
|   |-- prompts.py            # Personality and mode prompts
|   `-- config.py             # Environment configuration
|-- tests/
|   |-- unit_tests/           # Memory and configuration tests
|   `-- integration_tests/    # Agent invocation tests
|-- .env.example              # Provider configuration examples
|-- requirements.txt          # Runtime dependencies
|-- pyproject.toml            # Project metadata
|-- langgraph.json            # LangGraph config
`-- Makefile                  # Development commands
```

---

## Memory & Privacy

All user data is stored locally in `MEMORY_DIR` as JSON files:

- `{user}_chat.json`
- `{user}_moods.json`
- `{user}_journal.json`

The only external network traffic comes from whichever model provider you configure for inference. If you use a local Ollama model, inference can stay fully local.

---

## Development

```bash
make test          # Run unit tests
make integration   # Run integration tests (requires LLM access)
make lint          # Ruff lint
make format        # Ruff format
```

---

## License

MIT. See [LICENSE](JuneAI-app/LICENSE).
