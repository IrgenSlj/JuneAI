# JuneAI

> Your AI companion for love, life & growth.

JuneAI is a conversational AI companion that feels like talking to a brilliant, emotionally attuned friend. She listens without judgment, remembers what matters to you, helps you understand your emotions, and coaches you through your love life — all in one place.

Built on Google Gemini 2.0 Flash with a LangGraph-powered agent, June doesn't just chat — she thinks, uses tools, and builds a personal memory of who you are over time.

---

## What June Can Do

### Friend & Therapist Mode
June listens deeply before offering advice. She validates feelings, asks the right follow-up questions, offers gentle perspective shifts, and never rushes to "fix" things. When something meaningful comes up, she saves it to your journal automatically.

### Dating Coach Mode
June helps you figure out what you actually want in a partner. She analyzes compatibility between people, builds authentic dating profiles, and generates specific, genuine conversation starters — never generic lines.

### Mood Tracker Mode
June logs your emotional state as you talk and builds a timeline of how you've been feeling. Ask her "how have I been lately?" and she'll reflect your patterns back to you, helping you spot what lifts you up or drags you down.

---

## Capabilities

| Capability | Description |
|------------|-------------|
| **Persistent Memory** | Every conversation, mood log, and journal entry is saved locally per user |
| **Mood Logging** | June automatically logs moods as you express them during chat |
| **Mood History** | Review your emotional patterns over time, surfaced in the sidebar |
| **Journal Entries** | June saves meaningful exchanges as personal journal notes |
| **Compatibility Analysis** | Structured analysis of two people's personalities, values, and communication styles |
| **Conversation Starters** | Context-aware, specific openers for dating or friendship — no clichés |
| **Multi-user** | Each user gets their own isolated memory by name |

---

## How It Works

June is built as a [LangGraph](https://github.com/langchain-ai/langgraph) ReAct agent. On every message, Gemini decides whether to respond directly or call one of June's tools first (log a mood, retrieve history, save a journal entry, run a compatibility analysis). This makes her behavior feel natural rather than mechanical.

```
User Message
     │
     ▼
LangGraph ReAct Agent (Gemini 2.0 Flash)
     │
     ├── Tool: log_mood
     ├── Tool: get_mood_history
     ├── Tool: save_journal_entry
     ├── Tool: analyze_compatibility
     └── Tool: generate_conversation_starters
     │
     ▼
Streamlit UI ◄── Response + updated memory
```

Memory is stored as plain JSON files on disk — one file per user per data type. No database, no embeddings, no cloud sync. Transparent and easy to inspect.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI | [Streamlit](https://streamlit.io) |
| LLM | Google Gemini 2.0 Flash (`gemini-2.0-flash`) |
| Agent | [LangGraph](https://github.com/langchain-ai/langgraph) ReAct |
| LLM Integration | [LangChain](https://github.com/langchain-ai/langchain) + `langchain-google-genai` |
| Memory | Local JSON files (per-user) |
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

**3. Configure your API key**
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

**4. Run**
```bash
streamlit run app.py
```

June will be available at `http://localhost:8501`.

---

## Project Structure

```
JuneAI-app/
├── app.py                    # Streamlit UI — entry point
├── src/agent/
│   ├── graph.py              # LangGraph agent definition
│   ├── tools.py              # June's callable tools
│   ├── memory.py             # Local JSON memory system
│   ├── prompts.py            # June's personality & system prompt
│   └── config.py             # Env config loader
├── tests/
│   ├── unit_tests/           # Agent compilation tests
│   └── integration_tests/    # Live agent response tests
├── .env.example              # Required environment variables
├── requirements.txt          # Runtime dependencies
├── pyproject.toml            # Project metadata & dev tooling
├── langgraph.json            # LangGraph CLI config
└── Makefile                  # Dev commands (test, lint, format)
```

---

## Memory & Privacy

All data lives on your machine:

- `memory/{user}_chat.json` — last 50 messages per user
- `memory/{user}_moods.json` — full mood log with timestamps
- `memory/{user}_journal.json` — saved journal entries

Nothing is sent to any external service beyond the Gemini API for inference. You can inspect, edit, or delete any file directly.

---

## Development

```bash
make test          # Run unit tests
make integration   # Run integration tests (requires GEMINI_API_KEY)
make lint          # Ruff lint
make format        # Ruff format
```

---

## License

MIT — see [LICENSE](JuneAI-app/LICENSE).
