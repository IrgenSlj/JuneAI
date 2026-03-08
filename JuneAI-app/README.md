# JuneAI 🌸

Your AI companion for love, life & growth — built with LangGraph, Google Gemini, and Streamlit.

## Features

- **Friend & Therapist** — emotional support, active listening, and personal growth
- **Dating Coach** — compatibility analysis, conversation starters, and authentic profile building
- **Mood Tracker** — log and understand your emotional patterns over time

All conversations and mood logs are stored locally per user.

## Getting Started

### 1. Clone & install dependencies

```bash
git clone https://github.com/your-username/JuneAI-app.git
cd JuneAI-app
pip install -r requirements.txt
```

### 2. Set up your API key

```bash
cp .env.example .env
```

Edit `.env` and add your [Gemini API key](https://aistudio.google.com):

```env
GEMINI_API_KEY=your_key_here
MODEL_NAME=gemini-2.0-flash
MEMORY_DIR=.june_memory
```

### 3. Run the app

```bash
streamlit run app.py
```

## Project Structure

```
JuneAI-app/
├── app.py                  # Streamlit frontend
├── src/agent/
│   ├── graph.py            # LangGraph ReAct agent
│   ├── config.py           # Environment config
│   ├── prompts.py          # June's personality & system prompt
│   ├── memory.py           # Local JSON memory (chat, moods, journal)
│   └── tools.py            # Agent tools (mood logging, compatibility, etc.)
├── tests/
│   ├── unit_tests/
│   └── integration_tests/
└── requirements.txt
```

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent graph & ReAct loop
- [Google Gemini](https://aistudio.google.com) — language model
- [Streamlit](https://streamlit.io) — web frontend
- Python 3.9+
