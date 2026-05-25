# Telegram Integration Plan

## Overview

A bidirectional Telegram bot that lets the user chat with June through Telegram, receive notifications, and quickly capture information on the go.

## Architecture

```
Telegram User
     │
     │  (HTTPS Webhook or Long-Polling)
     ▼
┌─────────────────────┐
│  Telegram Bot        │
│  (python-telegram-bot│
│   or custom MCP      │
│   skill process)     │
└────────┬────────────┘
         │
         │  (MCP over stdio — daemon mode)
         ▼
┌─────────────────────┐
│  SkillSupervisor     │  ← enhanced with daemon support
│  (event queue)       │
└────────┬────────────┘
         │
         │  (event table polled by scheduler)
         ▼
┌─────────────────────┐
│  Scheduler Service   │  ← picks up inbound messages
│  → invokes agent     │
└─────────────────────┘
         │
         │  (agent response → notification bus)
         ▼
┌─────────────────────┐
│  Notification Bus    │  ← routes response to Telegram
└─────────────────────┘
```

## Directory Structure

```
skills/telegram/
├── pyproject.toml
├── src/june_skill_telegram/
│   ├── __init__.py
│   ├── __main__.py        # Entry point: MCP server loop
│   ├── server.py          # MCP stdio server
│   ├── bot.py             # python-telegram-bot interface (long-polling)
│   ├── handlers.py        # Message handlers (text, commands, buttons)
│   └── router.py          # Route incoming messages to MCP -> supervisor -> agent
├── tests/
└── README.md
```

## Skill Manifest (skills.toml)

```toml
[skills.telegram]
enabled = false
daemon = true
command = "uv"
args = ["run", "--package", "june-skill-telegram", "python", "-m", "june_skill_telegram"]
response_timeout_seconds = 60
env = {
    TELEGRAM_BOT_TOKEN = "",
    JUNE_API_URL = "http://127.0.0.1:8000",
    JUNE_API_KEY = "",
}
```

## Behavior

### Inbound (Telegram → June)

1. User sends message to Telegram bot
2. Bot process receives it via long-polling or webhook
3. Bot pushes event to supervisor via stdout JSON-RPC notification:
   ```json
   {"jsonrpc": "2.0", "method": "notification/message", "params": {
     "chat_id": 12345, "from": {"id": 678, "first_name": "User"},
     "text": "what's on my calendar today", "message_id": 99
   }}
   ```
4. Supervisor writes to `skill_inbound_events` table
5. Scheduler (polling every 5-10s) picks up unprocessed events
6. Scheduler invokes agent with message text, user_id mapped from Telegram chat_id
7. Agent response goes through Notification Bus
8. Notification Bus routes response back to Telegram via `POST /sendMessage`

### Outbound (June → Telegram)

1. Agent calls `send_notification(channel="telegram", body="...")`
2. Notification Bus dispatches to Telegram channel handler
3. Handler calls Telegram Bot API `sendMessage` with the configured chat_id

### Commands

- `/start` — welcome message, instructions
- `/chat <message>` — send a message to June (or just type naturally)
- `/mood <feel>` — quick mood log
- `/todo <task>` — quick task capture
- `/journal <entry>` — quick journal entry
- `/status` — June's current runtime status
- `/help` — list available commands

## Dependencies

- `python-telegram-bot` — official async Telegram Bot API library (or custom lightweight HTTP client)
- `httpx` — for Telegram API calls (already a dependency of research skill)

## Security

- Chat ID → user_id mapping stored in a new `telegram_links` table:
  ```sql
  CREATE TABLE telegram_links (
      chat_id INTEGER PRIMARY KEY,
      user_id TEXT NOT NULL,
      linked_at TEXT NOT NULL,
      active INTEGER DEFAULT 1
  );
  ```
- Link established once via `/start` + a secret token (user generates from June settings page)
- Bot token stored in `TELEGRAM_BOT_TOKEN` env var, managed through June's secret store

## Implementation Order

### Session T1 — Daemon MCP Support
- Prerequisite: Phase 1 Component 3 (Daemon MCP Skills) must be working
- [ ] Add `daemon` field to `SkillManifestEntry`
- [ ] Extend `SkillSupervisor` to handle daemon subprocesses with event read loop
- [ ] Add `skill_inbound_events` table to schema
- [ ] Integration test: mock daemon skill sends notification, supervisor receives it

### Session T2 — Telegram Skill Server
- [ ] Create `skills/telegram/` package structure
- [ ] Implement MCP stdio server (handshake, tools/list, tools/call)
- [ ] Implement `mcp_start` notification handler (init bot connection)
- [ ] Implement `mcp_stop` notification handler (shutdown bot)
- [ ] Long-polling loop for Telegram updates
- [ ] Push incoming messages as `notification/message` to supervisor

### Session T3 — Telegram Commands + Integration
- [ ] `/start` with pairing flow (generate link token, verify)
- [ ] `/chat` and natural message handling
- [ ] Quick-capture commands (`/mood`, `/todo`, `/journal`)
- [ ] Notification Bus → Telegram channel handler
- [ ] Error handling: offline, no response, long messages

### Session T4 — Polish
- [ ] Markdown formatting in Telegram messages
- [ ] Split long responses into multiple messages
- [ ] Typing indicator while June is "thinking"
- [ ] Settings page: link/unlink Telegram, view status
- [ ] Tests for all message handlers
