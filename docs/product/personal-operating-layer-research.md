# Personal Operating Layer Research

This document records the research behind June's next development track. It is
not a feature list. It is the technical selection memo for building a scalable,
local-first personal assistant with a simple UI.

## Product Patterns Worth Copying

| Product | Useful pattern for June | What June should do differently |
| --- | --- | --- |
| ChatGPT memory | Separate saved memories from chat history; expose memory controls. | Keep memory local, source-linked, and exportable by default. |
| Claude Projects | Scope context to a project or life area. | Make rooms lightweight and personal, not enterprise workspaces. |
| Reclaim | Turn tasks and habits into calendar-shaped time. | Keep scheduling suggestions inspectable before committing. |
| Motion | Use priority, deadline, and workload to build plans. | Avoid opaque automation; show the reason for each plan. |
| Granola | Capture messy meetings and turn them into structured notes. | Support pasted notes and voice later without requiring a meeting bot. |
| Limitless | Passive capture plus searchable personal recall. | Start with explicit quick capture and Telegram before hardware or always-on audio. |

## Open Technical Options

| Area | Options researched | Decision |
| --- | --- | --- |
| Agent runtime | LangGraph, Letta, custom loop | Keep LangGraph. Add persistence and interrupts for durable, approvable work. Letta is worth watching, but June already has a working memory and routing layer. |
| Skills | MCP, custom plugins, direct Python tools | Keep MCP. Add explicit permission metadata, per-tool toggles, and approval rules. |
| Durable jobs | SQLite scheduler, Hatchet, Temporal, Prefect | Keep SQLite scheduler for v0.1.1. Borrow durable-job concepts: runs, attempts, retries, events. Revisit Hatchet/Temporal only after real background load appears. |
| Vector memory | Chroma, LanceDB, Qdrant, sqlite-vec | Keep Chroma as a derived local index. Make SQLite the source of truth. Re-evaluate LanceDB if hybrid search and local file-backed portability become painful. |
| Local sync later | ElectricSQL, Automerge, Yjs, Replicache | Defer. Export/import remains the cross-device story until users prove sync pain is larger than privacy and complexity costs. |
| Voice | Web Speech API, whisper.cpp, Vosk, cloud STT | Defer. The likely path is whisper.cpp for desktop local speech-to-text, with web/native fallbacks later. |

## Closed-Source Product Lessons

The strongest closed-source assistants are not winning because they have a
flashier chat UI. They win where they reduce user bookkeeping:

- Capture happens where the user already is.
- Tasks get placed into time.
- Memories are retrievable in the future.
- Meetings, notes, and decisions become structured records.
- The assistant returns at the right moment instead of waiting forever.

June can compete by being more trustworthy:

- local-first by default
- no account required
- visible cloud boundary
- inspectable memory
- reversible local actions
- source-linked decisions

## Technical Standard For The Next Track

Every new feature should use the same lifecycle:

1. **Capture**: accept natural input from chat, quick capture, Telegram, or voice.
2. **Classify**: identify task, event, memory, decision, promise, feeling, idea,
   question, or note.
3. **Propose**: create action intents for writes and external actions.
4. **Approve**: ask for confirmation when risk requires it.
5. **Commit**: update memory, tasks, schedules, notifications, or skills.
6. **Record**: write a durable event.
7. **Return**: surface the result in Daily Home, reminders, reviews, and search.

This lifecycle is the difference between a chatbot that answers and an
assistant that operates.

## Reference Links

- LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph interrupts: https://docs.langchain.com/oss/python/langgraph/interrupts
- Letta stateful agents: https://docs.letta.com/guides/core-concepts/stateful-agents/
- Model Context Protocol: https://modelcontextprotocol.io/docs/getting-started/intro
- MCP security best practices: https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices
- Chroma persistent client: https://cookbook.chromadb.dev/core/clients/
- LanceDB docs: https://docs.lancedb.com/
- Qdrant quickstart: https://qdrant.tech/documentation/quickstart/
- ChatGPT memory: https://help.openai.com/en/articles/8983136-what-is-memory
- Claude Projects: https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects
- Reclaim features: https://help.reclaim.ai/en/articles/6210740-features-in-reclaim
- Motion help: https://www.usemotion.com/help
- Granola 101: https://docs.granola.ai/help-center/getting-started/granola-101
- Limitless: https://www.limitless.ai/new
- OpenAI sensitive conversations note: https://openai.com/index/strengthening-chatgpt-responses-in-sensitive-conversations/
