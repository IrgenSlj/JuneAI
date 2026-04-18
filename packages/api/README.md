# june-api

FastAPI boundary in front of `june-brain`. One HTTP surface that every shell (web, desktop, mobile) talks to.

## Routes

- `POST /chat` — SSE stream. Events: `token`, `tool_call`, `tool_result`, `done`, `error`. Runs `MemoryManager.extract` as a background task after the stream closes.
- `GET /memory/{user_id}` — snapshot across SQLite (goals, open loops, calendar), ChromaDB (semantic facts), and the knowledge graph (entities). Each fact carries a stable `ref` so the UI can target deletes.
- `DELETE /memory/{user_id}/fact/{ref}` — remove a fact. Ref prefixes: `semantic:<fact_id>`, `node:<node_id>`, `edge:<src>|<dst>|<kind>`.
- `GET /skills` — tools currently available to the agent.
- `GET /system` — runtime and Ollama status.

## Running locally

```bash
uvicorn june_api.app:app --reload --port 8000
```

or

```bash
python -m june_api
```

Then:

```bash
curl -N -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"user_id":"me","message":"hello"}'
```

## Why SSE

See [ADR 0007](../../docs/decisions/0007-sse-over-websockets.md).
