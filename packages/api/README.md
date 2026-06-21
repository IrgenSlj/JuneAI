# june-api

FastAPI boundary in front of `june-brain`. One HTTP surface that every shell (web, desktop, mobile) talks to.

## Routes

- `POST /chat` — SSE stream. Events: `token`, `tool_call`, `tool_result`, `recall`, `provenance`, `done`, `error`. The `provenance` frame carries the visible cloud boundary (tiers used, cloud yes/no, model ids, memories recalled, one-line rationale). Runs `MemoryManager.extract` as a background task after the stream closes.
- `GET /memory/{user_id}` — snapshot across SQLite (goals, open loops, calendar), the sqlite-vec index (semantic facts), and the knowledge graph (entities). Each fact carries a stable `ref` so the UI can target deletes.
- `POST /memory/{user_id}/fact` — write a structured, semantic, or graph fact.
- `PATCH /memory/{user_id}/fact/{ref}` — update supported structured facts.
- `DELETE /memory/{user_id}/fact/{ref}` — remove a fact. Ref prefixes: `semantic:<fact_id>`, `node:<node_id>`, `edge:<src>|<dst>|<kind>`.
- `GET /skills` — MCP skill processes and tools currently available to the agent.
- `POST /skills/{key}/toggle` — enable or disable a skill.
- `POST /skills/{key}/tools/{tool}/toggle` — enable or disable one tool inside a skill.
- `GET /system` — runtime and Ollama status.
- `GET /setup/status`, `POST /setup/apply`, `GET /settings`, `POST /settings/forget-key` — first-run setup and non-secret runtime settings.
- `POST /demo/seed` — seed a profile with demo memory.

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
