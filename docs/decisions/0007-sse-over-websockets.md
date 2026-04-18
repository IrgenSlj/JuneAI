# ADR 0007: SSE over WebSockets for Chat Streaming

**Status:** Accepted
**Date:** 2026-04-18

## Context

The chat loop is the central interaction in June. When the user sends a message, the server needs to push tokens back as the model generates them. Two practical choices exist at the HTTP layer: Server-Sent Events (SSE) and WebSockets. Both are supported natively in every browser and reach mobile shells through the same web platform.

The choice shapes the API surface, the shell integrations (browser PWA, Tauri, Capacitor), and the code JavaScript clients have to carry. Picking the wrong one now means rewriting the client transport when we ship the desktop and mobile apps.

Every other AI chat product we benchmark (OpenAI, Anthropic, Mistral, Groq, Together) ships SSE as the streaming interface. That is a strong prior: if WebSockets were clearly better, they would have converged there.

## Decision

June's `/chat` endpoint streams with Server-Sent Events over HTTP/1.1 (keep-alive) and HTTP/2 when available. The response content type is `text/event-stream`. Each event has a `type` field (`token`, `tool_call`, `tool_result`, `done`, `error`) so clients can drive the UI from one unified event channel.

Uploads and long-running one-shot requests keep using regular HTTP `POST`; SSE is only for responses that genuinely stream.

## Consequences

**Positive:**

- SSE is one-way (server → client), which matches how chat streaming actually works. The client POSTs a message, then passively consumes the stream. No duplex coordination is needed.
- Works over plain HTTP through every proxy, CDN, and corporate firewall. WebSockets are frequently blocked or down-graded by middleboxes.
- Auto-reconnect and `Last-Event-ID` replay are part of the spec. A client that drops and reconnects can resume without server-side reconnection logic.
- Pydantic + FastAPI emit SSE with `StreamingResponse`; the server is about 30 lines of code.
- Tauri's HTTP client and Capacitor's fetch both support SSE through `EventSource` or a streaming fetch polyfill. No native plugin work needed for Week 6 or Week 7.
- The browser API (`EventSource` or `fetch` + `ReadableStream`) is stable across every modern engine.

**Negative:**

- Clients cannot push additional frames mid-stream. Acceptable: June's interaction is turn-based; the next user message is a new request.
- SSE has a lower per-connection limit than WebSockets in some browsers (6 concurrent streams on HTTP/1.1). Mitigated by serving on HTTP/2 in production, which has no such limit.

## Alternatives Considered

**WebSockets.** Richer (duplex, binary) but overspecified for our use. Middlebox compatibility is worse. The async back-pressure model in Python servers for WebSockets is more fragile than SSE. Rejected.

**Long polling.** Works but the UX is worse (visible pauses, wasted reconnects). Rejected.

**Chunked JSON over plain HTTP.** Technically works with `fetch` + `ReadableStream` but reinvents what SSE already standardized (event framing, reconnection, `Last-Event-ID`). Rejected.

**gRPC-Web streaming.** Powerful but adds a heavy dependency (protoc, codegen per shell). Rejected for now; reconsider if skills need bidirectional streaming later.
