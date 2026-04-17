# ADR 0003: Retire Streamlit, Adopt SvelteKit Frontend over FastAPI

**Status:** Accepted
**Date:** 2026-04-17

## Context

Streamlit got June to a working demo quickly. Five sessions of product development and a sixth of code hygiene have taken the app to `app.py` at 3,726 lines plus 13 helper modules in `agent_ui/` that import Streamlit directly. The current roadmap schedules another six sessions of Streamlit cleanup.

Streamlit is a data-science prototyping framework. It cannot produce a native desktop app. It cannot produce a mobile app. Its browser app fights the developer on every custom interaction (hiding the native toolbar, forcing a layout, embedding CSS). Every session spent polishing Streamlit is a session not spent moving toward the v2 vision of browser + desktop + mobile parity.

## Decision

Streamlit is retired. The frontend is rebuilt as a SvelteKit application that talks to a FastAPI backend over HTTP and Server-Sent Events.

- **Backend:** FastAPI in `packages/api/`. Routes for chat (streaming), memory, skills, and system status. Pydantic schemas generate TypeScript types for the frontend.
- **Frontend:** SvelteKit in `apps/web/`. Shared components live in `packages/ui/`. The same codebase ships as the browser app, the desktop app (wrapped in Tauri), and the mobile app (wrapped in Capacitor).
- **Transport:** SSE for streaming token delivery. HTTP for everything else. WebSockets are not adopted at this stage; SSE is simpler and covers the use case.

The v1 `app.py` and `agent_ui/` modules are deleted. The brain modules under `src/agent/` are preserved and migrated to `packages/brain/`. Approximately 7,000 lines of UI code are removed. Approximately 5,000 lines of brain/memory/tools logic are preserved and restructured.

## Consequences

**Positive:**

- One UI codebase supports all three surfaces.
- The API boundary enables third-party UIs, CLIs, and integrations.
- SvelteKit produces small, fast, PWA-ready bundles. The browser app becomes installable.
- The backend becomes testable in isolation with standard HTTP tools.
- SSE streaming is natively supported by browsers, mobile webviews, and Tauri's webview — zero shim code.

**Negative:**

- A full UI rewrite is a multi-week effort. Non-trivial schedule commitment.
- During the migration, the v1 Streamlit app is live and v2 is empty. Mitigated by preserving v1 on a `legacy/streamlit` branch and keeping the brain package usable independently.
- The team (currently one person) needs to work in both Python and TypeScript. Acceptable; the alternative is worse.

## Alternatives Considered

**Continue Streamlit sessions.** Rejected because each session compounds sunk cost without moving toward the multi-platform vision.

**React + Next.js instead of SvelteKit.** Rejected for bundle size and developer ergonomics. Svelte's compiled output is smaller, the reactivity model fits a live streaming chat interface well, and SvelteKit's file-based routing is sufficient. Also — no JSX, which is preferred.

**Electron for desktop.** Rejected in ADR 0006 (see Tauri).

**Keep Streamlit for browser, build native apps separately.** Rejected because it creates two UI codebases to maintain forever.

**Gradio.** Rejected for the same reasons as Streamlit plus worse extensibility.

**HTMX + server-rendered HTML.** Considered. Appealing for simplicity. Rejected because the desktop and mobile shells expect a thick client that can operate without the API momentarily (offline skeleton UI, local state).
