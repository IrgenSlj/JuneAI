# ADR 0010: Agentic core — tasks, OAuth skills, browser/computer use, and MCP universal compatibility

Status: Proposed
Date: 2026-05-18

## Context

June 1.0 is a chat-with-memory product. The conversation turn is the only unit of work. Skills exist but are scoped to inputs the chat agent can already articulate in one turn: a calendar lookup, a journal log, a research summary. The user gets a smart pen-pal that remembers them.

That is not enough to be a useful personal assistant for everyday people. The Apple-Intelligence-shaped audience June targets (Identity A in the May 2026 strategic review) does not want a pen-pal. They want June to **do things for them** across their real apps, files, and services:

- "Find the flight confirmation from United and add it to my calendar."
- "Watch this product page and tell me when it drops below 80 dollars."
- "Summarize the unread messages from my partner today."
- "Find every PDF I saved this week about taxes and put them in one folder."
- "Draft a reply to Sara that sounds like me, using context from the last three messages."

None of these are chat turns. Each is a *task*: a stateful, multi-step, potentially long-running unit of work that calls multiple skills, reads and writes files or services, often runs partially in the background, and reports back when it is done. The chat turn cannot model this.

At the same time, the agentic capability frontier in 2026 is concentrated in three primitives that do not exist in June 1.0:

- **OAuth-based service skills** that touch the user's real Gmail, Calendar, Drive, Notion, Spotify, GitHub, etc. with real tokens stored in the OS credential store.
- **Browser automation** via headless Chromium driven by Playwright, for the long tail of services with no API and for any flow that requires being logged in as a human.
- **Computer use** as the universal escape hatch — vision-model-driven screen interaction when nothing else works.

And the ecosystem standard for connecting agents to capabilities — **Model Context Protocol (MCP)** — is now adopted by Anthropic, OpenAI, Cursor, Claude Desktop, and the long tail. Thousands of MCP servers are shipping (Linear, GitHub, Notion, Sentry, Slack, Stripe, Postgres, etc.). June already builds skills as MCP servers ([ADR 0005](0005-skills-as-mcp.md)) but does not yet let users install third-party MCP servers as skills. That is leaving the ecosystem on the table.

## Decision

June adopts an **agentic core**: four additions that, together, turn the product from "chat with memory" into "personal agent with memory."

### 1. The `task` primitive

A `task` is a first-class, persistable, observable unit of work, separate from a chat turn. It has:

- a goal (free-form natural language)
- a plan (LLM-produced, editable, JSON)
- a status (`planning`, `running`, `paused`, `awaiting_user`, `completed`, `failed`)
- a step trace (every tool call, every model call, with model provenance per [ADR 0009](0009-private-by-default-and-model-routing.md))
- an owner skill (optional — some tasks are skill-rooted, others are general)
- a schedule (optional — for recurring or future tasks)

Tasks are stored in a new `tasks` SQLite table inside the user data directory. The API surfaces them at `POST /tasks`, `GET /tasks`, `GET /tasks/{id}/events` (SSE), `PATCH /tasks/{id}` (pause / resume / cancel), and `DELETE /tasks/{id}`. The UI gains a new `/tasks` route showing active tasks with live step traces and recent completed tasks.

Chat turns can spawn tasks (`@create_task drafted; want me to run it?`). The user can also create tasks directly from the `/tasks` UI or via natural language at the chat composer (`/task`).

### 2. OAuth-backed service skills

Three skills move from stub to real in Sprint 1: **gmail**, **gcal**, and a renamed **drive** (combining filesystem + cloud drives behind one interface).

OAuth flow:

- On desktop (Tauri), a one-shot loopback HTTP server on `127.0.0.1:<random>` receives the redirect; the OAuth provider opens in the system browser.
- On web (PWA), a popup window handles the redirect to a same-origin callback page that posts the code back.
- Tokens are stored in the OS credential store: Keychain (macOS), Credential Manager (Windows), libsecret (Linux). On web, encrypted in IndexedDB with a key derived from a passphrase the user sets once.
- Refresh handled by the skill subprocess; the brain never sees tokens.

Each OAuth skill declares its scopes in the manifest, and the user sees them at install / first-call time. Sensitive operations (`gmail.send`, `gcal.delete`) always require an in-UI confirmation step, even inside a long-running task.

### 3. Browser automation and computer use

A new **browser** skill ships in Sprint 1, built on Playwright + headless Chromium. Tools: `navigate`, `click`, `fill`, `extract_text`, `extract_table`, `screenshot`, `wait_for`. The browser runs in a separate profile under the user data directory; cookies and local storage persist between turns so the agent can stay logged in to sites the user has authenticated.

A **compute** skill ships later (Sprint 2 or 3) for the long tail of "no API, no browser, only a desktop app": vision-model-driven screen interaction via Gemini Vision. Marked `cloud-required` in policy terms.

### 4. MCP universal compatibility

June's skill supervisor already speaks MCP ([ADR 0005](0005-skills-as-mcp.md)). What changes:

- A **registry** of installable MCP servers ships in-app. The first registry is a JSON file checked into the repo (`packages/brain/src/june_brain/skills/registry.json`); later it becomes a hosted index.
- Users can install any third-party MCP server in one click; the supervisor runs it the same way it runs first-party skills.
- Skills installed this way carry an `unsigned` badge and the user sees a one-time "this runs with your user privileges" warning before first use.
- Conversely, June's first-party skills are runnable standalone via `python -m june_skill_<name>` (already true) and become callable from Claude Desktop, Cursor, and anything else that speaks MCP, with no June-specific glue.

This positions June as the *personal-MCP home base* — the user installs June and the ecosystem of MCP servers becomes their skill library at no additional engineering cost to us.

## Alternatives considered

- **Stay chat-only and lean on extraction quality.** Rejected. Chat is not the right surface for "watch this URL for a week" or "find every PDF about X." The user has to keep the chat open, and the brain has to thread the work through the conversation. The conversation is the wrong place for that work.
- **Build a custom tool protocol instead of MCP.** Rejected. The ecosystem chose MCP. The cost of being incompatible is being incompatible.
- **Skip OAuth, ship only file-based skills first.** Considered, but rejected: the demo that sells the product is "June drafted a reply to Sara." That requires Gmail. The product needs the wow moment, and the wow moment needs OAuth.
- **Build computer use first, skip OAuth.** Rejected. Computer use is slow, brittle, and cloud-required; OAuth gets us to the same outcomes faster, more reliably, and (often) locally. Computer use is the escape hatch, not the front door.

## Consequences

**Positive:**

- The product can do things people actually want done. The "smart pen-pal" ceiling lifts.
- The MCP ecosystem becomes June's skill catalog without us building it.
- The Tasks UI gives users a coherent place to see what June is doing in the background, which is critical for trust as background work scales.
- The OAuth skills create the first real reason to install the desktop shell: the loopback OAuth pattern works much better outside the browser sandbox.

**Negative:**

- New `tasks` SQLite table, new API endpoints, new UI route. Real surface area.
- OAuth introduces credential handling, refresh logic, scope management, and revocation UX. Non-trivial to do safely.
- Browser automation adds a Chromium runtime to the install footprint (~150 MB). Worth it; users get a real agent.
- The skills supervisor must now handle skills with very different lifecycles (long-running browser, short-lived file reads, OAuth-token-holding service skills).

**Forced follow-ups:**

- [ADR 0005](0005-skills-as-mcp.md) needs an addendum on third-party / unsigned skill handling and a security model. Tracked.
- `docs/product/overview.md` and `docs/product/roadmap.md` need to be rewritten to reflect the agentic core. Done in the same PR as this ADR via the [agentic pivot plan](../product/agentic-pivot-plan.md).
- Skill manifest schema needs a `model_policy` field (from [ADR 0009](0009-private-by-default-and-model-routing.md)) and an `oauth` block (scopes, redirect handler, token storage hint).
