# ADR 0005: Skills as Model Context Protocol Servers

**Status:** Accepted
**Date:** 2026-04-17

## Context

The v1 tools layer is a single `tools.py` file with roughly 50 `@tool`-decorated functions covering calendar, health, journaling, water logging, body metrics, preferences, and more. The file is 1,473 lines and growing. All tools share one import graph and are loaded together.

This layout makes four things hard:

- **Isolation.** One broken tool can break the whole import and take the agent offline.
- **Versioning.** Any change to the tool surface requires a full app release.
- **Discoverability.** Users cannot enable or disable individual capabilities.
- **Community contribution.** Adding a tool requires touching the monolith.

Meanwhile, Anthropic released the Model Context Protocol (MCP) as an open standard for exposing tools, resources, and prompts to LLM agents. It defines a JSON-RPC surface that any agent (Claude Desktop, Cursor, continue.dev, custom) can consume. Ecosystem momentum is substantial and growing.

## Decision

Each skill becomes a standalone MCP server. The v1 `tools.py` is decomposed into one folder per skill under `skills/`:

- `skills/calendar/` — events, reminders, birthdays
- `skills/health/` — body metrics, workouts, water, habits
- `skills/research/` — web search, page reading, summarization
- `skills/files/` — filesystem access, PDF reading
- `skills/daily/` — chapters, journaling, moods, goals

Each skill is a standalone Python package with its own `pyproject.toml`, its own tests, and its own release cycle. Each exposes an MCP server binary.

The brain consumes skills via an MCP client in `packages/brain/src/june_brain/skills/loader.py`. The client discovers running skill servers from a manifest, negotiates capabilities, and exposes the union of their tools to LangGraph.

Skill state persistence stays in the shared memory layer described in ADR 0004. Skills call the memory manager through the MCP `resources` surface; they do not open their own SQLite connections.

## Consequences

**Positive:**

- Each skill is independently testable, versionable, and releasable.
- Users can enable, disable, and install skills from a UI registry.
- Community contributors can ship a new skill as a pip package without touching the brain.
- Any MCP-compatible client (Claude Desktop, Cursor, etc.) can use a June skill. Conversely, June can consume any third-party MCP server — web search, filesystem, git, databases — immediately, for free.
- Tool failures are isolated to one server process.

**Negative:**

- IPC overhead per tool call. Mitigated because MCP servers are local processes over stdio or local sockets; overhead is microseconds.
- More moving parts. Mitigated by a supervisor in the brain that manages skill server lifecycles (start, stop, restart on crash).
- MCP is a young protocol (released late 2024). Risk of breaking changes. Mitigated by pinning the protocol version and by the reference implementation being stable.

## Alternatives Considered

**Keep `tools.py` as one file, split only for readability.** Rejected because it does not deliver any of the real benefits above.

**Custom plugin protocol.** Considered. Rejected because MCP already exists, works, has an ecosystem, and is designed by people who had exactly this problem. Reinventing it is pure cost.

**LangChain tool registry with dynamic loading.** Rejected because LangChain's tool surface is tightly coupled to its own agent primitives. MCP is agent-agnostic and future-proofs the skills against LangGraph being replaced.

**OpenAPI-described HTTP tools.** Considered. Rejected because it is heavier (HTTP round-trip, schema overhead, process-per-tool is overkill) and offers no benefit over MCP for local skills.

## Implementation Notes

Shipped in Week 5:

- Five skill packages under `skills/{calendar,health,research,files,daily}/`, each a standalone `pyproject.toml` with a `python -m june_skill_<name>` entry point.
- Stdio MCP server helper: `packages/brain/src/june_brain/skills/server.py` (`MCPStdioServer`) — JSON-RPC 2.0 with `initialize` / `tools/list` / `tools/call`.
- Supervisor: `packages/brain/src/june_brain/skills/supervisor.py` — spawns each enabled skill, discovers its tools, bridges them to LangChain `StructuredTool` instances, and restarts on crash with a single retry.
- Manifest: `~/Library/Application Support/June/skills.toml`, loaded via `packages/brain/src/june_brain/skills/manifest.py` with default materialization, missing-entry recovery, and malformed-TOML tolerance.
- HTTP surface: `GET /skills` and `POST /skills/{key}/toggle` in `packages/api/src/june_api/routes/skills.py`. Toggling persists to the manifest and calls `brain_graph.reload_agent()` so the next turn sees the new tool list.
- Web route: `apps/web/src/routes/skills/+page.svelte` lists skills with enable/disable, live status, and the tools each exposes.

Critical pitfall resolved: a skill subprocess that imports `june_brain.memory` re-enters the brain's `__init__`. Without a guard, the agent would be rebuilt inside each child, which would spawn five more skill subprocesses — a fork bomb. The supervisor sets `JUNE_IS_SKILL_SUBPROCESS=1` and `JUNE_SKILLS_DISABLED=1` in the child environment; `graph.py` skips module-level agent creation under that flag, and `get_supervisor()` skips auto-start.
