# Agentic Pivot Plan

This document is the execution plan for the May 2026 reframing of June from "chat-with-memory" to "personal agent with memory." It runs for twelve weeks and replaces, in priority terms, the [open-source readiness plan](open-source-readiness-plan.md) and the [memory-skills plan](memory-skills-plan.md) Phase C items. Those plans remain valid as backlog and will be folded back in once the pivot's Sprint 1 is shipped.

The strategic decisions behind this plan are recorded in [ADR 0009](../decisions/0009-private-by-default-and-model-routing.md) and [ADR 0010](../decisions/0010-agentic-core-tasks-oauth-computer-use.md).

## The Bet

> Mainstream users do not want a smarter chatbot. They want an assistant that does work across the apps and services they already use, knows them well, and never asks them to log in.

June already has the memory and the privacy story. The pivot is to add **agency** — task execution across files, OS apps, OAuth services, the browser, and the broader MCP ecosystem — without breaking the "no-account, your-data-is-yours" promise that brought users in.

## The Bar (definition of "we shipped the pivot")

By the end of Sprint 4, the following is true:

1. A new user installs June on a Mac in under five minutes with no terminal commands.
2. On first run they pick a privacy level, connect Gmail and Calendar, and run a single demo task ("find my next flight and add the gate info to my calendar when it's announced").
3. The task runs to completion. The user can see which model handled which step, which files or services were read or written, and what is now stored in memory.
4. Closing the laptop does not stop the task. Reopening shows the result.
5. At least one third-party MCP server (probably the official Notion or Linear one) is installable from June's in-app registry and works end-to-end.
6. Fifty non-developer beta users are running June daily and reporting their failures in a shared Discord.

## Four Sprints

Each sprint is three weeks. Each ends with a public-ish artifact (a video, a blog post, an installer) that forces honest assessment.

### Sprint 1 — Agentic Core (weeks 1-3)

The architectural pivot. Detailed task list: Task #3 through #9 in the task tracker.

**Status as of 2026-05-19:** five of the seven planned modules have shipped to `main`, plus a four-module "Batch 1" of cross-department UX adds. Two modules remain (OAuth skills, browser skill) plus the desktop first-compile gate.

Outcomes:

- **SHIPPED — Three-tier model router** (Task #3, commit `017cca8b`). `routing.py` with `SkillModelPolicy` / `UserPrivacyDial` / `ResolvedTier` / `ModelRouter` / `ModelProvenance`. Privacy dial wired through `/settings` (commit `017cca8b`). Per-message provenance in chat (slice 1.1b) is the remaining piece.
- **SHIPPED — Tasks primitive** (Task #4, commits `017cca8b` + `04a1e432`). New `tasks` SQLite table sharing `june.db`, `tasks/{models,store,runtime}.py`, REST API at `/tasks/{user_id}`, `/tasks` SvelteKit page. `TaskRuntime` pipes the goal through the existing LangGraph agent and records every tool call as a step. SSE live-trace deferred until the runtime has more to stream.
- **PARTIAL — Real files skill** (Task #5, commit `017cca8b`). Tools `list_directory`, `read_file`, `search_files` added alongside `read_pdf`/`read_webpage`, all sandboxed to `$HOME` with symlink containment. Tauri-backed per-folder permission grants (1.3b) and Web File System Access path (1.3c) still pending.
- **PENDING — Real gmail and gcal skills** (Task #6). OAuth setup blocks on the Google verified-app review (1-2 week lead time).
- **PENDING — Browser skill** (Task #7).
- **SHIPPED — MCP registry connector** (Task #8, commit `017cca8b`). Curated catalog of six entries (filesystem, github, notion, postgres, brave-search, sqlite). `GET /skills/registry`, install, uninstall. "Browse the MCP registry" panel at the bottom of `/skills`.
- **PENDING — Desktop Phase 4.5 First Compile** (Task #9). Requires installing rustup on the dev machine, which is user-interactive.

#### Batch 1 — one module per department (2026-05-19)

Added on top of Sprint 1 to give each surface a tangible new capability:

- **TaskRuntime** (Tasks) — see above.
- **MemoryStats** (Memory, commit `e0c866c1`). `GET /memory/{user_id}/stats` returns per-store counts, last-write timestamp, and most-recent semantic facts. Card at the top of `/memory`.
- **SkillPlayground** (Skills, commit `05c45345`). Per-tool form generated from each tool's `input_schema`. `POST /skills/{key}/tools/{tool}/invoke` runs it; the panel shows the raw result plus an ok/latency chip.
- **SystemActivity** (System, commits `12db625e` + `68246709`). Rolling 1000-row sqlite log written by a FastAPI middleware; `GET /system/activity` reads it; "Recent activity" card at the bottom of `/system` with status chips, latency, and a Clear button.

Artifact at end of sprint: a three-minute screen recording of "June plans a weekend trip" — uses gmail, gcal, browser, and files together. (Will be possible once 1.4 + 1.5 ship.)

### Sprint 2 — Dogfooding (weeks 4-6)

Owner uses June daily for real work for three weeks. No new feature work that does not come from the dogfooding journal.

Outcomes:

- Three weeks of `docs/dogfood-log.md` entries, one per day.
- A rewritten backlog for Sprint 3 driven by observed pain, not by speculation.
- Top three failure modes identified and fixed in-sprint if they are small enough.
- A `compute` skill considered, scoped, and added to Sprint 3 only if dogfooding shows a real escape-hatch need.

Artifact at end of sprint: a written retro at `docs/retros/2026-sprint-2-dogfood.md` with the most-used five tool calls, the most painful five failures, and the three highest-value Sprint 3 items.

### Sprint 3 — Installable for humans (weeks 7-9)

Make a mainstream user able to use June without reading anything.

Outcomes:

- Signed installers: macOS (notarized .dmg), Windows (signed .msi). Linux as AppImage if cheap.
- Three-question first-run flow: name, services to connect (multi-select), privacy default (radio).
- Onboarding video (90 seconds, embedded in first-run).
- Landing page at june.ai or junepersonal.com — single page, video at top, install button, three-paragraph privacy explainer, no marketing fluff.
- README rewrite for the mainstream-user audience; the developer README moves to `docs/contributing.md`.
- MCP registry browser polished with descriptions, install counts (faked at first, real after we have any data), and per-skill ratings.
- Auto-update wired up so beta users get fixes without re-downloading.

Artifact at end of sprint: a public landing page with an installer that works for someone who has never used a terminal.

### Sprint 4 — 50-user closed beta (weeks 10-12)

Find out if we are right.

Outcomes:

- 50 non-developer beta users recruited (friends of friends, Twitter, ProductHunt ship list).
- Discord server with one channel per major feature surface.
- Weekly thirty-minute office hours.
- Per-week metric tracking: install completion rate, tasks-created per active user, retention by week, top three failure modes per surface.
- A failure log per user with a triage tag (`needs-fix`, `needs-doc`, `wontfix`).

Artifact at end of sprint: a written go/no-go decision at `docs/retros/2026-sprint-4-beta.md`. Go means scale to a public alpha. No-go means iterate inside the beta for another sprint with no shame.

## What this plan explicitly drops or defers

- **Mobile shell.** Stays trigger-gated per the existing [roadmap](roadmap.md). The trigger has not fired.
- **Skill marketplace economics.** Marketplace exists from Sprint 1.6 in install-only form; any economics around skills are out of scope for this plan.
- **Multi-user support.** Deferred until a user asks.
- **Voice.** Deferred to whichever sprint a beta user asks for it first.
- **Cross-device memory sync and hosted task execution.** Considered for a later sprint; not in scope for the twelve-week pivot.

## What this plan explicitly keeps from June 1.0

- The three-store memory architecture and the `MemoryManager` facade. Untouched.
- Skills-as-MCP. Extended, not replaced.
- One codebase, three surfaces. The web PWA stays as the try-before-install funnel.
- No account by default. The dial-to-local-only path remains for users who want June 1.0's promise.

## Open questions (answer before Sprint 1 starts)

1. **Domain.** June.ai is unlikely to be available; junepersonal.com, june.so, hellojune.app, getjune.com are plausible alternates. Pick before the landing page.
2. **License.** Currently the repo has no LICENSE file. Pick before Sprint 3 (Apache 2.0 is the default recommendation; MIT is the alternative).
3. **OAuth client IDs.** Gmail and gcal need a published OAuth application with verified scopes. This takes one to two weeks of Google review and must start in Sprint 1 week 1.

## How this plan supersedes existing plans

- [open-source readiness plan](open-source-readiness-plan.md): paused. Its items (provider correctness, conversation continuity, memory delete-edit correctness, fresh-clone reliability) are mostly already shipped or absorbed into Sprint 1. The hardening pass resumes after Sprint 4 if there is anything left worth doing.
- [memory-skills plan](memory-skills-plan.md): Phase A and Phase B shipped items stay. Phase C (`MemoryManager.write()` single write path; skills routing through it) is folded into Sprint 1.4 (the OAuth skills must write to memory through a unified path) and Sprint 1.5 (the browser skill too).
- [desktop-shell plan](desktop-shell-plan.md): Phase 4.5 First Compile becomes Sprint 1.7 of this plan. Phases 5-7 stay deferred until Sprint 3, where they merge with the "installable for humans" sprint.
- [responsive plan](responsive-plan.md): unchanged. Touch and tablet hardening is a Sprint 3 item.

## Why this plan is honest about scope

Twelve weeks is a lot of work for a small team. The plan does not pretend otherwise. The three things that make it possible:

- Memory, the three surfaces, the MCP skill substrate, the SvelteKit UI, the FastAPI API, and the supervisor are all already built. We are *adding* an agentic layer on top, not rewriting.
- The Tauri shell is 80 percent done; Phase 4.5 First Compile is the unlock.
- The MCP ecosystem provides skills we do not have to write.

What we are committing to write from scratch: the router, the task primitive, three OAuth skills, the browser skill, the registry connector, and a real first-run flow. Everything else is plumbing or polish.

## Status tracking

The twelve sprints are tracked as tasks #1 through #12 in the project task tracker. Foundation work (this document plus the ADRs) is Task #1. Vision and roadmap canonical updates are Task #2. Sprints 1.1 through 1.7 are Tasks #3 through #9. Sprints 2, 3, and 4 are Tasks #10, #11, #12.
