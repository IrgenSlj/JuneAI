# Changelog

All notable changes to June are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Canonical build specification** (`docs/product/build-spec.md`): June is a
  personal assistant whose center of gravity is the user, not the task, defined by
  four inversions of a coding agent and built in tiers (Tier 1 spine first). New
  ADRs lock the direction — 0015 (four inversions), 0016 (event-driven, no
  heartbeat), 0017 (model-specific provider layer) — and the CLEAR experiment
  scaffold (`docs/experiments/loop-clear.md`) is in place for the loop measurement.

### Changed

- **Direction realigned across the docs.** Vision, product overview, architecture
  overview, and both roadmaps now describe the Tier 1 spine (portable data
  directory, model-specific providers, measured loop, layered context with anchored
  compaction, salience recall, honest character, visible cloud boundary). The
  earlier "personal operating layer / Quick Capture / Daily Home" framing (ADR
  0013, ADR 0014, the agentic-pivot and v0.1.1 plans) is retained as historical
  context only.

### Added (earlier in this cycle)

- **Quick Capture**: `POST /capture` classifies a thought locally (rules first,
  with a local-only Gemma fallback that never reaches the cloud) into typed
  candidates, recorded through a durable event ledger; a capture box on the home
  screen shows what June understood and what it would do.
- **Durable event ledger** (`events`, `capture_items`, `action_intents` tables)
  underpinning the operating layer; included in export/import.
- **Local startup greeting** in the empty chat (`GET /greeting`), generated
  on-device with no model call.
- **`tools/run.sh`**: one-command launcher that starts Ollama (if needed), the
  API, and the web app together.
- Personal operating layer plan for v0.1.1: quick capture, Daily Home, durable
  event ledger, action intents, approvals, promises, agenda suggestions, and
  Telegram quick capture.
- ADR 0014 and a research memo comparing the relevant open-source and
  closed-source patterns.

### Changed

- Default local model is now `gemma4:e2b` (smallest, fastest start); the local
  model is warmed and kept resident (`keep_alive`) on API startup to remove
  cold starts.
- `check.sh` now enforces `ruff` and a focused mypy gate (`operator`/`call-arg`
  error classes); CI tests both Python 3.13 and 3.14.
- Roadmap and docs now treat v0.1.1 as the active development track.
- Documentation now reflects Python 3.13 and the shipped v0.1.0 Apple Silicon
  DMG.

### Fixed

- Scheduler/orchestration crash cluster: `MEMORY_DIR` (a string) was used as a
  path and `Schedule` was constructed without its required `id`, which left
  first-run default schedules silently uncreated and the `/schedules` route
  broken on every call.
- Settings page no longer hangs on "Loading…" when the brain is unreachable; it
  shows the offline notice with a retry, like every other screen.
- SQLite stores now create the data directory if it does not exist yet
  (fresh-install fix).

## [0.1.0] - 2026-05-25

First tagged release of the June v2 monorepo. June is the open personal AI that
remembers you: local-first by default via Gemma 4 through Ollama, with optional
cloud intelligence via Gemini on consent.

### Added

- **Three-tier model router** with per-skill model policy, a user privacy dial,
  and per-message provenance chips (provider, model, tier, latency) under each
  assistant turn.
- **Tasks primitive** end-to-end: REST API, a `/tasks` page with composer and
  active/recent grouping, a live SSE step trace, double-start rejection, and
  cooperative cancel.
- **Scheduler**: recurring tasks, a scheduler REST API, agent tools, a
  notification dispatch endpoint, and an event poller.
- **Daily orchestration**: morning briefing, end-of-day review, and
  carry-forward of unfinished items.
- **Telegram integration** and a pluggable notification channel.
- **Files skill** expanded with `list_directory`, `read_file`, and
  `search_files`, sandboxed to `$HOME` with symlink containment.
- **MCP registry connector**: a curated catalog with install/uninstall and a
  browse panel on `/skills`.
- **Memory, Skills, and System surfaces**: memory statistics, a per-tool skill
  playground, and a rolling system activity log.
- **Data portability**: export and import of all local data.
- **Desktop shell** (Tauri 2): Ollama process supervision, native affordances
  (tray, window state, autostart, global shortcut), and a first compile.
- **macOS release asset**: Apple Silicon DMG published on GitHub Releases. The
  artifact is ad-hoc signed and not notarized.
- **First-run flow** and a static landing page for june.ai.

### Changed

- Split the monolithic SQLite layer into per-domain data-access objects.
- Capped `chromadb` (<2.0) and `fastapi` (<1.0) major versions.
- Synced the generated OpenAPI spec and TypeScript types with the scheduler and
  notification endpoints.

### Fixed

- Codebase stabilisation pass: migrations, auth, tool aliases, traceback
  logging, defensive chat tool-argument coercion, and a `ruff check`-clean tree.
</content>
</invoke>
