# Next Sessions

This is the working list for the next development sessions. It is intentionally concrete and ordered.

## Session 07 — Finish `app.py` Decomposition

Goal: move the remaining heavy runtime and dialog sections out of [app.py](/Users/admin/JuneAI/JuneAI-app/app.py).

Work:

- extract `open_settings_dialog`
- extract `open_calendar_dialog`
- extract model download and startup recovery UI
- move runtime preset UI helpers into `agent_ui/`
- add targeted tests for the extracted helpers where practical

Success criteria:

- `app.py` is materially smaller and easier to scan
- dialog/runtime code has dedicated modules
- behavior stays unchanged under unit, integration, and smoke checks

## Session 08 — Runtime Validation and Preset UX

Goal: make runtime selection safer and clearer.

Work:

- detect invalid preset/model combinations before generation
- warn when local model tags do not match installed Ollama tags
- explain shared override caveats such as `LOCAL_LARGE_MODEL_NAME`
- improve `check_ollama` and related scripts for real runtime resolution
- surface current provider, model, tool strategy, and privacy mode consistently

Success criteria:

- fewer silent misconfigurations
- clearer local-model download and fallback guidance
- one obvious path to recover from model-not-installed states

## Session 09 — Tooling, Telemetry, and Evaluation

Goal: make agent behavior inspectable.

Work:

- add a dedicated tool evaluation/debug surface in the app
- expose per-turn tool outcomes and what June saved
- add golden-path evaluation fixtures for common user flows
- add developer commands for running targeted behavioral checks

Success criteria:

- tool behavior can be inspected without reading raw logs
- core agent flows have repeatable evaluation coverage

## Session 10 — Memory Reporting and Export

Goal: make memory easier to inspect, back up, and reason about.

Work:

- improve export formats and summaries
- add chapter-filtered exports
- add a documented backup and restore workflow
- review schema/query helpers for reporting use cases

Success criteria:

- user memory is portable and readable
- reporting no longer depends on ad hoc inspection

## Session 11 — Packaging and Release Readiness

Goal: make the project easier to run on a clean machine.

Work:

- harden Docker build and compose workflows
- verify package metadata and editable install behavior
- add a clean-machine setup checklist
- define release artifacts and release notes workflow

Success criteria:

- repo can be bootstrapped from docs alone
- packaging and compose flows are documented and repeatable

## Session 12 — Product Polish

Goal: improve the end-user experience after the architecture cleanup.

Work:

- refine onboarding and chapter-specific starter prompts
- tighten memory panel and workspace interactions
- review mobile layout behavior
- revisit the visual hierarchy of Today, Memory, and Workspace surfaces

Success criteria:

- fewer dead ends on first use
- clearer product value without reading docs first

## Environment Reminder

Before any new session, review [docs/setup/environment.md](/Users/admin/JuneAI/docs/setup/environment.md).

Variables most likely to matter:

- `MODEL_PRESET`
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LOCAL_GEMMA_MODEL_NAME`
- `LOCAL_LARGE_MODEL_NAME`
- `ANTHROPIC_API_KEY`
- `MEMORY_DIR`

Also check your local `JuneAI-app/.env`, because it may still hold an older preset selection from previous sessions.
