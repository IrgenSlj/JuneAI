# Product Roadmap

This is the current roadmap for JuneAI after the reliability, runtime, and docs cleanup phases.

## Current Baseline

- The app runs as a Streamlit shell with a LangGraph agent core
- Memory is persisted in SQLite
- Local and API-assisted runtime presets are supported
- Source-tree lint and type checks are enforced through `make lint`
- Unit and integration suites are green
- Documentation has been consolidated into the root `docs/` tree

## Completed Foundations

### Reliability and Runtime

- startup and smoke workflows are stable
- runtime defaults are aligned with the codebase
- model construction and reduced-tool runtime behavior are fixed
- source lint/type gates are now enforceable and passing

### Documentation and Repo Hygiene

- root and app guides are separated by responsibility
- architecture docs now match SQLite memory instead of legacy JSON storage
- environment variables are documented in one canonical place

## Next Strategic Themes

### 1. App Decomposition

Continue moving large, cohesive sections out of `app.py`, starting with:

- settings dialog and runtime picker flow
- calendar dialog and calendar rendering helpers
- model download/startup recovery screens

### 2. Runtime UX and Evaluation

- add explicit runtime validation before generation starts
- surface model/preset mismatches in the UI
- add tool and save outcome inspection screens

### 3. Observability and Confidence

- richer telemetry views for routes, tool calls, and persistence outcomes
- fixture-based evaluation transcripts for common workflows
- docs and scripts for repeatable local verification

### 4. Packaging and Release Readiness

- harden Docker and compose workflows
- add a release checklist and environment sanity checklist
- close remaining packaging/documentation gaps around app distribution

### 5. Product Iteration

- improve onboarding and first-run guidance
- strengthen chapter-specific suggestions and memory surfacing
- continue reducing friction between chat, workspace, and memory panels
