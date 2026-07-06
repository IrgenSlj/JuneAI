# June Documentation

This directory is the documentation root. Everything you need to understand, contribute to, or operate June lives here.

## Start here

- **[CURRENT.md](CURRENT.md)** — the single authoritative "state of the project"
  page (per-subsystem summary, active plan, release status). Read this first.
- **Active plan:** [`JUNE_V02_BRIEF.md`](../JUNE_V02_BRIEF.md) (the v0.2 lead
  document) with the [execution plan](product/v0.2-execution-plan.md) and the
  brief-vs-reality [reconciliation](RECONCILIATION.md).

## What's New

- **v0.2 phase (6 July 2026)** — auditable memory as a product: retrieval v2,
  memory provenance/quarantine, Night Shift consolidation, signed/notarized
  distribution. See [`JUNE_V02_BRIEF.md`](../JUNE_V02_BRIEF.md) and
  [CURRENT.md](CURRENT.md).
- **Current direction (28 June 2026)** — June is a trusted continuity engine.
  Chat is the input surface; the product center is what June is holding:
  Promises, Memory, Trust, Skills, and explicit Time. (The pre-v0.2
  [development plan](product/development-plan.md) is now superseded; the durable
  worldview is the [vision](vision.md).)
- **The four inversions** — June borrows a coding agent's skeleton but inverts its
  four operations: defer (not verify), continue (not complete), forget (not
  accumulate), stay quiet (not act fast). See
  [ADR 0015](decisions/0015-center-of-gravity-four-inversions.md).
- **No heartbeat** — June acts on user input or real-world events, never on a
  timer. See [ADR 0016](decisions/0016-event-driven-no-heartbeat.md).
- **Model-specific providers** — June is tuned for Gemma 4 + Gemini, not abstracted
  to be model-agnostic. See [ADR 0017](decisions/0017-model-specific-provider-layer.md).
- **Desktop shell builds** — The Tauri shell produced a v0.1.0 Apple Silicon DMG
  (ad-hoc signed, not notarized); signed distribution is deferred.

- [**Current state**](CURRENT.md) — authoritative state-of-the-project page (start here)
- [**v0.2 brief**](../JUNE_V02_BRIEF.md) — the active lead plan; [execution plan](product/v0.2-execution-plan.md), [reconciliation](RECONCILIATION.md)
- [**Vision**](vision.md) — the product premise and the non-negotiables
- [**Product overview**](product/overview.md) — what June is
- [**Roadmap**](product/roadmap.md) — what ships next
- [**Architecture overview**](architecture/overview.md) — how the system is layered
- [**Architecture decisions**](decisions/README.md) — the ADRs that justify the design
- [**Environment**](setup/environment.md) — runtime configuration reference
- [**Desktop setup**](setup/desktop.md) — Rust toolchain, run/build commands for the desktop shell
- _Superseded (history only):_ [development plan](product/development-plan.md), [rebuild plan](product/rebuild-plan.md), [design brief](design/master-brief.md), [ship-to-revenue](product/ship-to-revenue.md)
- _Historical:_ [experiments](experiments/loop-clear.md) — CLEAR measurements (e.g. loop engine)

## Structure

```
docs/
├── vision.md                      # product north star
├── CURRENT.md                    # authoritative state-of-the-project page
├── RECONCILIATION.md             # v0.2 brief vs. actual repo state
├── product/
│   ├── v0.2-execution-plan.md    # how the v0.2 brief gets built (sequencing)
│   ├── overview.md                # what June is
│   ├── roadmap.md                 # Tier 1/2/3 sequencing
│   ├── rebuild-plan.md            # SUPERSEDED — history only
│   └── development-plan.md        # SUPERSEDED — history only
├── architecture/
│   └── overview.md                # layered system architecture
├── decisions/                     # Architecture Decision Records (0001–0017)
│   ├── README.md                  # ADR index
│   ├── 0015-center-of-gravity-four-inversions.md
│   ├── 0016-event-driven-no-heartbeat.md
│   └── 0017-model-specific-provider-layer.md
├── experiments/
│   └── loop-clear.md              # CLEAR loop-engine measurement (C.2)
├── design/
│   └── master-brief.md            # UI design brief
├── setup/
│   ├── environment.md             # env vars, .env template
│   └── desktop.md                 # Rust toolchain, dev/build commands
└── README.md                      # this file
```

## Documentation Rules

1. [`JUNE_V02_BRIEF.md`](../JUNE_V02_BRIEF.md) is the active implementation direction and [CURRENT.md](CURRENT.md) is the authoritative state page. Where product premise conflicts, [vision](vision.md) and [overview](product/overview.md) win.
2. Every architectural decision gets an ADR. If it is worth debating, it is worth recording.
3. The product overview describes what June is. The roadmap describes what ships next and sequences the rebuild plan.
4. No emojis in documentation.
5. Complete sentences. Two-page maximum per document unless the content genuinely requires more.

## How the Docs Evolve

- The vision evolves rarely. Material changes require deliberate discussion.
- ADRs are append-only. Superseding an ADR means writing a new one that references and deprecates the old one.
- The product overview is updated when the shipped surface area changes.
- The roadmap is updated when a trigger fires, a feature ships, or priorities shift.
- The environment reference is updated whenever runtime configuration changes.
