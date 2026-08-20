# June Documentation

This directory is the documentation root. Everything you need to understand, contribute to, or operate June lives here.

## Start here

- **[CURRENT.md](CURRENT.md)** — the single authoritative "state of the project"
  page (per-subsystem summary, active plan, release status). Read this first.
- **Active plan:** [`docs/product/v0.4-development-plan.md`](product/v0.4-development-plan.md)
  (the current lead document). Previous plans (`v0.3-development-plan.md`,
  `JUNE_V02_BRIEF.md`, `v0.2-execution-plan.md`) are superseded — see
  [archive](archive/README.md).

## What's New

- **v0.4 plan (18 August 2026)** — correctness and coherence, from the
  [2026-08-18 audit](product/repo-audit-2026-08-18.md). Stream D deleted the v1
  tool surface, settled the memory tools June actually offers (ADR 0032), and
  turned the stated invariants into gate checks. Completed 20 August 2026.
  See [`v0.4-development-plan.md`](product/v0.4-development-plan.md).
- **v0.3 plan (24 July 2026)** — superseded. Repositioned the product from
  local-first to "the agent that can prove what it did"; that rationale still
  stands and is cited from `CURRENT.md`.
- **v0.2 phase (6 July 2026)** — auditable memory as a product: retrieval v2,
  memory provenance/quarantine, Night Shift consolidation, signed/notarized
  distribution. See [`JUNE_V02_BRIEF.md`](../JUNE_V02_BRIEF.md) (superseded).
- **The four inversions** — June borrows a coding agent's skeleton but inverts its
  four operations: defer (not verify), continue (not complete), forget (not
  accumulate), stay quiet (not act fast). See
  [ADR 0015](decisions/0015-center-of-gravity-four-inversions.md).

## Quick links

- [**Current state**](CURRENT.md) — authoritative state-of-the-project page (start here)
- [**v0.4 plan**](product/v0.4-development-plan.md) — the active lead plan
- [**Vision**](vision.md) — the product premise and the non-negotiables
- [**Product overview**](product/overview.md) — what June is
- [**Roadmap**](product/roadmap.md) — what ships next
- [**Architecture overview**](architecture/overview.md) — how the system is layered
- [**Architecture decisions**](decisions/README.md) — the ADRs that justify the design
- [**Environment**](setup/environment.md) — runtime configuration reference
- [**Desktop setup**](setup/desktop.md) — Rust toolchain, run/build commands for the desktop shell
- [**Archive**](archive/README.md) — superseded plans (history only)
- _Historical:_ [experiments](experiments/loop-clear.md) — CLEAR measurements (e.g. loop engine)

## Structure

```
docs/
├── vision.md                      # product north star
├── CURRENT.md                     # authoritative state-of-the-project page
├── RECONCILIATION.md              # v0.2 brief vs. actual repo state (historical reference)
├── product/
│   ├── v0.4-development-plan.md   # THE ACTIVE LEAD PLAN
│   ├── v0.3-development-plan.md   # SUPERSEDED — v0.3 repositioning rationale
│   ├── overview.md                # what June is
│   ├── roadmap.md                 # Tier 1/2/3 sequencing
│   ├── v0.2-execution-plan.md     # SUPERSEDED — v0.2 execution sequencing
│   ├── rebuild-plan.md            # SUPERSEDED — history only
│   └── development-plan.md        # SUPERSEDED — history only
├── architecture/
│   └── overview.md                # layered system architecture
├── decisions/                     # Architecture Decision Records (0001-0024)
│   ├── README.md                  # ADR index
│   ├── 0015-center-of-gravity-four-inversions.md
│   ├── 0016-event-driven-no-heartbeat.md
│   └── 0017-model-specific-provider-layer.md
├── experiments/
│   └── loop-clear.md              # CLEAR loop-engine measurement (C.2)
├── design/
│   └── master-brief.md            # SUPERSEDED — UI design brief
├── archive/
│   └── README.md                  # index of superseded docs
├── setup/
│   ├── environment.md             # env vars, .env template
│   └── desktop.md                 # Rust toolchain, dev/build commands
└── README.md                      # this file
```

## Documentation Rules

1. [`v0.4-development-plan.md`](product/v0.4-development-plan.md) is the active
   implementation direction and [CURRENT.md](CURRENT.md) is the authoritative
   state page. Where product premise conflicts, [vision](vision.md) and
   [overview](product/overview.md) win.
2. Every architectural decision gets an ADR. If it is worth debating, it is
   worth recording.
3. The product overview describes what June is. The roadmap describes what ships
   next and sequences the development plan.
4. No emojis in documentation.
5. Complete sentences. Two-page maximum per document unless the content genuinely
   requires more.

## How the Docs Evolve

- The vision evolves rarely. Material changes require deliberate discussion.
- ADRs are append-only. Superseding an ADR means writing a new one that
  references and deprecates the old one.
- The product overview is updated when the shipped surface area changes.
- The roadmap is updated when a trigger fires, a feature ships, or priorities
  shift.
- The environment reference is updated whenever runtime configuration changes.
