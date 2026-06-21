# June Documentation

This directory is the documentation root. Everything you need to understand, contribute to, or operate June lives here.

## What's New

- **New direction (28 May 2026)** — June is a personal assistant whose center of
  gravity is the user, not the task. The canonical, decision-by-decision working
  plan is the [rebuild plan](product/rebuild-plan.md); the durable worldview is
  the [vision](vision.md). This supersedes the earlier "personal operating layer /
  Quick Capture" framing (ADR 0013, ADR 0014), retained as historical context.
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

- [**Vision**](vision.md) — the product premise and the non-negotiables
- [**Product overview**](product/overview.md) — what June is
- [**Roadmap**](product/roadmap.md) — what ships next, sequencing the rebuild plan
- [**Architecture overview**](architecture/overview.md) — how the system is layered
- [**Architecture decisions**](decisions/README.md) — the ADRs that justify the design
- [**Rebuild plan**](product/rebuild-plan.md) — the authoritative, decision-by-decision working plan
- [**Experiments**](experiments/loop-clear.md) — CLEAR measurements (e.g. loop engine)
- [**Environment**](setup/environment.md) — runtime configuration reference
- [**Desktop setup**](setup/desktop.md) — Rust toolchain, run/build commands for the desktop shell
- [**Design brief**](design/claude-design-prompt.md) — the prompt for iterating on June's UI

## Structure

```
docs/
├── vision.md                      # product north star
├── product/
│   ├── rebuild-plan.md            # canonical, decision-by-decision working plan
│   ├── overview.md                # what June is
│   └── roadmap.md                 # Tier 1/2/3, sequences the rebuild plan
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
│   └── claude-design-prompt.md    # UI design brief
├── setup/
│   ├── environment.md             # env vars, .env template
│   └── desktop.md                 # Rust toolchain, dev/build commands
└── README.md                      # this file
```

## Documentation Rules

1. The [rebuild plan](product/rebuild-plan.md) is the authoritative direction. Where it conflicts with anything else, it wins; the [vision](vision.md) is the tiebreaker for product premise.
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
