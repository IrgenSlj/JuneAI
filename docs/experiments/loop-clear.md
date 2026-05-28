# CLEAR — Loop Engine Experiment (C.2)

This file records the measured comparison between the hand-written harness loop and the LangGraph
loop, both run behind the same `HarnessLoop` interface (see
[build-spec.md C.2](../product/build-spec.md)). The winner becomes the default engine; the loser is
kept behind the interface until the result is stable.

Status: **not yet run.** Populated by `packages/brain/june_brain/loop/experiment.py` when C.2 lands.

## Method

Run both engines on five representative tasks:

1. A recall question.
2. A multi-step task with two tool calls.
3. A long conversation that triggers compaction.
4. A cloud-escalation case.
5. A "stay quiet" case.

## CLEAR metrics

| Metric | Meaning |
|---|---|
| **C**ost | Total tokens (local + cloud) per task. |
| **L**atency | Wall-clock ms per task. |
| **E**fficacy | Task succeeded (yes/no, or rubric score). |
| **A**ssurance | Privacy/provenance events were correct and complete. |
| **R**eliability | Variance across five runs of the same task. |

## Results

_To be filled in by the experiment harness._

| Task | Engine | Cost (tok) | Latency (ms) | Efficacy | Assurance | Reliability |
|---|---|---|---|---|---|---|
| — | handwritten | — | — | — | — | — |
| — | langgraph | — | — | — | — | — |

## Decision

_Hypothesis: hand-written wins on cost and latency with equal efficacy. Confirm or refute here, then
set the default engine in config._
