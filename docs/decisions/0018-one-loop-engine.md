# ADR 0018 — One Loop Engine (Hand-Written)

## Status

Accepted. Supersedes the dual-engine arrangement of build-spec C.2 (the CLEAR
experiment that ran a hand-written loop alongside a LangGraph engine behind the
`JUNE_LOOP_ENGINE` / `JUNE_CHAT_USE_HARNESS` flags). Anchored by the
[rebuild plan](../product/rebuild-plan.md), S1.

## Context

June's harness loop shipped with two implementations behind a stable
`HarnessLoop` interface: a hand-written loop and a LangGraph wrapper. The C.2
experiment (CLEAR) existed to decide between them on cost, latency, efficacy,
assurance, and reliability. The hand-written loop was always the default; the
LangGraph engine was retained as a fallback and a comparison baseline.

The experiment has concluded. Recorded in
[`docs/experiments/loop-clear.md`](../experiments/loop-clear.md) and copied into
[`docs/experiments/baseline-2026-06.md`](../experiments/baseline-2026-06.md):
the hand-written loop is **3-17x faster** than LangGraph on every task at equal
efficacy (100%) and assurance (100%). LangGraph showed lower run-to-run variance
on three of five tasks, but the absolute latency gap (1-7 s vs 22-37 s) is
decisive for a local-first personal agent; the variance is addressed
independently by structured tool calling (S5), not by keeping a second engine.

Keeping the second engine carried ongoing cost:

- It violated the standing invariant against any dependency that can be
  implemented customly (CLAUDE.md, build-spec): LangGraph + LangChain pulled a
  large transitive tree into every install.
- It widened the supply-chain surface and the install footprint — the opposite
  of the rebuild's distribution goal (a small, trustworthy desktop bundle).
- It doubled the maintenance surface of the most security-sensitive code path
  (the loop that dispatches tools), and made the loop's behavior depend on which
  engine was active rather than on data flowing through one fixed shape.

The LangGraph engine was also load-bearing beyond chat: it backed the task
planner-executor and a set of agent-lifecycle hooks in the API layer, which had
to be rewired before it could be removed.

## Decision

June has **one loop engine: the hand-written loop**. The `HarnessLoop` interface
and its boundary types (`SessionState`, `StreamEvent`, `TurnResult`,
`TurnProvenance`) remain — they are the fixed shape (CLAUDE.md invariant: the
harness core is fixed and never self-modified) — but there is exactly one
implementation behind them.

- **Removed:** `graph.py` (the LangGraph agent), `loop/langgraph_loop.py`, the
  `JUNE_LOOP_ENGINE` and `JUNE_CHAT_USE_HARNESS` flags, and the LangGraph
  fallback branch in the chat route. `loop/engine.get_loop()` now returns the
  hand-written loop unconditionally.
- **Relocated:** the engine-neutral helpers the loop depends on (prose-JSON
  tool-call extraction, runtime tool selection, per-turn recall) moved from
  `graph.py` to `loop/agent_helpers.py`, free of any LangChain import.
- **Rewired:** the scheduler's synchronous invocation (`scheduler/agent.py`) and
  the task runtime (`tasks/runtime.py`) now run through the provider stack and
  the hand-written loop's `stream_turn` respectively; the API agent-lifecycle
  hooks (`invalidate_agent` / `reload_agent` / `startup_error`) are gone because
  the hand-written loop has no compiled agent to rebuild — it resolves config,
  keys, and the tool surface fresh each turn.
- **Safety net:** the glass-box trace store (`loop/trace.py`) is the durable
  record of what the single engine did each turn — it replaces the dual-engine
  comparison as the mechanism for catching regressions.

This ADR removes only the LangGraph *engine*. LangChain primitives still back
the tool abstraction (`tools.py`, the skill loader) and the `models.py`
chat-model construction; replacing those and dropping the `langgraph` /
`langchain` dependencies is a separate slice (rebuild plan S1.2).

## Alternatives Considered

- **Keep LangGraph as an opt-in experimental fallback.** Rejected. A fallback
  that is never the default still pays the full dependency, supply-chain, and
  maintenance cost, and its existence weakens the "no avoidable dependency"
  invariant the rebuild is trying to strengthen. The trace store gives us the
  observability the fallback was nominally for.
- **Adopt LangGraph as the single engine instead.** Rejected by the CLEAR
  evidence: 3-17x slower at equal efficacy, on a local-first product where
  latency is the felt experience.

## Consequences

Positive: one code path for the most security-sensitive part of the system;
loop behavior is a function of data through a fixed shape, not of engine
identity; the install footprint and supply-chain surface shrink (fully realized
once S1.2 drops the dependencies); the maintenance burden of the loop halves.

Negative / accepted: the lower run-to-run variance LangGraph showed on some
tasks is given up; S5 (provider-native structured tool calling) is the planned,
measured path to reducing that variance on the one remaining engine. Removed env
vars (`JUNE_LOOP_ENGINE`, `JUNE_CHAT_USE_HARNESS`) are simply ignored if still
set; they are documented as removed in the rebuild's migration notes.
