# ADR 0020 — Provider-Native Structured Tool Calling (with Prose Fallback)

## Status

Accepted. Extends the provider boundary (ADR 0017) and the one hand-written loop
(ADR 0018). Anchored by the [rebuild plan](../product/rebuild-plan.md), S5.

## Context

The hand-written loop advertised tools in the system prompt and parsed tool
calls out of the model's *prose* — the model was asked to emit
`{"tool_calls": [...]}` as text, which the loop extracted with a JSON-in-prose
parser. This is the single biggest source of run-to-run unreliability: the CLEAR
experiment (ADR 0018) measured recall cv at 75.6%, dominated by the model
sometimes wrapping, narrating, or malforming the JSON. Local Gemma-class models
are more prone to this than frontier models, and June runs them by design.

Ollama's OpenAI-compatible endpoint supports native function calling (constrained
tool-call emission), as does Gemini. Using it moves tool invocation from "hope
the model formats prose correctly" to a structured channel the provider parses.

## Decision

**Tool calling is provider-native first, prose-JSON as the documented fallback.**

- **Provider boundary.** `GenerateRequest.tools: list[ToolSpec] | None` and
  `GenerateResult.tool_calls: list[ToolCall]` (providers/base). Shared OpenAI
  helpers (`tool_specs_to_openai`, `parse_openai_tool_calls`, tolerant of
  garbled arguments) are implemented by both `GemmaProvider` and
  `GeminiProvider`. A provider without support simply ignores `tools` and
  returns no `tool_calls`.
- **Loop.** `make_tool_specs` builds `ToolSpec`s from June's `Tool` abstraction
  (`Tool.args` → JSON Schema). `run_turn` advertises them on the generate call
  and `_resolve_tool_calls` prefers structured `result.tool_calls`, falling back
  to the prose extractor when none are returned (invariant 6 — graceful
  degradation ships in the same change).
- **Reliability suite.** `june_brain.experiments.reliability` (pure cv% math,
  unit-tested) + `tools/reliability_harness.py` measure run-to-run variance, so
  the improvement is verified, not assumed. Target: recall cv < 25%.
- **Tunable salience (no auto-tuning).** Salience weights (`rel`/`rec`/`freq`)
  move from import-time env reads to `SalienceWeights.load()` with precedence
  env > config store > default, so re-tuning takes effect on the next recall
  without a restart. The user turns the knobs; June never self-modifies the
  engine (invariant 3).

### Deliberately scoped out (follow-ups, not regressions)

- **Streaming-native tool calls.** `stream_turn` (the live chat path) still uses
  the prose path. Surfacing native tool-call *deltas* through the token stream
  requires changing the `stream()` contract to yield structured events; passing
  `tools` to a stream that can only yield strings would silently drop native
  calls and break tool use. Done as a separate, careful change on the live path
  rather than rushed here. `run_turn` — which the reliability harness and the
  task runtime use — gets native calling now.
- **Salience settings form + `/memory` feedback aggregate view.** The tuning
  *mechanism* (config-backed weights) is in place; the writable UI and the weekly
  thumb-feedback aggregate are incremental UI work.

## Alternatives Considered

- **Keep prose parsing only, harden the parser.** Rejected: no parser fixes a
  model that narrates its JSON; the structured channel removes the failure mode
  at the source. The parser stays as the fallback for models without support.
- **Constrained decoding via a JSON-schema `format` only (no function calling).**
  Useful for the difficulty classifier (S4) but the function-calling surface is
  the right abstraction for multi-tool dispatch and is normalized across both
  providers.
- **Add a tool-calling framework dependency.** Rejected (invariant 4): the
  OpenAI-compatible shape is a handful of dicts; `tools_base` already owns the
  tool abstraction.

## Consequences

Positive: the most unreliable step of the loop becomes structured on `run_turn`;
a smaller prompt when native calling is active; salience is tunable from
observation during dogfooding; the reliability harness makes the win measurable.

Negative / accepted: two code paths for tool calls (native + prose) until prose
can be retired — kept deliberately as the degradation path. The live streaming
path does not yet benefit from native calling; that is the first follow-up. The
reliability numbers require a local Ollama run to populate.
