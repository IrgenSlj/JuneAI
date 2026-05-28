# ADR 0017 — Model-Specific Provider Layer

## Status

Accepted. Reinforces ADR 0002 (Gemma 4 and Gemini only) and ADR 0009 (private by
default and model routing), and reverses the earlier "provider-agnostic /
OpenAI-compatible abstraction" stance carried in the model-client construction.
Anchored by the [build specification](../product/build-spec.md), C.1.

## Context

June's brain reaches models today through an OpenAI-compatible client construction
that treats Ollama/Gemma and Gemini as interchangeable endpoints. The implicit goal
was provider-agnosticism: a thin, generic interface so a new model could be dropped
in with minimal change.

That goal is wrong for an agent harness. The thing that makes a harness good — a
coding agent or a personal one — is that it is *tuned* for the specific model it
runs: prompt formatting, stop sequences, JSON-mode behavior, context-window
handling, streaming quirks, temperature defaults, and how aggressively to lean on
the model's judgment for compaction or salience. A lowest-common-denominator
abstraction actively blocks that tuning. June already committed to exactly two
models (ADR 0002); pretending to be model-agnostic buys nothing and costs tuning.

## Decision

June has a **model-specific provider layer**, deeply tuned for a known roster of
exactly two models, with a clean seam for a third.

- **Providers, not a generic client.** `packages/brain/june_brain/providers/` holds
  `base.py` (a `Provider` Protocol plus `GenerateRequest` / `GenerateResult` /
  `ProviderHealth` Pydantic models), `gemma.py` (Ollama HTTP, streaming),
  `gemini.py` (official Google client, streaming), and `registry.py`.
- **Roles, resolved from config.** The brain references *roles* — `local-fast`,
  `local-deep` (Gemma 4 configurations), and `cloud-capable` (Gemini). A
  `config/providers.toml` names the concrete models. Adding a third model is a new
  provider file plus a config line, with no brain changes — that is the seam,
  distinct from a generic abstraction.
- **One path to models.** All model access goes through a provider. No raw Ollama or
  HTTP model call lives anywhere else in the brain.
- **Provenance and health are first-class.** Every cloud (Gemini) call emits a
  provenance event before and after (ADR 0016's transparency companion / build-spec
  C.6). `health()` lets callers detect an unreachable or unloaded model and surface
  it rather than hang.

`GenerateResult` carries the concrete `model_id` (e.g. `gemma4:e2b`) and the `tier`
so provenance and the difficulty-routed tier selection (build-spec C.6) are exact.

## Alternatives Considered

- **Keep the OpenAI-compatible generic client.** Rejected. It optimizes for a
  flexibility June does not want (arbitrary providers) at the cost of the tuning
  June does want.
- **A plugin system for arbitrary providers.** Rejected. It contradicts ADR 0002
  (two models only) and the "no third model" boundary; the role seam already covers
  the one realistic case (replacing a model).

## Consequences

Positive: each model can be tuned to its strengths; provenance is precise; the
role→model indirection keeps the brain stable while config evolves; mock providers
make the loop and routing testable.

Negative: supporting a genuinely new provider takes real work (a tuned provider
file), by design; the two providers must each be maintained against upstream API
changes.

## References

- [build-spec.md](../product/build-spec.md) — C.1, Part A Principle 6
- ADR 0002 — Gemma 4 and Gemini as the only supported models
- ADR 0009 — Private by Default and Model Routing
- ADR 0016 — Event-Driven Proactivity; No Heartbeat (provenance companion)
