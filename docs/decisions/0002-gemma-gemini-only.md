# ADR 0002: Gemma 4 and Gemini as the Only Supported Models

**Status:** Accepted
**Date:** 2026-04-17

## Context

The v1 config exposes six model presets: Llama 3.2, Mistral (3 variants), Gemma 4, and Claude. Each preset has its own prompt formatting quirks, tool-calling behavior, and failure modes. The brain already carries branching code for Mistral-style "recovery" tool parsing versus OpenAI-native tool calls.

Supporting six model families multiplies every prompt-engineering decision, every test fixture, and every edge case. Users see a dropdown of unfamiliar names and must understand the tradeoffs between them. None of this serves the product goal.

## Decision

June supports exactly two model families:

- **Gemma 4** via Ollama for local inference. This is the default. It runs on consumer hardware, including Apple Silicon and modest GPUs, and delivers competitive quality for daily assistant work.
- **Gemini Flash** (and newer Gemini releases) via the Google AI Studio API. This is the cloud escape valve. Its free tier is the most generous in the industry (1M context, vision, code execution).

All other presets are removed from the config. Llama, Mistral, and Claude presets are deleted, including the shared `LOCAL_LARGE_MODEL_NAME` override and the Anthropic-specific code path.

## Consequences

**Positive:**

- A single prompt-engineering strategy applies across both models (both follow Google's Gemma-family instruction format).
- Test matrix collapses from six model paths to two.
- Users see a clear choice: local or cloud, both Google, both free.
- The settings UI becomes trivial: one toggle.
- Documentation, error messages, and onboarding all get shorter.

**Negative:**

- Users who prefer Claude or Llama lose first-class support. Acceptable because the brain package can be extended by anyone who wants to add a provider; the supported surface is deliberately narrow.
- Coupling to Google's model families means June benefits and suffers from Google's release cadence. Mitigated by the open-weight nature of Gemma — Google cannot unilaterally take it away.

## Alternatives Considered

**Support every model the user has installed via Ollama.** Rejected because it forces generic prompt engineering that underperforms model-specific prompts. Also increases support surface.

**Gemma 4 only, no cloud.** Rejected because many users do not have hardware to run Gemma 4 locally. Removing the cloud path excludes them from the product.

**Anthropic Claude as the cloud tier instead of Gemini.** Rejected because Claude has no free tier, and June's positioning is "free and open."

**Provider-agnostic abstraction layer.** Considered. Deferred. The brain package will define a clean `LLM` interface, but only `GemmaProvider` and `GeminiProvider` are implemented and tested. Others are a community extension point.
