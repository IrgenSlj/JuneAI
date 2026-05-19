# ADR 0009: Private-by-default with three-tier model routing

Status: Proposed (supersedes the absolute reading of [ADR 0002](0002-gemma-gemini-only.md))
Date: 2026-05-18

## Context

[ADR 0002](0002-gemma-gemini-only.md) committed June to two model providers: Gemma 4 via local Ollama and Gemini via Google AI Studio. The accompanying framing in `docs/vision.md` Non-Negotiable #2 read "local-first, cloud as escape valve" and was applied as an absolute: every conversational turn defaulted to Gemma, and Gemini was a user-toggled fallback.

Two facts collided with that framing during the May 2026 strategic review:

1. **Local Gemma is not capable enough for agentic work on consumer hardware.** Multi-step planning, structured tool-call recovery, long-context reasoning (a long email thread, a PDF, a screenshot), and reliable JSON-shaped output are all materially worse on a 4B-parameter local model than on a frontier cloud model. The gap is widening in 2026, not closing. June's pivot toward agentic capability (see [ADR 0010](0010-agentic-core-tasks-oauth-computer-use.md)) cannot be served by local-only inference without shipping a product that is visibly slow and frequently wrong.
2. **Mainstream users do not want a binary choice.** They want a chat to feel snappy and private, *and* they want an agent to actually finish the task. Apple Intelligence solved this by routing small interactions to a local model, harder ones to Private Cloud Compute, and the hardest to ChatGPT with explicit consent. That pattern works because it matches the user's mental model of "small is fast and private, big is slow and powerful."

The original "local-first, always" framing remains correct as a *default*, but is wrong as an *absolute*. Treating it as absolute forced a false choice: either accept a weak agent, or break the principle quietly and lose the credibility that comes with it.

## Decision

June adopts **private-by-default with three-tier model routing.**

The three tiers:

1. **Local (Gemma via Ollama)** — the default for chat tone, memory recall, classification, short summarisation, journaling, and any turn where the user has set a "local-only" policy. No data leaves the machine. This tier is also the only tier available when Ollama is reachable and the user has set their privacy dial to local-only.
2. **Cloud-on-consent (Gemini)** — used for agentic planning, multi-step tool use, long context (>4k tokens of input), vision, computer-use screenshots, and any skill whose declared policy requires it. Each invocation is visible in the UI before and after it happens. The user can pause, cancel, or downgrade the turn.
3. **Per-skill policy** — every skill manifest declares a `model_policy` field: `local-only`, `cloud-allowed`, or `cloud-required`. The router resolves the effective tier per tool call, not per turn. A single turn can mix local recall, local planning, and one cloud-required tool call, and the UI shows exactly that.

The user controls the policy with three dial positions, surfaced once at first run and editable in `/settings`:

- **Local-only** — never call cloud. Skills that require cloud are disabled with a visible explanation. June behaves like June 1.0 for users who want that.
- **Private-by-default** *(default)* — chat and recall are local; agentic capability is allowed to call cloud per the per-skill policy, with confirmation on the first call of each kind per session.
- **Cloud-first** — prefer cloud for capability, fall back to local only when offline. For users who do not have a local model installed.

Transient cloud calls are not logged on Google's side beyond the request itself; no training, no retention beyond Google's stated retention for the API. The user-visible privacy label in the system header reflects the *current effective tier*, not the user's policy.

## Why this is not a betrayal of the local-first promise

The promise that matters to users is "**my data is mine and I see what leaves my machine.**" That promise is preserved:

- Memory stays local. Every recall, every extraction, every write — local.
- Conversations stay local at rest. Cloud calls send the *turn's* context, not the archive.
- The user sees, per call, what went where. Provenance is rendered inline on every assistant message and every tool result.
- The dial exists. A user who wants pure local-only can set it once and never see cloud.

What is being changed is the unspoken second promise — "you can do everything locally" — which was always partially false (Gemini was already in the product) and which we now make explicit.

## Alternatives considered

- **Hold the line: ship a local-only agent.** Rejected. The capability gap is too large in 2026 for this to produce an agent worth using. We would ship a product whose first-impression demo is its weakest moment.
- **Flip to cloud-first.** Rejected. The differentiator is memory ownership and local default, not raw capability. We would compete head-on with ChatGPT on its strongest axis.
- **Bring a third local model (Llama, Mistral) to close the gap.** Rejected. [ADR 0002](0002-gemma-gemini-only.md) is right about the cost of provider proliferation. No third model closes the agentic gap anyway; that gap is about scale and training data, not architecture.
- **Use Private Cloud Compute-style attestation.** Out of scope. Apple owns the silicon-to-server attestation chain; we do not. We trust Google's stated terms and surface them honestly to the user.

## Consequences

**Positive:**

- Agentic skills become possible without lying about model choice.
- The product can be marketed honestly: "private chat and memory by default, with optional cloud intelligence when you ask June to do real work."
- The per-skill policy field gives third-party skill authors a way to declare their requirements without hardcoding them into the brain.

**Negative:**

- The header UI must do more work. A single label is no longer sufficient; we need per-message provenance.
- Skills now have a policy field that did not exist. All existing skill manifests need a one-line addition.
- `docs/vision.md` Non-Negotiable #2 needs to be rewritten. This is the most visible breaking change and will be done in the same PR that lands this ADR.

**Carried forward:**

- "No third model" still holds. Gemma and Gemini are the two providers. The third tier ("per-skill policy") is a *routing* concept, not a new provider.
- "No account by default" still holds. The user can still run June with zero cloud calls if they set the dial to local-only.

## Status of related ADRs

- [ADR 0002](0002-gemma-gemini-only.md) is **amended in spirit, not superseded.** The two-provider list still stands; the "local-first as absolute default" framing is replaced by what this ADR specifies. A note pointing to this ADR will be added at the top of 0002.
- [ADR 0008](0008-ollama-supervision.md) is **unaffected.** Local Ollama supervision remains the way we install and start Gemma for the local tier.
