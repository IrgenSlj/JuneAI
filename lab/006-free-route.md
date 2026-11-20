# 006 What should "free" mean in June now that OpenCode Zen's free models are closed to it?

## Question

Lab 005 found OpenCode Zen's free models work only inside OpenCode. Is there still a free way to start using June that keeps the label honest?

## Hypothesis

OpenRouter's `:free` models replace Zen with no signup cost, and are as easy to start with.

## Cheapest experiment

Read OpenRouter's limits, FAQ and privacy pages (2026-11-20) and compare with what June already does for local models.

## Result

Surprise first: **OpenRouter's free models are not a no-signup path.** An account and an API key are required; the limit is 50 requests a day without buying credits, 1,000 with at least 10 credits bought ("If you're using a free model variant (with an ID ending in `:free`), the following limits apply", https://openrouter.ai/docs/api-reference/limits). Training depends on the provider behind each free model, and there are "separate settings for paid and free models" for routing to providers that may train (https://openrouter.ai/docs/features/privacy-and-logging). The docs do not list which free models are served only by providers that do not train.

| Option | Signup | Cost | Leaves the Mac | Training risk | Already in June |
| --- | --- | --- | --- | --- | --- |
| Local (Ollama, LM Studio) | none | free | no | none | yes, and now shown as ready only when running |
| OpenRouter `:free` models | account and key | free, 50/day | yes | depends on provider; unclear per model | yes, through the OpenRouter route with a key |
| OpenCode Zen free | OpenCode's app only | free | yes | some models train | not possible |

## Decision

Proposed for the founder: drop "free gateway" as a separate first-run promise. First run offers two doors: "Run a model on this Mac" (detects Ollama or LM Studio, links to their installers) and "Use a key" (any provider, OpenRouter included, whose free models then work through the same route). The label already states "leaves this Mac" and the data note says OpenRouter forwards to providers with their own policies. No code changes until the founder decides; the route kind `gateway` stays for paid Zen.
