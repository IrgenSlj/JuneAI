// Route presets hold only facts that were checked. A preset with no checked
// data practice says "unknown" rather than guess. Remote presets arrive with
// their checked sources in a later M1 slice.
import type { Route } from "./types.js";

export const OLLAMA: Route = {
  id: "ollama",
  name: "Ollama (this Mac)",
  kind: "local",
  host: "localhost:11434",
  leavesDevice: false,
  billing: "none (local)",
  dataNote: "Runs on this Mac. Nothing is sent anywhere.",
};

export const LM_STUDIO: Route = {
  id: "lm-studio",
  name: "LM Studio (this Mac)",
  kind: "local",
  host: "localhost:1234",
  leavesDevice: false,
  billing: "none (local)",
  dataNote: "Runs on this Mac. Nothing is sent anywhere.",
};

/** Base URLs of the OpenAI-compatible endpoints the local presets use. */
export const LOCAL_BASE_URLS: Record<string, string> = {
  ollama: "http://localhost:11434/v1",
  "lm-studio": "http://localhost:1234/v1",
};

const CHECKED = "2026-11-15";

const remote = (
  id: string,
  name: string,
  host: string,
  dataNote: string,
  sourceUrl: string,
  kind: "api-key" | "gateway" = "api-key",
): Route => ({
  id,
  name,
  kind,
  host,
  leavesDevice: true,
  billing: "your API key",
  // One sentence from the provider's own page, with its URL and the date checked (lab 005).
  dataNote,
  dataNoteSource: { url: sourceUrl, checked: CHECKED },
});

/** How June reaches a route: which adapter, where, and whether it needs a key. */
export interface RoutePreset {
  route: Route;
  adapter: "openai-compatible" | "anthropic";
  baseUrl: string;
  needsKey: boolean;
}

export const ROUTE_PRESETS: readonly RoutePreset[] = [
  { route: OLLAMA, adapter: "openai-compatible", baseUrl: "http://localhost:11434/v1", needsKey: false },
  { route: LM_STUDIO, adapter: "openai-compatible", baseUrl: "http://localhost:1234/v1", needsKey: false },
  {
    route: remote(
      "openai",
      "OpenAI",
      "api.openai.com",
      "OpenAI says API data is not used to train its models, and keeps abuse-monitoring logs for up to 30 days.",
      "https://developers.openai.com/api/docs/guides/your-data",
    ),
    adapter: "openai-compatible",
    baseUrl: "https://api.openai.com/v1",
    needsKey: true,
  },
  {
    route: remote(
      "anthropic",
      "Anthropic",
      "api.anthropic.com",
      "Anthropic says API inputs and outputs are not used for training by default and are deleted within 30 days, with exceptions such as policy enforcement.",
      "https://privacy.claude.com/en/articles/7996866-how-long-do-you-store-my-organization-s-data",
    ),
    adapter: "anthropic",
    baseUrl: "https://api.anthropic.com/v1",
    needsKey: true,
  },
  {
    route: remote(
      "mistral",
      "Mistral",
      "api.mistral.ai",
      "Mistral's terms rule out training unless you opt in or leave an opt-in-by-default product on; check the training toggle in its Admin panel. Retention is not stated.",
      "https://legal.mistral.ai/terms/commercial-terms-of-service",
    ),
    adapter: "openai-compatible",
    baseUrl: "https://api.mistral.ai/v1",
    needsKey: true,
  },
  {
    route: remote(
      "openrouter",
      "OpenRouter",
      "openrouter.ai",
      "OpenRouter says it does not train on your inputs, but forwards each request to a provider whose own logging and training policy applies; its privacy settings can exclude providers that train.",
      "https://openrouter.ai/docs/features/privacy-and-logging",
    ),
    adapter: "openai-compatible",
    baseUrl: "https://openrouter.ai/api/v1",
    needsKey: true,
  },
  {
    route: remote(
      "opencode-zen",
      "OpenCode Zen",
      "opencode.ai",
      "OpenCode says training and retention vary by model: most providers keep nothing and do not train, but some retain data and may train on it.",
      "https://opencode.ai/docs/zen",
      "gateway",
    ),
    adapter: "openai-compatible",
    // Free models work only inside OpenCode's own app (lab 005), so this route uses paid models.
    baseUrl: "https://opencode.ai/inference/openai/v1",
    needsKey: true,
  },
];
