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

const UNKNOWN = "unknown";

const remote = (id: string, name: string, host: string, kind: "api-key" | "gateway" = "api-key"): Route => ({
  id,
  name,
  kind,
  host,
  leavesDevice: true,
  billing: kind === "gateway" ? "free" : "your API key",
  // Filled in only from the provider's own page, with its URL and the date checked.
  dataNote: UNKNOWN,
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
  { route: remote("openai", "OpenAI", "api.openai.com"), adapter: "openai-compatible", baseUrl: "https://api.openai.com/v1", needsKey: true },
  { route: remote("anthropic", "Anthropic", "api.anthropic.com"), adapter: "anthropic", baseUrl: "https://api.anthropic.com/v1", needsKey: true },
  { route: remote("mistral", "Mistral", "api.mistral.ai"), adapter: "openai-compatible", baseUrl: "https://api.mistral.ai/v1", needsKey: true },
  { route: remote("openrouter", "OpenRouter", "openrouter.ai"), adapter: "openai-compatible", baseUrl: "https://openrouter.ai/api/v1", needsKey: true },
  { route: remote("opencode-zen", "OpenCode Zen", "opencode.ai", "gateway"), adapter: "openai-compatible", baseUrl: "https://opencode.ai/zen/v1", needsKey: true },
];
