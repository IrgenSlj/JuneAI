// From a route id and the user's key to a working Provider. The key is passed
// in by the caller (the daemon reads it from the keychain); nothing here stores it.
import { anthropicMessages } from "./anthropic.js";
import { openAICompatible } from "./openai-compatible.js";
import { ROUTE_PRESETS, type RoutePreset } from "./presets.js";
import type { FetchLike, Provider } from "./types.js";

export function findPreset(routeId: string): RoutePreset | undefined {
  return ROUTE_PRESETS.find((p) => p.route.id === routeId);
}

export function makeProvider(preset: RoutePreset, opts: { apiKey?: string; fetch: FetchLike }): Provider {
  if (preset.needsKey && (opts.apiKey === undefined || opts.apiKey === "")) {
    throw new Error(`${preset.route.name} needs an API key; add one in settings`);
  }
  if (preset.adapter === "anthropic") {
    return anthropicMessages({ route: preset.route, apiKey: opts.apiKey ?? "", fetch: opts.fetch, baseUrl: preset.baseUrl });
  }
  return openAICompatible({ route: preset.route, baseUrl: preset.baseUrl, fetch: opts.fetch, ...(opts.apiKey === undefined ? {} : { apiKey: opts.apiKey }) });
}
