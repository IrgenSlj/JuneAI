// Where local routes listen, when the user moved them. Ollama's own
// OLLAMA_HOST ("127.0.0.1:11500" or "http://localhost:11500") is honoured so
// June finds Ollama wherever the user told Ollama to listen. Only loopback
// addresses count: a local route must stay on this Mac, or its label would lie.
import type { RoutePreset } from "@june/providers";

const LOOPBACK = new Set(["127.0.0.1", "localhost", "[::1]"]);

/** The base URL for Ollama from OLLAMA_HOST, or undefined if unset or not on this Mac. */
export function ollamaFromEnv(value: string | undefined): string | undefined {
  if (value === undefined || value.trim() === "") return undefined;
  const raw = value.trim();
  let url: URL;
  try {
    url = new URL(/^https?:\/\//.test(raw) ? raw : `http://${raw}`);
  } catch {
    return undefined;
  }
  if (!LOOPBACK.has(url.hostname)) return undefined;
  return `${url.protocol}//${url.host}/v1`;
}

/** The preset with its base URL and host moved, keeping every other fact. */
export function relocate(preset: RoutePreset, baseUrl: string | undefined): RoutePreset {
  if (baseUrl === undefined) return preset;
  return { ...preset, baseUrl, route: { ...preset.route, host: new URL(baseUrl).host } };
}
