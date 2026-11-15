import { describe, expect, it } from "vitest";
import { ROUTE_PRESETS } from "./presets.js";
import { findPreset, makeProvider } from "./registry.js";

describe("route presets", () => {
  it("have unique ids, and only local routes stay on this Mac", () => {
    const ids = ROUTE_PRESETS.map((p) => p.route.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const { route, baseUrl, needsKey } of ROUTE_PRESETS) {
      const local = route.kind === "local";
      expect(route.leavesDevice, route.id).toBe(!local);
      expect(new URL(baseUrl).hostname === "localhost", route.id).toBe(local);
      expect(needsKey, route.id).toBe(!local);
      if (!local) expect(baseUrl.startsWith("https://"), route.id).toBe(true);
    }
  });

  it("never state a data practice without a checked source", () => {
    for (const { route } of ROUTE_PRESETS) {
      if (route.kind !== "local" && route.dataNote !== "unknown") {
        expect(route.dataNoteSource?.url, route.id).toMatch(/^https:\/\//);
        expect(route.dataNoteSource?.checked, route.id).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      }
    }
  });

  it("refuse to build a keyed route without a key", () => {
    const fetch = async () => {
      throw new Error("no network in tests");
    };
    const openai = findPreset("openai");
    expect(openai).toBeDefined();
    if (openai === undefined) return;
    expect(() => makeProvider(openai, { fetch })).toThrow(/needs an API key/);
    expect(makeProvider(openai, { apiKey: "sk-1", fetch }).route.id).toBe("openai");
    const ollama = findPreset("ollama");
    if (ollama !== undefined) expect(makeProvider(ollama, { fetch }).route.kind).toBe("local");
  });
});
