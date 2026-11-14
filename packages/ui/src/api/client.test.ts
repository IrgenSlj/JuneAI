import { describe, expect, it } from "vitest";
import { ApiError, createJuneClient } from "./client.js";
import { takeApiToken } from "./token.js";

function fakeFetch(status: number, body: unknown) {
  const calls: { url: string; init: RequestInit }[] = [];
  const impl = (async (url: string, init: RequestInit) => {
    calls.push({ url, init });
    return new Response(typeof body === "string" ? body : JSON.stringify(body), { status });
  }) as unknown as typeof fetch;
  return { impl, calls };
}

describe("client", () => {
  it("calls /api on the same origin with the token header", async () => {
    const { impl, calls } = fakeFetch(200, { path: null, defaultPath: "/Users/a/June" });
    const client = createJuneClient({ token: "t0k", fetchImpl: impl });
    expect(await client.getVault()).toEqual({ path: null, defaultPath: "/Users/a/June" });
    expect(calls[0]?.url).toBe("/api/vault");
    expect(new Headers(calls[0]?.init.headers).get("X-June-Token")).toBe("t0k");
  });

  it("surfaces the daemon's error message", async () => {
    const { impl } = fakeFetch(400, { error: "not a folder" });
    const error = await createJuneClient({ fetchImpl: impl })
      .openVault({ path: "/x" })
      .catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(400);
    expect((error as ApiError).reason).toBe("not a folder");
  });
});

describe("token handoff", () => {
  function env(hash: string, stored?: string) {
    const store = new Map<string, string>(stored ? [["june.token", stored]] : []);
    const state = { cleared: false };
    return {
      state,
      store,
      env: {
        hash,
        storage: { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => void store.set(k, v) },
        clearHash: () => void (state.cleared = true),
      },
    };
  }

  it("moves the token from the fragment into session storage and clears the URL", () => {
    const t = env("#token=abc123");
    expect(takeApiToken(t.env)).toBe("abc123");
    expect(t.store.get("june.token")).toBe("abc123");
    expect(t.state.cleared).toBe(true);
  });

  it("falls back to the stored token after a reload", () => {
    const t = env("", "abc123");
    expect(takeApiToken(t.env)).toBe("abc123");
    expect(t.state.cleared).toBe(false);
  });

  it("has no token when neither is present", () => {
    expect(takeApiToken(env("").env)).toBeUndefined();
  });
});
