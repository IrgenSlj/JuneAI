import { findPreset } from "@june/providers";
import { describe, expect, it } from "vitest";
import { ollamaFromEnv, relocate } from "./local-routes.js";

describe("OLLAMA_HOST", () => {
  it("accepts loopback addresses with or without a scheme", () => {
    expect(ollamaFromEnv("127.0.0.1:11500")).toBe("http://127.0.0.1:11500/v1");
    expect(ollamaFromEnv("http://localhost:11500")).toBe("http://localhost:11500/v1");
    expect(ollamaFromEnv(undefined)).toBeUndefined();
    expect(ollamaFromEnv("")).toBeUndefined();
  });

  it("ignores anything that would leave this Mac, so the label stays true", () => {
    expect(ollamaFromEnv("0.0.0.0:11434")).toBeUndefined();
    expect(ollamaFromEnv("192.168.1.20:11434")).toBeUndefined();
    expect(ollamaFromEnv("http://ollama.example.com")).toBeUndefined();
    expect(ollamaFromEnv("http://localhost.evil.com:11434")).toBeUndefined();
  });

  it("moves the preset's host for the label and keeps it local", () => {
    const ollama = findPreset("ollama");
    if (ollama === undefined) throw new Error("no ollama preset");
    const moved = relocate(ollama, "http://127.0.0.1:11500/v1");
    expect(moved.route).toMatchObject({ host: "127.0.0.1:11500", leavesDevice: false, kind: "local" });
    expect(relocate(ollama, undefined)).toBe(ollama);
  });
});
