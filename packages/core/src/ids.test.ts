import { describe, expect, it } from "vitest";
import { createIdGenerator, idTime, isId, newId } from "./ids.js";

describe("ids", () => {
  it("makes prefixed ULIDs that carry their creation time", () => {
    const gen = createIdGenerator({ now: () => 1_731_600_000_000, random80: () => 0n });
    const id = gen("thread");
    expect(id).toMatch(/^th_[0-9A-Z]{26}$/);
    expect(isId(id, "thread")).toBe(true);
    expect(isId(id, "note")).toBe(false);
    expect(idTime(id)).toBe(1_731_600_000_000);
  });

  it("stays sortable within one millisecond and across a clock step back", () => {
    let t = 1_000;
    const gen = createIdGenerator({ now: () => t, random80: () => 5n });
    const a = gen("note");
    const b = gen("note");
    t = 999; // clock moved backwards
    const c = gen("note");
    t = 1_001;
    const d = gen("note");
    expect([a, b, c, d]).toEqual([a, b, c, d].slice().sort());
    expect(new Set([a, b, c, d]).size).toBe(4);
  });

  it("rejects malformed ids", () => {
    expect(isId("th_123")).toBe(false);
    expect(isId("xx_01JD3K9V2Q8XN4M6P0R7S5T1WB")).toBe(false);
    expect(isId("th_01JD3K9V2Q8XN4M6P0R7S5T1WI")).toBe(false); // I is not Crockford
    expect(() => idTime("nope")).toThrow(TypeError);
  });

  it("generates unique ids from the system source", () => {
    const ids = Array.from({ length: 1000 }, () => newId("plan"));
    expect(new Set(ids).size).toBe(1000);
    expect(ids).toEqual(ids.slice().sort());
  });
});
