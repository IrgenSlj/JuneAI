import { describe, expect, it } from "vitest";
import { dayGroup, groupByDay, snippetParts, sourceName } from "./threads.js";

const now = new Date(2026, 10, 21, 15, 0); // 21 November 2026, local time
const local = (y: number, m: number, d: number, h = 12) => new Date(y, m, d, h).toISOString();

describe("thread list helpers", () => {
  it("names day groups from local time", () => {
    expect(dayGroup(local(2026, 10, 21, 0), now)).toBe("Today");
    expect(dayGroup(local(2026, 10, 20, 23), now)).toBe("Yesterday");
    expect(dayGroup(local(2026, 10, 15), now)).toBe("Previous 7 days");
    expect(dayGroup(local(2026, 9, 25), now)).toBe("Previous 30 days");
    expect(dayGroup(local(2025, 2, 2), now)).toBe("March 2025");
    expect(dayGroup(undefined, now)).toBe("Undated");
    expect(dayGroup("garbage", now)).toBe("Undated");
  });

  it("groups consecutive items, preferring last activity", () => {
    const groups = groupByDay(
      [
        { id: "a", activity: local(2026, 10, 21) },
        { id: "b", started: local(2026, 10, 21) },
        { id: "c", activity: local(2025, 2, 2), started: local(2026, 10, 21) },
      ],
      now,
    );
    expect(groups.map((g) => [g.name, g.items.map((i) => i.id)])).toEqual([
      ["Today", ["a", "b"]],
      ["March 2025", ["c"]],
    ]);
  });

  it("splits snippets on the match markers", () => {
    expect(snippetParts("…Das Küchen\u0002budget\u0003 liegt\u0002 bei\u0003")).toEqual([
      { text: "…Das Küchen", match: false },
      { text: "budget", match: true },
      { text: " liegt", match: false },
      { text: " bei", match: true },
    ]);
    expect(snippetParts("<img onerror=x>")).toEqual([{ text: "<img onerror=x>", match: false }]);
    expect(sourceName("claude-export")).toBe("Claude");
    expect(sourceName("june")).toBeUndefined();
  });
});
