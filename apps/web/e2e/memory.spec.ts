import { test, expect } from "@playwright/test";
import { mockApi } from "./_mocks.js";

test("Memory page renders heading and search input with no uncaught errors", async ({
  page,
}) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  await mockApi(page, {
    memorySnapshot: {
      user_id: "default",
      goals: [],
      open_loops: [],
      calendar: [],
      journal: [],
      body_metrics: [],
      semantic_facts: [],
      entities: [],
      recent_messages: 0,
    },
    memoryStats: { user_id: "default", total: 0, buckets: [], last_write: "", recent_messages: 0, recent_facts: [] },
    forgotten: { user_id: "default", memories: [], count: 0 },
  });

  await page.goto("/memory");

  // The h1 always renders.
  await expect(page.getByRole("heading", { name: "Memory" })).toBeVisible();

  // The filter search input is always rendered in the header.
  await expect(page.getByLabel("Filter memories")).toBeVisible();

  expect(pageErrors, `Uncaught page errors: ${pageErrors.join(" | ")}`).toHaveLength(0);
});
