import { test, expect } from "@playwright/test";
import { mockApi } from "./_mocks.js";

test("Trust page renders heading and link-cards for Glass Box, Receipts, Silence with no uncaught errors", async ({
  page,
}) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  // Provide all endpoints the Trust page polls on mount and every 5s.
  await mockApi(page, {
    activity: { entries: [], count: 0 },
    traces: { traces: [], count: 0 },
    capability: {
      summarization: "good",
      structured_output: "good",
      long_context: "good",
      relevance_scoring: "good",
      checked_at: "",
    },
    skills: { skills: [], count: 0 },
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
    tasks: { tasks: [], count: 0 },
  });

  await page.goto("/system");

  // The h1 "Trust" always renders.
  await expect(page.getByRole("heading", { name: "Trust" })).toBeVisible();

  // The three link-card headings must be present.
  await expect(page.getByRole("heading", { name: "Receipts" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Silence" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Glass Box" })).toBeVisible();

  expect(pageErrors, `Uncaught page errors: ${pageErrors.join(" | ")}`).toHaveLength(0);
});
