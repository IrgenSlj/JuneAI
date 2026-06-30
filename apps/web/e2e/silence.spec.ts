import { test, expect } from "@playwright/test";
import { mockApi } from "./_mocks.js";

test("Silence page renders heading and empty decisions with no uncaught errors", async ({
  page,
}) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  // Empty surfacing page.
  await mockApi(page, { surfacing: { entries: [], count: 0, next_cursor: null } });

  await page.goto("/system/silence");

  // The h1 always renders.
  await expect(page.getByRole("heading", { name: "Silence" })).toBeVisible();

  // With no decisions, the empty-state hint appears.
  await expect(page.getByText("No decisions recorded yet")).toBeVisible();

  expect(pageErrors, `Uncaught page errors: ${pageErrors.join(" | ")}`).toHaveLength(0);
});
