import { test, expect } from "@playwright/test";
import { mockApi } from "./_mocks.js";

test("Promises page renders heading and task-creation form with no uncaught errors", async ({
  page,
}) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  await mockApi(page, { tasks: { tasks: [], count: 0 } });

  await page.goto("/tasks");

  // The h1 "Promises" always renders.
  await expect(page.getByRole("heading", { name: "Promises" })).toBeVisible();

  // The task-creation input is always rendered regardless of task list state.
  await expect(page.locator("#task-goal")).toBeVisible();

  expect(pageErrors, `Uncaught page errors: ${pageErrors.join(" | ")}`).toHaveLength(0);
});
