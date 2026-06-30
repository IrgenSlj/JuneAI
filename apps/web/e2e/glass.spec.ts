import { test, expect } from "@playwright/test";
import { mockApi } from "./_mocks.js";

test("Glass Box page renders heading and lede with no uncaught errors", async ({
  page,
}) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  await mockApi(page, { traces: { traces: [], count: 0 } });

  await page.goto("/system/glass");

  // The h1 always renders regardless of data.
  await expect(page.getByRole("heading", { name: "Glass Box" })).toBeVisible();

  // The lede paragraph is static HTML — always present regardless of trace data.
  await expect(page.getByText("Every step June takes", { exact: false })).toBeVisible();

  expect(pageErrors, `Uncaught page errors: ${pageErrors.join(" | ")}`).toHaveLength(0);
});
