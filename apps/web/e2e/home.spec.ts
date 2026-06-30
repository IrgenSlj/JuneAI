import { test, expect } from "@playwright/test";
import { mockApi } from "./_mocks.js";

test("Home page renders ready state with no uncaught errors", async ({ page }) => {
  // Collect any uncaught JS errors thrown inside the page.
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  // Register all API interceptors before navigation so no real network
  // request escapes to 127.0.0.1:8099 (nothing listens there).
  await mockApi(page);

  await page.goto("/");

  // With zero open promises and nothing held, the page renders the
  // "clear" hero state. The h1.hero-summary carries "You're clear."
  const heroSummary = page.locator("h1.hero-summary");
  await expect(heroSummary).toBeVisible();
  await expect(heroSummary).toContainText("You're clear.");

  // No uncaught JS errors should have been thrown.
  expect(pageErrors, `Uncaught page errors: ${pageErrors.join(" | ")}`).toHaveLength(0);
});
