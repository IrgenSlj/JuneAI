import { test, expect } from "@playwright/test";
import { MOCK_API_BASE, mockApi } from "./_mocks.js";

/** Radios are addressed by value, not label text: the dial's labels carry
 *  nested badges and copy that will be reworded, and the value is the thing
 *  the API actually receives. */
const dial = (page: import("@playwright/test").Page, value: string) =>
  page.locator(`input[type="radio"][value="${value}"]`);

test("/settings renders the provider and the privacy dial", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  await mockApi(page);
  await page.goto("/settings");

  await expect(page.getByRole("heading", { name: "Settings", level: 1 })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Model provider" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Privacy dial" })).toBeVisible();

  // All three positions are offered, and the one the backend reports is the
  // one selected. The dial is the control behind the product's central claim,
  // so "the section rendered" is not enough — it has to agree with the server.
  await expect(dial(page, "local_only")).toBeChecked();
  await expect(dial(page, "private_by_default")).not.toBeChecked();
  await expect(dial(page, "cloud_first")).not.toBeChecked();

  expect(pageErrors).toEqual([]);
});

test("/settings sends the dial change and confirms it saved", async ({ page }) => {
  const dialWrites: string[] = [];
  await mockApi(page);
  page.on("request", (req) => {
    if (req.url().includes("/settings/privacy-dial")) dialWrites.push(req.method());
  });

  await page.goto("/settings");
  await expect(dial(page, "local_only")).toBeChecked();

  await dial(page, "private_by_default").check();

  await expect(page.getByText("Saved.", { exact: true })).toBeVisible();
  expect(dialWrites).toContain("PUT");
  await expect(dial(page, "private_by_default")).toBeChecked();
});

test("/settings puts the dial back when the write fails", async ({ page }) => {
  await mockApi(page);
  // Registered after mockApi so it wins over the success route.
  await page.route(`${MOCK_API_BASE}/settings/privacy-dial`, (route) =>
    route.fulfill({ status: 500, contentType: "application/json", body: "{}" }),
  );

  await page.goto("/settings");
  await expect(dial(page, "local_only")).toBeChecked();

  // click(), not check(): check() asserts the radio ends up selected, and the
  // whole point here is that it does not — the page reverts on a failed write.
  await dial(page, "cloud_first").click();

  // A dial that looks changed but is not would be the worst failure this page
  // can have: the user would believe egress is blocked, or permitted, wrongly.
  await expect(page.getByText(/Couldn't update/)).toBeVisible();
  await expect(dial(page, "local_only")).toBeChecked();
  await expect(dial(page, "cloud_first")).not.toBeChecked();
});

test("/settings still renders when the backend is unreachable", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  const esc = MOCK_API_BASE.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  await mockApi(page);
  await page.route(new RegExp(`${esc}/settings(\\?|$)`), (route) =>
    route.fulfill({ status: 503, contentType: "application/json", body: "{}" }),
  );

  await page.goto("/settings");

  await expect(page.getByRole("heading", { name: "Settings", level: 1 })).toBeVisible();
  expect(pageErrors).toEqual([]);
});
