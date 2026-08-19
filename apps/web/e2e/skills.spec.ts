import { test, expect } from "@playwright/test";
import { MOCK_API_BASE, SKILLS_POPULATED_FIXTURE, mockApi } from "./_mocks.js";

test("/skills lists installed skills with their capability contracts", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  await mockApi(page, { skills: SKILLS_POPULATED_FIXTURE });
  await page.goto("/skills");

  await expect(page.getByRole("heading", { name: "Skills", level: 1 })).toBeVisible();
  await expect(page.getByText("research", { exact: true })).toBeVisible();
  await expect(page.getByText("calendar", { exact: true })).toBeVisible();
  await expect(page.getByText("1 of 2 enabled")).toBeVisible();

  // The declared scopes are the point of this page: a skill that can send data
  // off the device has to say so where the user is deciding whether to enable
  // it, not only in a manifest they will never open.
  await expect(page.getByText("sends data off device")).toBeVisible();
  await expect(page.getByText("reads local data")).toBeVisible();

  expect(pageErrors).toEqual([]);
});

test("/skills toggling a skill calls the API", async ({ page }) => {
  const toggles: string[] = [];
  await mockApi(page, { skills: SKILLS_POPULATED_FIXTURE });
  page.on("request", (req) => {
    const m = req.url().match(/\/skills\/([^/?]+)\/toggle/);
    if (m) toggles.push(`${req.method()} ${m[1]}`);
  });

  await page.goto("/skills");
  await expect(page.getByText("research", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Disable skill" }).click();

  await expect.poll(() => toggles).toContain("POST research");
});

test("/skills surfaces a failed toggle instead of silently reverting", async ({ page }) => {
  await mockApi(page, { skills: SKILLS_POPULATED_FIXTURE });
  // Registered after mockApi so it wins over the success route.
  await page.route(new RegExp(`${MOCK_API_BASE.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}/skills/[^/?]+/toggle`), (route) =>
    route.fulfill({ status: 500, contentType: "application/json", body: "{}" }),
  );

  await page.goto("/skills");
  await expect(page.getByText("research", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Disable skill" }).click();

  await expect(page.getByText(/Couldn't update that skill/)).toBeVisible();
});

test("/skills says so when nothing is installed", async ({ page }) => {
  await mockApi(page);
  await page.goto("/skills");

  await expect(page.getByText("No skills installed.")).toBeVisible();
});

test("/skills still renders when the backend is unreachable", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  await mockApi(page);
  await page.route(new RegExp(`${MOCK_API_BASE.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}/skills`), (route) =>
    route.fulfill({ status: 503, contentType: "application/json", body: "{}" }),
  );

  await page.goto("/skills");

  await expect(page.getByRole("heading", { name: "Skills", level: 1 })).toBeVisible();
  expect(pageErrors).toEqual([]);
});
