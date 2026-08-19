import { test, expect } from "@playwright/test";
import { MOCK_API_BASE, REGISTRY_FIXTURE, SKILLS_POPULATED_FIXTURE, mockApi } from "./_mocks.js";

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

// --- MCP registry -----------------------------------------------------------
// Coverage added ahead of extracting this section into packages/ui, so the move
// has something to fail against. Installing a third-party MCP server is the
// most consequential thing this page does.
//
// Scoped through the section's aria-label rather than its CSS classes: the
// point of these specs is to survive the markup being moved, and a selector
// tied to `.registry-entry` would have to be rewritten by the same change it
// is supposed to be checking.
const registry = (page: import("@playwright/test").Page) =>
  page.getByRole("region", { name: "MCP registry" });

test("/skills registry lists entries with their trust signals", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  await mockApi(page, { skills: SKILLS_POPULATED_FIXTURE });
  await page.goto("/skills");

  const section = registry(page);
  await expect(section.getByRole("heading", { name: "Browse the MCP registry" })).toBeVisible();
  await expect(section.getByText("Weather")).toBeVisible();
  await expect(section.getByText("Research")).toBeVisible();

  // "verified" is the signal a user leans on before running someone else's
  // code, so it has to be attached to one entry and absent from the other —
  // a badge that renders for everything says nothing.
  await expect(section.getByText("verified", { exact: true })).toHaveCount(1);

  expect(pageErrors).toEqual([]);
});

test("/skills registry offers Install for absent entries and Uninstall for present ones", async ({ page }) => {
  await mockApi(page, { skills: SKILLS_POPULATED_FIXTURE });
  await page.goto("/skills");

  const section = registry(page);
  await expect(section.getByRole("heading", { name: "Browse the MCP registry" })).toBeVisible();

  // exact: true — accessible-name matching is substring by default, and
  // "Uninstall" contains "Install".
  await expect(section.getByRole("button", { name: "Install", exact: true })).toHaveCount(1);
  await expect(section.getByRole("button", { name: "Uninstall", exact: true })).toHaveCount(1);
});

test("/skills registry surfaces a load failure without breaking the page", async ({ page }) => {
  const esc = MOCK_API_BASE.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  await mockApi(page, { skills: SKILLS_POPULATED_FIXTURE });
  await page.route(new RegExp(`${esc}/skills/registry`), (route) =>
    route.fulfill({ status: 500, contentType: "application/json", body: "{}" }),
  );

  await page.goto("/skills");

  await expect(page.getByText(/Registry failed to load/)).toBeVisible();
  // The installed-skills list above it must still be usable.
  await expect(page.getByText("research", { exact: true })).toBeVisible();
});
