import { test, expect } from "@playwright/test";
import { mockApi } from "./_mocks.js";

test("Receipts page renders heading and empty ledger with no uncaught errors", async ({
  page,
}) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  // Empty ledger + /system returns null ledger_summary (already default in SYSTEM_FIXTURE).
  await mockApi(page, { ledger: { entries: [], count: 0, next_cursor: null } });

  await page.goto("/system/receipts");

  // The h1 always renders.
  await expect(page.getByRole("heading", { name: "Receipts" })).toBeVisible();

  // With no entries, the empty-state hint appears.
  await expect(page.getByText("The ledger is empty")).toBeVisible();

  expect(pageErrors, `Uncaught page errors: ${pageErrors.join(" | ")}`).toHaveLength(0);
});

// Canary: prove the mock layer actually drives the UI, not just that the page
// renders. If the mock were dead (e.g. pattern mismatch), entries would be empty
// and "Ran write_memory" would never appear.
test("Receipts page renders a mocked ledger entry (canary)", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  // One action entry — title() produces "Ran write_memory" for kind=action + tool payload.
  await mockApi(page, {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ledger: {
      entries: [
        {
          seq: 1,
          id: "canary-entry-1",
          ts: "2024-01-15T10:00:00Z",
          kind: "action",
          actor: "june",
          payload: { tool: "write_memory", action_class: "local" },
          prev_hash: "0".repeat(64),
          entry_hash: "a1b2c3d4e5f6a7b8",
        },
      ] as any,
      count: 1,
      next_cursor: null,
    },
  });

  await page.goto("/system/receipts");

  await expect(page.getByRole("heading", { name: "Receipts" })).toBeVisible();

  // The entry title derived from the mocked payload must be visible, confirming
  // the page.route() intercept is live and the mock data drives the rendered list.
  await expect(page.getByText("Ran write_memory")).toBeVisible();

  expect(pageErrors, `Uncaught page errors: ${pageErrors.join(" | ")}`).toHaveLength(0);
});
