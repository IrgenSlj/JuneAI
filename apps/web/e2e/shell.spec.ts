// M0 "done when": the UI opens against a running daemon, picks a vault folder
// and shows the chat page empty; a request without the token is refused.
import { stat } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { type Daemon, startDaemon } from "./daemon.js";

let daemon: Daemon;
const cspViolations: string[] = [];

test.beforeAll(async () => {
  // Port 9 (discard) has no Ollama, on this Mac or in CI, so the empty state is the same everywhere.
  daemon = await startDaemon({ env: { OLLAMA_HOST: "127.0.0.1:9" } });
});
test.afterAll(() => daemon.stop());

test.beforeEach(({ page }) => {
  page.on("console", (msg) => {
    if (/Content Security Policy|Refused to/i.test(msg.text())) cspViolations.push(msg.text());
  });
});

test("without the token the page explains how to open June, and the API refuses", async ({ page, request }) => {
  await page.goto(`${daemon.origin}/`);
  await expect(page.getByText("needs the link June printed")).toBeVisible();
  expect((await request.get(`${daemon.origin}/api/vault`)).status()).toBe(401);
});

test("first run: pick a vault, land on an empty chat, keep the session on reload", async ({ page }) => {
  await page.goto(`${daemon.origin}/#token=${daemon.token}`);

  await expect(page.getByRole("heading", { name: "Where should June keep your conversations?" })).toBeVisible();
  expect(page.url()).not.toContain("token=");

  const vault = join(daemon.root, "My Vault");
  await page.getByLabel("Vault folder").fill(vault);
  await page.getByRole("button", { name: "Use this folder" }).click();

  await expect(page).toHaveURL(/\/chat$/);
  await expect(page.getByText("No model is connected. Start Ollama or LM Studio on this Mac, or add an API key in Settings.")).toBeVisible();
  await expect(page.getByText(vault)).toBeVisible();
  expect((await stat(vault)).isDirectory()).toBe(true);

  await page.reload();
  await expect(page.getByText("No model is connected")).toBeVisible();
});

test("the page runs under the strict CSP without a single violation", () => {
  expect(cspViolations).toEqual([]);
});
