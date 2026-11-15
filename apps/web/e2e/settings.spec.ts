// Keys: saved through the settings page into the secret store (memory in
// tests, the macOS keychain in the app), never shown again, never on disk.
import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { type Daemon, startDaemon } from "./daemon.js";

let daemon: Daemon;
const KEY = "sk-e2e-secret-0123456789";

test.beforeAll(async () => {
  daemon = await startDaemon();
});
test.afterAll(() => daemon.stop());

async function everyFile(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { recursive: true, withFileTypes: true });
  return entries.filter((e) => e.isFile()).map((e) => join(e.parentPath, e.name));
}

/** Open June with a vault, from a fresh page, and go to Settings. */
async function openSettings(page: import("@playwright/test").Page): Promise<void> {
  await page.goto(`${daemon.origin}/#token=${daemon.token}`);
  await expect(page).toHaveURL(/\/(setup|chat)$/);
  if (page.url().endsWith("/setup")) {
    await page.getByLabel("Vault folder").fill(join(daemon.root, "Vault"));
    await page.getByRole("button", { name: "Use this folder" }).click();
  }
  await expect(page).toHaveURL(/\/chat$/);
  await page.getByRole("link", { name: "Settings" }).click();
  await expect(page).toHaveURL(/\/settings$/);
}

test("a key saved in settings makes the route usable, is never shown, and never touches disk", async ({ page }) => {
  await openSettings(page);
  const openai = page.getByRole("article", { name: "OpenAI" });
  await expect(openai.getByText("No key", { exact: true })).toBeVisible();
  await expect(openai.getByText("sends to api.openai.com")).toBeVisible();
  await openai.getByLabel("OpenAI API key").fill(KEY);
  await openai.getByRole("button", { name: "Save" }).click();
  await expect(openai.getByText("Key saved in the macOS keychain.")).toBeVisible();
  await expect(openai.getByText("Key stored")).toBeVisible();
  await expect(openai.getByLabel("OpenAI API key")).toHaveValue("");
  expect(await page.content()).not.toContain(KEY);

  for (const file of await everyFile(daemon.root)) expect(await readFile(file, "utf8"), file).not.toContain(KEY);

  await page.getByRole("link", { name: "June", exact: true }).click();
  await expect(page).toHaveURL(/\/chat$/);
  await page.getByLabel("Route", { exact: true }).selectOption("openai");
  await expect(page.getByLabel("Where this message goes")).toContainText("leaves this Mac for api.openai.com");

  await page.getByRole("link", { name: "Settings" }).click();
  await expect(page).toHaveURL(/\/settings$/);
  await page.getByRole("article", { name: "OpenAI" }).getByRole("button", { name: "Remove" }).click();
  await expect(page.getByRole("article", { name: "OpenAI" }).getByText("No key", { exact: true })).toBeVisible();
});

test("a pasted key that is not one line of plain ASCII is refused with a reason", async ({ page }) => {
  await openSettings(page);
  const mistral = page.getByRole("article", { name: "Mistral" });
  await mistral.getByLabel("Mistral API key").fill("sk with spaceé");
  await mistral.getByRole("button", { name: "Save" }).click();
  await expect(mistral.getByRole("alert")).toContainText("printable ASCII");
});
