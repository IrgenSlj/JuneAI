// M2 in the browser against the real daemon: import a Claude export, see a
// report that adds up, find a conversation by search, continue it on a local
// model as a branch (the imported file stays untouched), and export one.
import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { type Daemon, startDaemon } from "./daemon.js";
import { type FakeOllama, startFakeOllama } from "./fake-ollama.js";

const fixture = join(import.meta.dirname, "../../../packages/core/src/import/fixtures/claude-sample.json");
let daemon: Daemon;
let ollama: FakeOllama;

test.beforeAll(async () => {
  ollama = await startFakeOllama(["Day two: ", "Belém."]);
  daemon = await startDaemon({ env: { OLLAMA_HOST: ollama.host } });
});
test.afterAll(async () => {
  daemon.stop();
  await ollama.close();
});

test("import a Claude export, search it, continue a conversation as a branch, export it", async ({ page }) => {
  const vault = join(daemon.root, "Vault");
  await page.goto(`${daemon.origin}/#token=${daemon.token}`);
  await page.getByLabel("Vault folder").fill(vault);
  await page.getByRole("button", { name: "Use this folder" }).click();
  await expect(page).toHaveURL(/\/chat$/);
  await expect(page.getByText("Import your Claude or ChatGPT history")).toBeVisible();

  await page.getByRole("link", { name: "Import", exact: true }).click();
  await page.locator('input[type="file"]').setInputFiles({ name: "conversations.json", mimeType: "application/json", buffer: await readFile(fixture) });
  const report = page.getByRole("region", { name: "Import report" });
  await expect(report.getByRole("heading")).toHaveText("Imported 2 conversations");
  await expect(report).toContainText("4 conversations");
  await expect(report).toContainText("Every conversation in the export is accounted for.");

  await page.locator('input[type="file"]').setInputFiles({ name: "conversations.json", mimeType: "application/json", buffer: await readFile(fixture) });
  await expect(report.getByRole("heading")).toHaveText("Already up to date");

  await report.getByRole("link", { name: "Open conversations" }).click();
  const list = page.getByRole("navigation", { name: "Conversations" });
  await expect(list.getByRole("button", { name: /Kitchen quotes/ })).toContainText("Claude");

  await page.getByLabel("Search conversations").fill("alfama");
  const results = page.getByRole("list", { name: "Search results" });
  await expect(results.locator("mark")).toHaveText("Alfama");
  await results.getByRole("button", { name: /Plan a trip to Lisbon/ }).click();
  await expect(page.getByText("Imported from Claude")).toBeVisible();
  await expect(page.getByLabel("Message June")).toHaveCount(0);

  const before = await readdir(join(vault, "threads"), { recursive: true });
  await page.getByRole("button", { name: "Continue in a new thread" }).click();
  await expect(page.getByText("Branched from")).toBeVisible();
  await page.getByLabel("Message June").fill("And day two?");
  await page.getByLabel("Message June").press("Shift+Enter");
  await expect(page.getByLabel("Message June")).toHaveValue("And day two?\n");
  await page.getByLabel("Message June").press("Backspace");
  await page.getByLabel("Message June").press("Enter");
  await expect(page.getByText("Day two: Belém.")).toBeVisible();
  await expect(page.getByText("Saved to your vault")).toBeVisible();
  const sentToModel = JSON.parse(ollama.requests.filter((r) => r.path.includes("chat")).at(-1)?.body ?? "{}") as { messages: { content: string }[] };
  expect(sentToModel.messages.map((m) => m.content)).toEqual(["Plan a trip to Lisbon\nwith two days", "Day one: Alfama.", "And day two?"]);
  const after = await readdir(join(vault, "threads"), { recursive: true });
  for (const file of before.filter((f) => f.endsWith(".md"))) expect(after).toContain(file);

  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export .md" }).click();
  const saved = await (await download).path();
  const text = await readFile(saved, "utf8");
  expect(text).toMatch(/^# Plan a trip to Lisbon/);
  expect(text).toContain("Day two: Belém.");
  expect(text).not.toContain("june:turn");

  await page.getByRole("button", { name: "Delete" }).click();
  await page.getByRole("button", { name: "Move to trash?" }).click();
  await expect(page.getByText("Moved to .trash in your vault")).toBeVisible();
  await expect(list.getByRole("button", { name: /\(branch\)/ })).toHaveCount(0);
  const trashed = await readdir(join(vault, ".trash", "threads"), { recursive: true });
  expect(trashed.some((f) => f.endsWith(".md"))).toBe(true);
});
