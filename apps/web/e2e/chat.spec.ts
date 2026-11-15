// M1 "done when", in the browser against the real daemon: a conversation with
// a local model streams, is saved as readable Markdown in the vault, and
// survives a reload and a daemon restart.
import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { type Daemon, startDaemon } from "./daemon.js";
import { type FakeOllama, startFakeOllama } from "./fake-ollama.js";

let daemon: Daemon;
let ollama: FakeOllama;
const cspViolations: string[] = [];

test.beforeAll(async () => {
  ollama = await startFakeOllama(["Around ", "12k, ", "appliances included."]);
  daemon = await startDaemon({ env: { OLLAMA_HOST: ollama.host } });
});
test.afterAll(async () => {
  daemon.stop();
  await ollama.close();
});

test.beforeEach(({ page }) => {
  page.on("console", (msg) => {
    if (/Content Security Policy|Refused to/i.test(msg.text())) cspViolations.push(msg.text());
  });
});

async function markdownFiles(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { recursive: true });
  return entries.filter((e) => e.endsWith(".md")).map((e) => join(dir, e));
}

test("chat with a local model: streams, says where it went, saves to the vault, survives a restart", async ({ page }) => {
  const vault = join(daemon.root, "Vault");
  await page.goto(`${daemon.origin}/#token=${daemon.token}`);
  await page.getByLabel("Vault folder").fill(vault);
  await page.getByRole("button", { name: "Use this folder" }).click();
  await expect(page).toHaveURL(/\/chat$/);

  await expect(page.getByLabel("Model")).toHaveValue("llama-test");
  await expect(page.getByLabel("Where this message goes")).toContainText("stays on this Mac");

  await page.getByLabel("Message June").fill("What should a 12 m2 kitchen cost?");
  await page.getByRole("button", { name: "Send message" }).click();

  await expect(page.getByText("Around 12k, appliances included.")).toBeVisible();
  await expect(page.getByText("Saved to your vault")).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Conversations" }).getByText("What should a 12 m2 kitchen cost?")).toBeVisible();

  const [file] = await markdownFiles(join(vault, "threads"));
  const markdown = await readFile(file ?? "", "utf8");
  expect(markdown).toContain("## User\n\nWhat should a 12 m2 kitchen cost?");
  expect(markdown).toContain("## Assistant\n\nAround 12k, appliances included.");
  expect(ollama.requests.some((r) => r.path === "/v1/chat/completions" && r.body.includes("12 m2 kitchen"))).toBe(true);

  // Restart the daemon on the same config: the conversation is still there.
  daemon.stop();
  daemon = await startDaemon({ root: daemon.root, env: { OLLAMA_HOST: ollama.host } });
  await page.goto(`${daemon.origin}/#token=${daemon.token}`);
  const nav = page.getByRole("navigation", { name: "Conversations" });
  await nav.getByText("What should a 12 m2 kitchen cost?", { exact: true }).click();
  await expect(page.getByText("Around 12k, appliances included.")).toBeVisible();

  // Branch: the history comes along, the original stays as it was.
  await page.getByRole("button", { name: "Continue in a new branch" }).click();
  await expect(nav.getByText("What should a 12 m2 kitchen cost? (branch)")).toBeVisible();
  await expect(page.getByText("Branched from")).toBeVisible();
  await expect(page.getByText("Around 12k, appliances included.")).toBeVisible();
  await page.getByLabel("Message June").fill("And with an island?");
  await page.getByRole("button", { name: "Send message" }).click();
  await expect(page.getByText("Saved to your vault")).toBeVisible();
  expect(await markdownFiles(join(vault, "threads"))).toHaveLength(2);
  const original = await readFile(file ?? "", "utf8");
  expect(original).not.toContain("And with an island?");
});

test("the chat page runs under the strict CSP without a single violation", () => {
  expect(cspViolations).toEqual([]);
});
