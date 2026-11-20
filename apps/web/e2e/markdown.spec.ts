// Model output renders as Markdown and never as HTML: a hostile reply shows
// its markup as text, loads nothing and runs nothing.
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { type Daemon, startDaemon } from "./daemon.js";
import { type FakeOllama, startFakeOllama } from "./fake-ollama.js";

let daemon: Daemon;
let ollama: FakeOllama;

test.beforeAll(async () => {
  ollama = await startFakeOllama([
    "## Plan\n\nA **firm** budget:\n\n- oven\n- sink\n\n",
    '<img src="https://evil.example/x.png" onerror="window.pwned=1"> ',
    "[click](javascript:window.pwned=2) and [docs](https://example.com)\n\n```\n<script>window.pwned=3</script>\n```",
  ]);
  daemon = await startDaemon({ env: { OLLAMA_HOST: ollama.host } });
});
test.afterAll(async () => {
  daemon.stop();
  await ollama.close();
});

test("a reply renders as Markdown, and its HTML stays inert text", async ({ page }) => {
  const outbound: string[] = [];
  page.on("request", (req) => {
    if (!req.url().startsWith(daemon.origin)) outbound.push(req.url());
  });
  await page.goto(`${daemon.origin}/#token=${daemon.token}`);
  await page.getByLabel("Vault folder").fill(join(daemon.root, "Vault"));
  await page.getByRole("button", { name: "Use this folder" }).click();
  await page.getByLabel("Message June").fill("Plan the kitchen");
  await page.getByRole("button", { name: "Send message" }).click();
  await expect(page.getByText("Saved to your vault")).toBeVisible();

  const reply = page.locator('[data-role="assistant"]').last();
  await expect(reply.getByRole("heading", { name: "Plan" })).toBeVisible();
  await expect(reply.locator("strong")).toHaveText("firm");
  await expect(reply.getByRole("listitem")).toHaveText(["oven", "sink"]);
  await expect(reply.getByRole("link", { name: "docs" })).toHaveAttribute("href", "https://example.com");
  await expect(reply.getByRole("link", { name: "click" })).toHaveCount(0);
  await expect(reply.getByText("<img src=")).toBeVisible();
  await expect(reply.locator("pre code")).toContainText("<script>window.pwned=3</script>");
  expect(await reply.locator("img, script, iframe").count()).toBe(0);
  expect(await page.evaluate(() => (window as unknown as { pwned?: number }).pwned)).toBeUndefined();
  expect(outbound).toEqual([]);
});

test("in the Mac app, a link in a reply opens in the browser through the opener, not in June's window", async ({ page }) => {
  // Stand in for the Tauri bridge, recording what the page asks the shell to do.
  await page.addInitScript(() => {
    const calls: unknown[] = [];
    (window as unknown as { __calls: unknown[] }).__calls = calls;
    (window as unknown as { __TAURI_INTERNALS__: unknown }).__TAURI_INTERNALS__ = {
      invoke: (cmd: string, args: unknown) => {
        calls.push({ cmd, args });
        return Promise.resolve(null);
      },
      transformCallback: () => 0,
    };
  });
  await page.goto(`${daemon.origin}/#token=${daemon.token}`);
  const nav = page.getByRole("navigation", { name: "Conversations" });
  await nav.getByText("Plan the kitchen", { exact: true }).click();
  const before = page.context().pages().length;
  await page.locator('[data-role="assistant"]').last().getByRole("link", { name: "docs" }).click();
  await expect
    .poll(() => page.evaluate(() => JSON.stringify((window as unknown as { __calls: unknown[] }).__calls)))
    .toContain("https://example.com");
  const calls = await page.evaluate(() => JSON.stringify((window as unknown as { __calls: unknown[] }).__calls));
  expect(calls).toContain("opener");
  expect(page.context().pages().length).toBe(before);
  expect(page.url()).toContain(daemon.origin);
});
