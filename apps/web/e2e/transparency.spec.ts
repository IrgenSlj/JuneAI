// M3 in the browser: the transparency level changes only what is shown.
// Full asks before the first send on a route and keeps the draft on cancel;
// Quiet hides the route line and chips; the receipts page checks the chain.
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { type Daemon, startDaemon } from "./daemon.js";
import { type FakeOllama, startFakeOllama } from "./fake-ollama.js";

let daemon: Daemon;
let ollama: FakeOllama;

test.beforeAll(async () => {
  ollama = await startFakeOllama(["Fine, ", "thanks."]);
  daemon = await startDaemon({ env: { OLLAMA_HOST: ollama.host } });
});
test.afterAll(async () => {
  daemon.stop();
  await ollama.close();
});

const chats = () => ollama.requests.filter((r) => r.path.includes("chat")).length;

test("Full shows the label first, Quiet shows nothing, receipts check out either way", async ({ page }) => {
  await page.goto(`${daemon.origin}/#token=${daemon.token}`);
  await page.getByLabel("Vault folder").fill(join(daemon.root, "Vault"));
  await page.getByRole("button", { name: "Use this folder" }).click();
  await expect(page).toHaveURL(/\/chat$/);

  await page.getByRole("link", { name: "Settings" }).click();
  await page.getByRole("radio", { name: /Full/ }).check();
  await expect(page.getByRole("radio", { name: /Full/ })).toBeChecked();
  await page.getByRole("link", { name: "June", exact: true }).click();

  await page.getByLabel("Message June").fill("How are you?");
  await page.getByRole("button", { name: "Send message" }).click();
  const sheet = page.getByRole("dialog");
  await expect(sheet.getByRole("heading", { level: 2 })).toHaveText("About to send, staying on this Mac");
  await expect(sheet).toContainText("Your message");
  await expect(sheet).toContainText("12 characters");
  await sheet.getByRole("button", { name: "Cancel" }).click();
  await expect(sheet).toHaveCount(0);
  await expect(page.getByLabel("Message June")).toHaveValue("How are you?");
  expect(chats()).toBe(0);

  await page.getByRole("button", { name: "Send message" }).click();
  await sheet.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("Fine, thanks.")).toBeVisible();
  await expect(page.getByText(/receipt #\d+/)).toBeVisible();

  await page.getByLabel("Message June").fill("And again?");
  await page.getByRole("button", { name: "Send message" }).click();
  await expect(page.getByText("Fine, thanks.")).toHaveCount(2);
  await expect(sheet).toHaveCount(0);

  await page.getByRole("link", { name: "Settings" }).click();
  await page.getByRole("radio", { name: /Quiet/ }).check();
  await expect(page.getByRole("radio", { name: /Quiet/ })).toBeChecked();
  await page.getByRole("link", { name: "June", exact: true }).click();
  await page.getByRole("navigation", { name: "Conversations" }).getByRole("button", { name: /How are you/ }).click();
  await expect(page.getByText("Fine, thanks.").first()).toBeVisible();
  await expect(page.getByText("stays on this Mac")).toHaveCount(0);
  await expect(page.getByText(/receipt #/)).toHaveCount(0);

  await page.getByRole("link", { name: "Settings" }).click();
  await page.getByRole("link", { name: "Receipts" }).click();
  const list = page.getByRole("list", { name: "Receipts" });
  await expect(list.getByRole("listitem")).toHaveCount(2);
  await expect(list.getByRole("listitem").first()).toContainText("Sent");
  await expect(list.getByRole("listitem").first()).toContainText("How are you?");
  await page.getByRole("button", { name: "Check the chain" }).click();
  await expect(page.getByRole("status")).toHaveText("All 2 receipts check out.");
});
