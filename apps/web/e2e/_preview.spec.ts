import { test } from "@playwright/test";

const OUT = process.env.SHOT_DIR ?? "/tmp";
const BASE = process.env.PREVIEW_BASE ?? "http://localhost:5174";

async function say(page: any, text: string, shot: string) {
  const box = page.getByPlaceholder(/Write to June/i);
  await box.fill(text);
  await box.press("Meta+Enter");
  await page.getByText(/Thinking locally/i).waitFor({ state: "hidden", timeout: 480_000 }).catch(() => {});
  await page.waitForTimeout(3500);
  await page.screenshot({ path: `${OUT}/${shot}.png`, fullPage: false });
}

test("store then recall", async ({ page }) => {
  test.setTimeout(900_000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.setViewportSize({ width: 1280, height: 900 });

  await page.goto(`${BASE}/chat`, { waitUntil: "networkidle" }).catch(() => {});
  await page.waitForTimeout(1500);

  await say(page, "Remember that my sister is called Mira and she lives in Lisbon.", "live-1-store");
  await say(page, "Where does my sister live?", "live-2-recall");

  console.log("errors:", errors.length);
  console.log("TRANSCRIPT:", (await page.locator("main").innerText()).replace(/\s+/g, " ").slice(0, 900));
});
