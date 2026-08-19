import { test, expect } from "@playwright/test";
import { MOCK_API_BASE, mockApi } from "./_mocks.js";

/**
 * /help/ollama is where a user lands when the local model is missing, so it is
 * the page most likely to be read by someone whose install is already broken.
 * It has to render, and its live status has to reflect the backend rather than
 * a hopeful default — telling someone Ollama is reachable when it is not sends
 * them looking in the wrong place.
 */

test("/help/ollama renders the install steps and the download link", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  await mockApi(page);
  await page.goto("/help/ollama");

  await expect(page.getByRole("heading", { name: "Get Ollama running", level: 1 })).toBeVisible();

  const download = page.getByRole("link", { name: /ollama\.com\/download/ });
  await expect(download.first()).toHaveAttribute("href", "https://ollama.com/download");
  // External links open in a new tab and must not hand the opener over.
  await expect(download.first()).toHaveAttribute("rel", /noreferrer/);

  expect(pageErrors).toEqual([]);
});

test("/help/ollama reports a healthy install from the backend status", async ({ page }) => {
  await mockApi(page);
  await page.goto("/help/ollama");

  await expect(page.getByText(/Ollama is reachable at/)).toBeVisible();
  await expect(page.getByText(/is pulled\./).first()).toBeVisible();
});

test("/help/ollama reports an unreachable Ollama rather than assuming health", async ({ page }) => {
  await mockApi(page, {
    setupStatus: {
      ollama_reachable: false,
      ollama_has_model: false,
      embedding_available: false,
      semantic_recall_status: "unavailable",
    },
    system: {
      ollama_reachable: false,
      ollama_has_model: false,
      embedding_available: false,
      semantic_recall_status: "unavailable",
    },
  });
  await page.goto("/help/ollama");

  await expect(page.getByText(/Ollama is not reachable/)).toBeVisible();
});

test("/help/ollama switches OS instructions", async ({ page }) => {
  await mockApi(page);
  await page.goto("/help/ollama");

  await page.getByRole("button", { name: "Linux", exact: true }).click();
  await expect(page.getByText("curl -fsSL https://ollama.com/install.sh | sh")).toBeVisible();

  await page.getByRole("button", { name: "macOS", exact: true }).click();
  await expect(page.getByRole("link", { name: /ollama\.com\/download/ }).first()).toBeVisible();
});

test("/help/ollama still renders when the backend is unreachable", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  const esc = MOCK_API_BASE.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  await mockApi(page);
  await page.route(new RegExp(`${esc}/setup/status`), (route) =>
    route.fulfill({ status: 503, contentType: "application/json", body: "{}" }),
  );

  await page.goto("/help/ollama");

  await expect(page.getByRole("heading", { name: "Get Ollama running", level: 1 })).toBeVisible();
  expect(pageErrors).toEqual([]);
});
