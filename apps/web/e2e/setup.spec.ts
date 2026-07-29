import { test, expect } from "@playwright/test";
import { MOCK_API_BASE, mockApi } from "./_mocks.js";

/** GET /setup/status — not configured. */
const NOT_CONFIGURED = {
  is_configured: false,
  provider: "gemma",
  model: "gemma4:e2b",
  ollama_reachable: true,
  ollama_has_model: true,
  embedding_model: "nomic-embed-text",
  embedding_available: true,
  semantic_recall_status: "ready",
  semantic_recall_detail: "",
  api_key_present: false,
  user_name: "",
};

/** GET /setup/status — configured, done state. */
const CONFIGURED = {
  is_configured: true,
  provider: "gemma",
  model: "gemma4:e2b",
  ollama_reachable: true,
  ollama_has_model: true,
  embedding_model: "nomic-embed-text",
  embedding_available: true,
  semantic_recall_status: "ready",
  semantic_recall_detail: "",
  api_key_present: false,
  user_name: "",
};

/** POST /setup/apply — success response. */
function applyFixture(ok: boolean, provider = "gemma") {
  return {
    ok,
    provider,
    model: ok ? "gemma4:e2b" : "",
    verified: ok,
    message: ok ? "Provider verified successfully." : "Verification failed.",
    hint: ok ? "" : "Check that Ollama is running and try again.",
  };
}

test("/setup redirects to /chat when already configured", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  await mockApi(page, { setupStatus: { is_configured: true } });

  await page.goto("/setup");
  await expect(page.getByText("Ready to chat.")).toBeVisible();
  await expect(page.getByText("Continue to chat")).toBeVisible();

  expect(pageErrors, `Uncaught page errors: ${pageErrors.join(" | ")}`).toHaveLength(0);
});

test("/setup shows provider selection when unconfigured", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  await mockApi(page, { setupStatus: NOT_CONFIGURED });

  await page.goto("/setup");

  // Should show the provider selection card
  await expect(page.getByText("Set up your assistant")).toBeVisible();
  await expect(page.getByText("Gemma 4 (local)")).toBeVisible();
  await expect(page.getByText("Gemini (cloud)")).toBeVisible();
  await expect(page.getByText("Ollama is ready")).toBeVisible();
  await expect(page.getByRole("button", { name: /Verify and continue/ })).toBeVisible();

  expect(pageErrors, `Uncaught page errors: ${pageErrors.join(" | ")}`).toHaveLength(0);
});

test("/setup shows Ollama-not-running state", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  await mockApi(page, {
    setupStatus: { ...NOT_CONFIGURED, ollama_reachable: false, ollama_has_model: false },
  });

  await page.goto("/setup");

  await expect(page.getByText("Ollama isn't running.")).toBeVisible();
  await expect(page.getByText("See the three-step guide.")).toBeVisible();

  expect(pageErrors, `Uncaught page errors: ${pageErrors.join(" | ")}`).toHaveLength(0);
});

test("/setup shows Ollama-missing-model state", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  await mockApi(page, {
    setupStatus: { ...NOT_CONFIGURED, ollama_reachable: true, ollama_has_model: false },
  });

  await page.goto("/setup");

  await expect(page.getByText("Ollama is running but the model isn't pulled.")).toBeVisible();

  expect(pageErrors, `Uncaught page errors: ${pageErrors.join(" | ")}`).toHaveLength(0);
});

test("/setup shows degraded-memory state", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  await mockApi(page, {
    setupStatus: {
      ...NOT_CONFIGURED,
      semantic_recall_status: "degraded",
      semantic_recall_detail: "embedding model not found",
    },
  });

  await page.goto("/setup");

  await expect(page.getByText("memory is running degraded")).toBeVisible();
  await expect(page.getByText("Pull the embedding model.")).toBeVisible();

  expect(pageErrors, `Uncaught page errors: ${pageErrors.join(" | ")}`).toHaveLength(0);
});

test("/setup apply succeeds with Gemma", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  await mockApi(page, { setupStatus: NOT_CONFIGURED });

  // Mock the POST /setup/apply endpoint.
  await page.route(`${MOCK_API_BASE}/setup/apply`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(applyFixture(true)),
    });
  });

  await page.goto("/setup");

  await page.getByRole("button", { name: /Verify and continue/ }).click();

  // Should show verifying state then done.
  await expect(page.getByText("Ready to chat.")).toBeVisible({ timeout: 10000 });
  await expect(page.getByText("Continue to chat")).toBeVisible();

  expect(pageErrors, `Uncaught page errors: ${pageErrors.join(" | ")}`).toHaveLength(0);
});

test("/setup apply shows error on failure", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  await mockApi(page, { setupStatus: NOT_CONFIGURED });

  await page.route(`${MOCK_API_BASE}/setup/apply`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(applyFixture(false)),
    });
  });

  await page.goto("/setup");

  await page.getByRole("button", { name: /Verify and continue/ }).click();

  // Should show error state.
  await expect(page.getByText("Verification failed.")).toBeVisible({ timeout: 10000 });
  await expect(page.getByText("Check that Ollama is running and try again.")).toBeVisible();

  expect(pageErrors, `Uncaught page errors: ${pageErrors.join(" | ")}`).toHaveLength(0);
});
