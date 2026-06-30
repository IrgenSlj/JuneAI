import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "e2e",
  timeout: 30_000,
  expect: { timeout: 8_000 },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: "list",
  use: {
    baseURL: "http://localhost:5199",
    trace: "on-first-retry",
    // Block service worker registration so page.route() can intercept API calls
    // that would otherwise be re-fetched by the PWA service worker from its own
    // context (which Playwright's page/context-level routing cannot see).
    serviceWorkers: "block",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    // Dedicated port + no reuse: always boot a fresh dev server with the dummy
    // PUBLIC_JUNE_API_URL so page.route() mocks actually intercept (reusing a
    // stale server on 5173 from another session would bypass the mocks). 5199
    // avoids colliding with a normal dev server.
    command: "pnpm exec vite dev --port 5199 --strictPort",
    url: "http://localhost:5199",
    reuseExistingServer: false,
    env: {
      PUBLIC_JUNE_API_URL: "http://127.0.0.1:8099",
    },
    timeout: 60_000,
  },
});
