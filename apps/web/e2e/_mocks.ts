/**
 * Hermetic API mock helpers for Playwright e2e tests.
 *
 * All backend calls hit http://127.0.0.1:8099 (set via PUBLIC_JUNE_API_URL in
 * playwright.config.ts webServer env). Nothing real listens there — every
 * request is intercepted by page.route() before it leaves the browser.
 *
 * Usage:
 *   await mockApi(page);           // default fixtures
 *   await mockApi(page, { system: { model: "gemini-2.0-flash" } });
 */

import type { Page } from "@playwright/test";

export const MOCK_API_BASE = "http://127.0.0.1:8099";

// ---------------------------------------------------------------------------
// Default fixtures
// ---------------------------------------------------------------------------

/** GET /system — local provider, ready, local-only dial. */
export const SYSTEM_FIXTURE = {
  provider: "gemma",
  label: "Gemma 3",
  model: "gemma3:4b",
  mode: "local",
  privacy_label: "local-only",
  privacy_dial: "local_only",
  base_url: "http://localhost:11434",
  ollama_reachable: true,
  ollama_has_model: true,
  embedding_model: "nomic-embed-text",
  embedding_available: true,
  semantic_recall_status: "ready",
  semantic_recall_detail: "",
  api_key_present: false,
  version: "test",
  ledger_summary: null,
};

/** GET /setup/status — app is configured, no redirect to /setup. */
export const SETUP_STATUS_FIXTURE = {
  is_configured: true,
  provider: "gemma",
  model: "gemma3:4b",
  ollama_reachable: true,
  ollama_has_model: true,
  embedding_model: "nomic-embed-text",
  embedding_available: true,
  semantic_recall_status: "ready",
  semantic_recall_detail: "",
  api_key_present: false,
  user_name: "",
};

/** GET /home/{user_id}/holdings — zero open promises, clear state. */
export const HOLDINGS_FIXTURE = {
  open_promises: 0,
  waiting_on_user: 0,
  blocked_by_local_only: 0,
  next_deadline: null,
  held_digest: [],
  held_count: 0,
  needs_now: [],
  egress_today: 0,
  chain_verified: null,
};

/** GET /tasks/{user_id} — no tasks. */
export const TASKS_FIXTURE = {
  tasks: [],
  count: 0,
};

/** GET /greeting/{user_id} — static greeting. */
export const GREETING_FIXTURE = {
  greeting: "Good to see you.",
  has_context: false,
};

// ---------------------------------------------------------------------------
// Override surface
// ---------------------------------------------------------------------------

export interface MockApiOverrides {
  system?: Partial<typeof SYSTEM_FIXTURE>;
  setupStatus?: Partial<typeof SETUP_STATUS_FIXTURE>;
  holdings?: Partial<typeof HOLDINGS_FIXTURE>;
  tasks?: Partial<typeof TASKS_FIXTURE>;
  greeting?: Partial<typeof GREETING_FIXTURE>;
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

function jsonReply(body: unknown, status = 200) {
  return {
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  };
}

/**
 * Register page.route() interceptors for every endpoint the Home page
 * (and the shared layout) fetches on mount.
 *
 * Call this BEFORE page.goto() so the mocks are in place before the first
 * network request fires.
 *
 * All routes use the base URL from MOCK_API_BASE so they match exactly
 * what the SvelteKit app sends (driven by PUBLIC_JUNE_API_URL).
 */
export async function mockApi(page: Page, overrides: MockApiOverrides = {}): Promise<void> {
  const base = MOCK_API_BASE;

  // /setup/status — layout load() checks is_configured; false → redirect to /setup.
  await page.route(`${base}/setup/status`, (route) =>
    route.fulfill(jsonReply({ ...SETUP_STATUS_FIXTURE, ...overrides.setupStatus })),
  );

  // /system — layout onMount; populates the runtime badge.
  // Use regex to avoid matching /system/... sub-paths with this interceptor.
  await page.route(new RegExp(`${base.replace(".", "\\.")}/system$`), (route) =>
    route.fulfill(jsonReply({ ...SYSTEM_FIXTURE, ...overrides.system })),
  );

  // /home/{user_id}/holdings — Home page onMount.
  await page.route(`${base}/home/*/holdings`, (route) =>
    route.fulfill(jsonReply({ ...HOLDINGS_FIXTURE, ...overrides.holdings })),
  );

  // /tasks/{user_id}?... — Home page onMount (Promise.all alongside holdings).
  await page.route(new RegExp(`${base.replace(".", "\\.")}/tasks/`), (route) =>
    route.fulfill(jsonReply({ ...TASKS_FIXTURE, ...overrides.tasks })),
  );

  // /greeting/{user_id}?... — Home page best-effort after initial load.
  await page.route(new RegExp(`${base.replace(".", "\\.")}/greeting/`), (route) =>
    route.fulfill(jsonReply({ ...GREETING_FIXTURE, ...overrides.greeting })),
  );
}
