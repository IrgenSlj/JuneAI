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

/** GET /system/traces — empty trace list. */
export const TRACES_FIXTURE = {
  traces: [],
  count: 0,
};

/** GET /system/ledger — empty ledger page. */
export const LEDGER_FIXTURE = {
  entries: [],
  count: 0,
  next_cursor: null,
};

/** GET /system/surfacing — empty decisions page. */
export const SURFACING_FIXTURE = {
  entries: [],
  count: 0,
  next_cursor: null,
};

/** GET /system/activity — empty activity log. */
export const ACTIVITY_FIXTURE = {
  entries: [],
  count: 0,
};

/** GET /system/capability — all verdicts unknown / not yet measured. */
export const CAPABILITY_FIXTURE = {
  summarization: "good",
  structured_output: "good",
  long_context: "good",
  relevance_scoring: "good",
  checked_at: "",
};

/** GET /memory/{user_id} — empty snapshot. */
export const MEMORY_SNAPSHOT_FIXTURE = {
  user_id: "default",
  goals: [],
  open_loops: [],
  calendar: [],
  journal: [],
  body_metrics: [],
  semantic_facts: [],
  entities: [],
  recent_messages: 0,
};

/** GET /memory/{user_id}/stats — zero totals. */
export const MEMORY_STATS_FIXTURE = {
  user_id: "default",
  total: 0,
  buckets: [],
  last_write: "",
  recent_messages: 0,
  recent_facts: [],
};

/** GET /memory/{user_id}/forgotten — empty trash. */
export const FORGOTTEN_FIXTURE = {
  user_id: "default",
  memories: [],
  count: 0,
};

/** GET /skills — no skills registered. */
export const SKILLS_FIXTURE = {
  skills: [],
  count: 0,
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
  traces?: Partial<typeof TRACES_FIXTURE>;
  ledger?: Partial<typeof LEDGER_FIXTURE>;
  surfacing?: Partial<typeof SURFACING_FIXTURE>;
  activity?: Partial<typeof ACTIVITY_FIXTURE>;
  capability?: Partial<typeof CAPABILITY_FIXTURE>;
  memorySnapshot?: Partial<typeof MEMORY_SNAPSHOT_FIXTURE>;
  memoryStats?: Partial<typeof MEMORY_STATS_FIXTURE>;
  forgotten?: Partial<typeof FORGOTTEN_FIXTURE>;
  skills?: Partial<typeof SKILLS_FIXTURE>;
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
 * NOTE: playwright.config.ts sets serviceWorkers: "block" so the PWA service
 * worker is never registered during tests. This ensures all fetch() calls from
 * the page go through Playwright's network interception layer where page.route()
 * can intercept them. Without SW blocking, the SW would re-issue cross-origin
 * API fetches from its own context, bypassing page.route() entirely.
 *
 * Call this BEFORE page.goto() so the mocks are in place before the first
 * network request fires.
 *
 * All routes use the base URL from MOCK_API_BASE so they match exactly
 * what the SvelteKit app sends (driven by PUBLIC_JUNE_API_URL).
 */
export async function mockApi(page: Page, overrides: MockApiOverrides = {}): Promise<void> {
  const base = MOCK_API_BASE;
  const esc = base.replace(/\./g, "\\.");

  // /setup/status — layout load() checks is_configured; false → redirect to /setup.
  await page.route(`${base}/setup/status`, (route) =>
    route.fulfill(jsonReply({ ...SETUP_STATUS_FIXTURE, ...overrides.setupStatus })),
  );

  // /system — layout onMount; populates the runtime badge.
  // Use regex to avoid matching /system/... sub-paths with this interceptor.
  await page.route(new RegExp(`${esc}/system$`), (route) =>
    route.fulfill(jsonReply({ ...SYSTEM_FIXTURE, ...overrides.system })),
  );

  // /home/{user_id}/holdings — Home page onMount.
  await page.route(`${base}/home/*/holdings`, (route) =>
    route.fulfill(jsonReply({ ...HOLDINGS_FIXTURE, ...overrides.holdings })),
  );

  // /tasks/{user_id}?... — Home page onMount (Promise.all alongside holdings).
  await page.route(new RegExp(`${esc}/tasks/`), (route) =>
    route.fulfill(jsonReply({ ...TASKS_FIXTURE, ...overrides.tasks })),
  );

  // /greeting/{user_id}?... — Home page best-effort after initial load.
  await page.route(new RegExp(`${esc}/greeting/`), (route) =>
    route.fulfill(jsonReply({ ...GREETING_FIXTURE, ...overrides.greeting })),
  );

  // /system/traces — Glass Box page + Trust page.
  // Must come before the /system$ wildcard so the more-specific path wins.
  await page.route(new RegExp(`${esc}/system/traces`), (route) =>
    route.fulfill(jsonReply({ ...TRACES_FIXTURE, ...overrides.traces })),
  );

  // /system/ledger — Receipts page (GET) and verify (POST).
  await page.route(new RegExp(`${esc}/system/ledger`), (route) =>
    route.fulfill(jsonReply({ ...LEDGER_FIXTURE, ...overrides.ledger })),
  );

  // /system/surfacing — Silence page.
  await page.route(new RegExp(`${esc}/system/surfacing`), (route) =>
    route.fulfill(jsonReply({ ...SURFACING_FIXTURE, ...overrides.surfacing })),
  );

  // /system/activity — Trust page.
  await page.route(new RegExp(`${esc}/system/activity`), (route) =>
    route.fulfill(jsonReply({ ...ACTIVITY_FIXTURE, ...overrides.activity })),
  );

  // /system/capability — Trust page.
  await page.route(new RegExp(`${esc}/system/capability`), (route) =>
    route.fulfill(jsonReply({ ...CAPABILITY_FIXTURE, ...overrides.capability })),
  );

  // /skills — Trust page.
  await page.route(new RegExp(`${esc}/skills`), (route) =>
    route.fulfill(jsonReply({ ...SKILLS_FIXTURE, ...overrides.skills })),
  );

  // /memory/{user_id}/stats — Memory page.
  await page.route(new RegExp(`${esc}/memory/[^/?]+/stats`), (route) =>
    route.fulfill(jsonReply({ ...MEMORY_STATS_FIXTURE, ...overrides.memoryStats })),
  );

  // /memory/{user_id}/forgotten — Memory page.
  await page.route(new RegExp(`${esc}/memory/[^/?]+/forgotten`), (route) =>
    route.fulfill(jsonReply({ ...FORGOTTEN_FIXTURE, ...overrides.forgotten })),
  );

  // /memory/{user_id} — Memory page + Trust page.
  // Must come after the more-specific /memory/*/stats and /memory/*/forgotten routes.
  // The [^/?]+ prevents matching sub-paths; the (?:[?#]|$) allows query strings.
  await page.route(new RegExp(`${esc}/memory/[^/?]+(\\?|$)`), (route) =>
    route.fulfill(jsonReply({ ...MEMORY_SNAPSHOT_FIXTURE, ...overrides.memorySnapshot })),
  );
}
