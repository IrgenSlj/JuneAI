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

/** GET /healthz — startup gate can render the app shell. */
export const HEALTHZ_FIXTURE = {
  ok: true,
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

/** GET /chat/history/{user_id} — empty persisted transcript. */
export const CHAT_HISTORY_FIXTURE = {
  messages: [],
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

/** GET /settings — Gemma preset, local-only dial, no cloud key. */
export const SETTINGS_FIXTURE = {
  provider: "gemma",
  model: "gemma4:e2b",
  gemma_model: "",
  gemini_model: "",
  ollama_base_url: "",
  ollama_reachable: true,
  ollama_has_model: true,
  api_key_present: false,
  key_storage: "none",
  key_storage_label: "No key stored",
  privacy_dial: "local_only",
};

/** PUT /settings/privacy-dial — accepted. */
export const PRIVACY_DIAL_FIXTURE = {
  ok: true,
  privacy_dial: "private_by_default",
  label: "Private by default",
};

/** GET /skills — two skills, one of them declaring a sensitive scope.
 *
 * The Skills page is where a user decides what June may do on their behalf, so
 * the fixture has to carry a real capability contract: a skill with
 * "sends data off device" is the case the page exists to make visible. */
export const SKILLS_POPULATED_FIXTURE = {
  skills: [
    {
      key: "research",
      description: "Web search via Brave Search or DuckDuckGo.",
      enabled: true,
      status: "running",
      error: "",
      model_policy: "local_ok",
      tools: [
        { name: "web_search", description: "Search the web.", enabled: true, input_schema: {} },
        { name: "fetch_url", description: "Fetch a URL.", enabled: true, input_schema: {} },
      ],
      scopes: ["sends data off device"],
      scope_drift: { undeclared: [], unused: [], has_drift: false },
    },
    {
      key: "calendar",
      description: "Calendar events, reminders, and birthdays.",
      enabled: false,
      status: "stopped",
      error: "",
      model_policy: "local_ok",
      tools: [
        { name: "save_calendar_item", description: "Save an item.", enabled: true, input_schema: {} },
      ],
      scopes: ["reads local data", "writes local data"],
      scope_drift: { undeclared: [], unused: [], has_drift: false },
    },
  ],
  count: 2,
};

/** GET /skills/registry — one installed entry, one not, one unverified.
 *
 * The registry is where a third-party MCP server gets installed into June, so
 * the fixture carries the two signals a user needs before doing that: whether
 * the entry is verified, and what model policy it declares. */
export const REGISTRY_FIXTURE = {
  schema_version: 1,
  source: "bundled",
  updated_at: "2026-08-01T00:00:00Z",
  entries: [
    {
      key: "research",
      name: "Research",
      description: "Web search via Brave Search or DuckDuckGo.",
      homepage: "https://example.invalid/research",
      publisher: "June",
      verified: true,
      model_policy: "local_ok",
      install: {},
      tools_preview: ["web_search", "fetch_url"],
      installed: true,
    },
    {
      key: "weather",
      name: "Weather",
      description: "Current conditions and forecasts.",
      homepage: "https://example.invalid/weather",
      publisher: "Third Party",
      verified: false,
      model_policy: "local_ok",
      install: {},
      tools_preview: ["get_forecast"],
      installed: false,
    },
  ],
  count: 2,
};

/** POST /skills/{key}/toggle — accepted. */
export const SKILL_TOGGLE_FIXTURE = {
  ok: true,
  key: "research",
  enabled: false,
  status: "stopped",
  error: "",
};

// ---------------------------------------------------------------------------
// Override surface
// ---------------------------------------------------------------------------

export interface MockApiOverrides {
  healthz?: Partial<typeof HEALTHZ_FIXTURE>;
  system?: Partial<typeof SYSTEM_FIXTURE>;
  setupStatus?: Partial<typeof SETUP_STATUS_FIXTURE>;
  holdings?: Partial<typeof HOLDINGS_FIXTURE>;
  tasks?: Partial<typeof TASKS_FIXTURE>;
  greeting?: Partial<typeof GREETING_FIXTURE>;
  chatHistory?: Partial<typeof CHAT_HISTORY_FIXTURE>;
  traces?: Partial<typeof TRACES_FIXTURE>;
  ledger?: Partial<typeof LEDGER_FIXTURE>;
  surfacing?: Partial<typeof SURFACING_FIXTURE>;
  activity?: Partial<typeof ACTIVITY_FIXTURE>;
  capability?: Partial<typeof CAPABILITY_FIXTURE>;
  memorySnapshot?: Partial<typeof MEMORY_SNAPSHOT_FIXTURE>;
  memoryStats?: Partial<typeof MEMORY_STATS_FIXTURE>;
  forgotten?: Partial<typeof FORGOTTEN_FIXTURE>;
  skills?: Partial<typeof SKILLS_FIXTURE>;
  settings?: Partial<typeof SETTINGS_FIXTURE>;
  registry?: Partial<typeof REGISTRY_FIXTURE>;
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

  // /healthz — layout startup gate; false/missing keeps every route on warming UI.
  await page.route(`${base}/healthz`, (route) =>
    route.fulfill(jsonReply({ ...HEALTHZ_FIXTURE, ...overrides.healthz })),
  );

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

  // /chat/history/{user_id} — Chat page best-effort transcript restore.
  await page.route(new RegExp(`${esc}/chat/history/`), (route) =>
    route.fulfill(jsonReply({ ...CHAT_HISTORY_FIXTURE, ...overrides.chatHistory })),
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

  // /skills and its sub-paths.
  //
  // The list route is anchored rather than ordered. Playwright matches routes
  // last-registered-first, so an unanchored `/skills` swallows
  // `/skills/registry` and `/skills/{key}/toggle` no matter where it sits — the
  // registry request comes back as the installed-skills list, the page renders
  // empty, and a spec that only checked the heading would pass on it.
  await page.route(new RegExp(`${esc}/skills(\\?|$)`), (route) =>
    route.fulfill(jsonReply({ ...SKILLS_FIXTURE, ...overrides.skills })),
  );

  await page.route(new RegExp(`${esc}/skills/registry(\\?|$)`), (route) =>
    route.fulfill(jsonReply({ ...REGISTRY_FIXTURE, ...overrides.registry })),
  );

  await page.route(new RegExp(`${esc}/skills/[^/?]+/toggle`), (route) =>
    route.fulfill(jsonReply(SKILL_TOGGLE_FIXTURE)),
  );

  // /settings/privacy-dial — Settings page (PUT). The /settings route below is
  // anchored, so this is not order-dependent.
  await page.route(new RegExp(`${esc}/settings/privacy-dial`), (route) =>
    route.fulfill(jsonReply(PRIVACY_DIAL_FIXTURE)),
  );

  // /settings — Settings page.
  await page.route(new RegExp(`${esc}/settings(\\?|$)`), (route) =>
    route.fulfill(jsonReply({ ...SETTINGS_FIXTURE, ...overrides.settings })),
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
