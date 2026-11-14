import { defineConfig } from "@playwright/test";

// The Mac app's window is WebKit (WKWebView), so WebKit is the right engine
// here. Playwright 1.65's WebKit build crashes on launch on macOS 26.0
// (segfault before any page loads, inside and outside the sandbox), so the
// suite runs on Chromium until that is fixed; PLAN.md lists the gap.
export default defineConfig({
  testDir: "e2e",
  fullyParallel: false,
  reporter: "list",
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
