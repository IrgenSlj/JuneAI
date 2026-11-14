import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    projects: ["packages/core", "packages/node", "packages/providers", "packages/ui", "apps/server"],
  },
});
