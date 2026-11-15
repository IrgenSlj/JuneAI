import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    projects: ["tools", "packages/core", "packages/node", "packages/providers", "packages/engine", "packages/ui", "apps/server"],
  },
});
