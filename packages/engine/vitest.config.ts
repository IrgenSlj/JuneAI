import { defineProject } from "vitest/config";

export default defineProject({
  test: {
    name: "engine",
    include: ["src/**/*.test.ts"],
  },
});
