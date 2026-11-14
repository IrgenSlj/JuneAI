// Enforces the adapter rule in PLAN.md: core source never imports Node
// built-ins or Electron, and never reaches the network or the process. If this
// fails, route the need through an interface in adapters.ts instead.
import { readdirSync, readFileSync } from "node:fs";
import { builtinModules } from "node:module";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";

const SRC = new URL(".", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const FORBIDDEN_MODULES = new Set(["electron", ...builtinModules]);
// Global fetch only: an injected function called as opts.fetch(...) is the allowed pattern.
const FORBIDDEN_GLOBALS = [/(?<![.\w])fetch\s*\(/, /(?<![.\w])process\./, /\brequire\s*\(/, /\bXMLHttpRequest\b/, /\bWebSocket\b/];

function sourceFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) return sourceFiles(full);
    return /\.ts$/.test(entry.name) && !/\.(test|d)\.ts$/.test(entry.name) ? [full] : [];
  });
}

describe("core boundaries", () => {
  const files = sourceFiles(SRC);

  it("finds the core source", () => {
    expect(files.length).toBeGreaterThan(0);
  });

  it.each(files.map((f) => [relative(SRC, f), f]))("%s imports no platform module", (_name, file) => {
    const text = readFileSync(file, "utf8");
    const specifiers = [...text.matchAll(/(?:from|import)\s*\(?\s*["']([^"']+)["']/g)].map((m) => m[1] ?? "");
    const bad = specifiers.filter((s) => s.startsWith("node:") || FORBIDDEN_MODULES.has(s.split("/")[0] ?? ""));
    expect(bad).toEqual([]);
    expect(FORBIDDEN_GLOBALS.filter((re) => re.test(text)).map(String)).toEqual([]);
  });
});
