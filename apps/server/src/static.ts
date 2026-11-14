// Serves the built UI. Paths are resolved and then checked to still be inside
// the static folder, so no encoding trick reaches a file outside it. Unknown
// paths get index.html, because the UI routes on the client.
import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { extname, join, resolve, sep } from "node:path";

const TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  ".txt": "text/plain; charset=utf-8",
};

export interface StaticFile {
  body: Buffer;
  type: string;
  /** CSP hashes of inline scripts, for HTML files. */
  scriptHashes: string[];
}

export function inlineScriptHashes(html: string): string[] {
  const hashes: string[] = [];
  for (const match of html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi)) {
    const code = match[1] ?? "";
    if (code.trim() !== "") hashes.push(`'sha256-${createHash("sha256").update(code).digest("base64")}'`);
  }
  return hashes;
}

export function createStaticHandler(dir: string) {
  const root = resolve(dir);
  const cache = new Map<string, StaticFile>();

  async function load(path: string): Promise<StaticFile | undefined> {
    const hit = cache.get(path);
    if (hit) return hit;
    try {
      if (!(await stat(path)).isFile()) return undefined;
    } catch {
      return undefined;
    }
    const body = await readFile(path);
    const type = TYPES[extname(path)] ?? "application/octet-stream";
    const file = { body, type, scriptHashes: type.startsWith("text/html") ? inlineScriptHashes(body.toString("utf8")) : [] };
    cache.set(path, file);
    return file;
  }

  return async function serve(urlPath: string): Promise<StaticFile | undefined> {
    let decoded: string;
    try {
      decoded = decodeURIComponent(urlPath);
    } catch {
      return undefined;
    }
    if (decoded.includes("\0")) return undefined;
    const candidate = resolve(join(root, decoded));
    if (candidate !== root && !candidate.startsWith(root + sep)) return undefined;
    return (await load(candidate)) ?? (extname(decoded) === "" ? load(join(root, "index.html")) : undefined);
  };
}
