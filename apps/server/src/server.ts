// The June daemon. Loopback only, a random port, one origin for UI and API.
// Rules (PLAN.md, "Daemon rules"): every /api route except health needs the
// per-launch token; a foreign Host header is refused (DNS rebinding); no CORS
// headers at all; a strict CSP that forbids framing.
import { timingSafeEqual } from "node:crypto";
import { mkdir, stat } from "node:fs/promises";
import { type IncomingMessage, type ServerResponse, createServer } from "node:http";
import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";
import type { Health, OpenVaultRequest, VaultInfo } from "@june/ui/api/types";
import { readConfig, writeConfig } from "./config.js";
import { createStaticHandler } from "./static.js";

export interface ServerOptions {
  token: string;
  /** Built UI to serve; without it only the API answers. */
  staticDir?: string;
  configDir: string;
  /** Home folder, for "~" and the default vault. */
  home?: string;
  version: string;
  /** 0 picks a free port. */
  port?: number;
}

export interface RunningServer {
  host: "127.0.0.1";
  port: number;
  close(): Promise<void>;
}

const HOST = "127.0.0.1";
const MAX_BODY = 1_000_000;

class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

function sameSecret(given: string | string[] | undefined, expected: Buffer): boolean {
  if (typeof given !== "string") return false;
  const buffer = Buffer.from(given);
  return buffer.length === expected.length && timingSafeEqual(buffer, expected);
}

function securityHeaders(res: ServerResponse, scriptHashes: string[] = []): void {
  res.setHeader(
    "Content-Security-Policy",
    [
      "default-src 'self'",
      `script-src 'self'${scriptHashes.map((h) => ` ${h}`).join("")}`,
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data:",
      "font-src 'self'",
      // ipc: is how the Tauri window reaches the shell's native dialogs.
      "connect-src 'self' ipc: http://ipc.localhost",
      "object-src 'none'",
      "base-uri 'none'",
      "form-action 'self'",
      "frame-ancestors 'none'",
    ].join("; "),
  );
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("Referrer-Policy", "no-referrer");
  res.setHeader("Cache-Control", "no-store");
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  securityHeaders(res);
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
}

async function readJson(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    size += (chunk as Buffer).length;
    if (size > MAX_BODY) throw new HttpError(413, "request body too large");
    chunks.push(chunk as Buffer);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw new HttpError(400, "body is not JSON");
  }
}

export async function startServer(options: ServerOptions): Promise<RunningServer> {
  const home = options.home ?? homedir();
  const token = Buffer.from(options.token);
  const serveStatic = options.staticDir === undefined ? undefined : createStaticHandler(options.staticDir);
  let vault = (await readConfig(options.configDir)).vault ?? null;
  let allowedHosts = new Set<string>();

  const expandHome = (path: string): string => (path === "~" ? home : path.startsWith("~/") ? join(home, path.slice(2)) : path);
  const vaultInfo = (): VaultInfo => ({ path: vault, defaultPath: join(home, "June") });

  async function openVault(body: unknown): Promise<VaultInfo> {
    const raw = (body as Partial<OpenVaultRequest> | null)?.path;
    if (typeof raw !== "string" || raw.trim() === "") throw new HttpError(400, "path is required");
    const path = expandHome(raw.trim());
    if (!isAbsolute(path)) throw new HttpError(400, "path must be absolute or start with ~/");
    const target = resolve(path);
    try {
      if (!(await stat(target)).isDirectory()) throw new HttpError(400, `not a folder: ${target}`);
    } catch (error) {
      if (error instanceof HttpError) throw error;
      await mkdir(target, { recursive: true });
    }
    vault = target;
    await writeConfig(options.configDir, { vault });
    return vaultInfo();
  }

  async function handleApi(req: IncomingMessage, res: ServerResponse, path: string): Promise<void> {
    if (req.method === "GET" && path === "/api/health") {
      const health: Health = { ok: true, name: "june", version: options.version };
      return sendJson(res, 200, health);
    }
    if (!sameSecret(req.headers["x-june-token"], token)) throw new HttpError(401, "missing or wrong token");
    if (req.method === "GET" && path === "/api/vault") return sendJson(res, 200, vaultInfo());
    if (req.method === "POST" && path === "/api/vault/open") return sendJson(res, 200, await openVault(await readJson(req)));
    throw new HttpError(404, "not found");
  }

  const server = createServer(async (req, res) => {
    try {
      if (!allowedHosts.has(req.headers.host ?? "")) throw new HttpError(403, "unexpected Host header");
      const path = (req.url ?? "/").split("?")[0] ?? "/";
      if (path === "/api" || path.startsWith("/api/")) return await handleApi(req, res, path);
      if (req.method !== "GET" && req.method !== "HEAD") throw new HttpError(405, "method not allowed");
      const file = serveStatic === undefined ? undefined : await serveStatic(path);
      if (file === undefined) throw new HttpError(404, "not found");
      securityHeaders(res, file.scriptHashes);
      res.writeHead(200, { "Content-Type": file.type, "Content-Length": file.body.length });
      res.end(req.method === "HEAD" ? undefined : file.body);
    } catch (error) {
      if (res.headersSent) return void res.destroy();
      const status = error instanceof HttpError ? error.status : 500;
      // An unexpected failure gets a generic message; details stay in the daemon's log.
      if (status === 500) console.error(error);
      sendJson(res, status, { error: status === 500 ? "internal error" : (error as Error).message });
      if (status === 413) req.destroy();
    }
  });

  await new Promise<void>((resolveListen, reject) => {
    server.once("error", reject);
    server.listen(options.port ?? 0, HOST, () => resolveListen());
  });
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("server did not bind a TCP port");
  const port = address.port;
  allowedHosts = new Set([`127.0.0.1:${port}`, `localhost:${port}`]);

  return {
    host: HOST,
    port,
    close: () =>
      new Promise<void>((resolveClose) => {
        server.closeAllConnections();
        server.close(() => resolveClose());
      }),
  };
}
