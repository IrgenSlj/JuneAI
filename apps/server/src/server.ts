// The June daemon. Loopback only, a random port, one origin for UI and API.
// Rules (PLAN.md, "Daemon rules"): every /api route except health needs the
// per-launch token; a foreign Host header is refused (DNS rebinding); no CORS
// headers at all; a strict CSP that forbids framing.
import { mkdir, stat } from "node:fs/promises";
import { type IncomingMessage, type ServerResponse, createServer } from "node:http";
import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";
import type { Health, OpenVaultRequest, VaultInfo } from "@june/ui/api/types";
import type { SecretStore } from "@june/core";
import { platformSecretStore } from "@june/node";
import type { FetchLike } from "@june/providers";
import { createChatApi } from "./chat.js";
import { readConfig, updateConfig } from "./config.js";
import { HttpError, readJson, sameSecret, securityHeaders, sendJson } from "./http.js";
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
  /** Where API keys live; the macOS keychain by default. */
  secrets?: SecretStore;
  /** The only way out to model providers; global fetch by default. */
  fetch?: FetchLike;
  /** Ollama's base URL if the user moved it (see local-routes.ts). */
  ollamaUrl?: string;
}

/** Global fetch, narrowed to what providers need. */
const globalFetch: FetchLike = async (url, init) => {
  const res = await fetch(url, {
    method: init.method,
    headers: init.headers,
    ...(init.body === undefined ? {} : { body: init.body }),
    ...(init.signal === undefined ? {} : { signal: init.signal as AbortSignal }),
  });
  return { ok: res.ok, status: res.status, text: () => res.text(), body: res.body as AsyncIterable<Uint8Array> | null };
};

export interface RunningServer {
  host: "127.0.0.1";
  port: number;
  close(): Promise<void>;
}

const HOST = "127.0.0.1";
export async function startServer(options: ServerOptions): Promise<RunningServer> {
  const home = options.home ?? homedir();
  const token = Buffer.from(options.token);
  const serveStatic = options.staticDir === undefined ? undefined : createStaticHandler(options.staticDir);
  let vault = (await readConfig(options.configDir)).vault ?? null;
  let allowedHosts = new Set<string>();
  const chat = createChatApi({
    vaultPath: () => vault,
    settings: async () => {
      const { lastChoice } = await readConfig(options.configDir);
      return lastChoice === undefined ? {} : { lastChoice };
    },
    onSaved: async (lastChoice) => void (await updateConfig(options.configDir, { lastChoice })),
    secrets: options.secrets ?? platformSecretStore(), fetch: options.fetch ?? globalFetch,
    ...(options.ollamaUrl === undefined ? {} : { ollamaUrl: options.ollamaUrl }),
  });

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
    await updateConfig(options.configDir, { vault });
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
    if (await chat(req, res, path, new URL(req.url ?? "/", "http://localhost").searchParams)) return;
    throw new HttpError(404, "not found");
  }

  // JUNE_LOG_REQUESTS=1 logs method, path and status to stderr. Never bodies,
  // headers or query strings, so nothing private reaches the log.
  const logRequests = process.env.JUNE_LOG_REQUESTS === "1";

  const server = createServer(async (req, res) => {
    if (logRequests) res.on("finish", () => console.error(`${req.method} ${(req.url ?? "/").split("?")[0]} ${res.statusCode}`));
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
