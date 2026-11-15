// The chat API: routes and keys, threads, and the streamed turn (PLAN.md,
// "HTTP API"). Thin: the engine runs the turn, providers speak to models, and
// this file only maps HTTP onto them.
//
// Keys go to the secret store (the macOS keychain) as "route:<id>" and never
// come back out through the API. A turn streams as SSE over POST, one JSON
// object per data: line, in the order label, text*, usage, saved | error, done.
import type { IncomingMessage, ServerResponse } from "node:http";
import {
  type ReceiptLog,
  type SecretStore,
  SecretStoreError,
  type Thread,
  type VaultFs,
  type VaultLock,
  createReceiptLog,
  isId,
  parseThread,
} from "@june/core";
import { createThread, runTurn } from "@june/engine";
import { nodeFileLock, nodeVaultFs } from "@june/node";
import { type FetchLike, ROUTE_PRESETS, findPreset, makeProvider } from "@june/providers";
import type {
  ModelList,
  NewThreadRequest,
  ProviderInfo,
  PutProviderRequest,
  SendTurnRequest,
  ThreadDetail,
  ThreadSummary,
  TurnFrame,
  TurnInfo,
} from "@june/ui/api/types";
import { HttpError, readJson, securityHeaders, sendJson } from "./http.js";

interface VaultSession {
  root: string;
  fs: VaultFs;
  lock: VaultLock;
  receipts: ReceiptLog;
}

export interface ChatApiOptions {
  /** The open vault's absolute path, or null. */
  vaultPath(): string | null;
  secrets: SecretStore;
  fetch: FetchLike;
}

const secretAccount = (routeId: string): string => `route:${routeId}`;

function summary(thread: Thread): ThreadSummary {
  return {
    id: thread.id,
    title: thread.title,
    source: thread.source,
    ...(thread.started === undefined ? {} : { started: thread.started }),
    ...(thread.updated === undefined ? {} : { updated: thread.updated }),
    ...(thread.parent === undefined ? {} : { parent: thread.parent }),
    turns: thread.turns.length,
  };
}

function detail(thread: Thread): ThreadDetail {
  const { turns: _count, ...rest } = summary(thread);
  const turns: TurnInfo[] = thread.turns.map((t) => ({
    id: t.id,
    role: t.role,
    ...(t.at === undefined ? {} : { at: t.at }),
    ...(t.model === undefined ? {} : { model: t.model }),
    ...(typeof t.meta?.route === "string" ? { route: t.meta.route } : {}),
    ...(typeof t.meta?.receipt === "number" ? { receipt: t.meta.receipt } : {}),
    text: t.text,
  }));
  return { ...rest, turns };
}

export function createChatApi(options: ChatApiOptions) {
  let session: VaultSession | undefined;
  /** Threads with a reply streaming right now, so a second send can be told so plainly. */
  const busy = new Set<string>();

  async function vault(): Promise<VaultSession> {
    const root = options.vaultPath();
    if (root === null) throw new HttpError(409, "open a vault first");
    if (session?.root !== root) {
      const fs = await nodeVaultFs(root);
      const lock = nodeFileLock(root);
      session = { root, fs, lock, receipts: createReceiptLog(fs, { lock }) };
    }
    return session;
  }

  async function threadFiles(fs: VaultFs): Promise<string[]> {
    return (await fs.list("threads")).filter((p) => p.endsWith(".md"));
  }

  /** Thread files end in the id's last five characters, so a lookup parses only the few candidates. */
  async function findThread(fs: VaultFs, id: string): Promise<{ path: string; thread: Thread }> {
    if (!isId(id, "thread")) throw new HttpError(404, "no such thread");
    const tail = `-${id.slice(-5).toLowerCase()}.md`;
    for (const path of (await threadFiles(fs)).filter((p) => p.endsWith(tail))) {
      const thread = parseThread((await fs.read(path)) ?? "");
      if (thread.id === id) return { path, thread };
    }
    throw new HttpError(404, "no such thread");
  }

  async function providers(): Promise<ProviderInfo[]> {
    return Promise.all(
      ROUTE_PRESETS.map(async (preset) => ({
        route: preset.route,
        needsKey: preset.needsKey,
        ready: !preset.needsKey || (await options.secrets.get(secretAccount(preset.route.id))) !== undefined,
      })),
    );
  }

  async function provider(routeId: string) {
    const preset = findPreset(routeId);
    if (preset === undefined) throw new HttpError(404, `no route called ${routeId}`);
    const apiKey = preset.needsKey ? await options.secrets.get(secretAccount(routeId)) : undefined;
    try {
      return makeProvider(preset, { fetch: options.fetch, ...(apiKey === undefined ? {} : { apiKey }) });
    } catch (error) {
      throw new HttpError(400, (error as Error).message);
    }
  }

  async function putProvider(routeId: string, body: unknown): Promise<ProviderInfo[]> {
    const preset = findPreset(routeId);
    if (preset === undefined) throw new HttpError(404, `no route called ${routeId}`);
    const apiKey = (body as Partial<PutProviderRequest> | null)?.apiKey;
    if (apiKey !== null && typeof apiKey !== "string") throw new HttpError(400, "apiKey must be a string, or null to remove it");
    try {
      if (apiKey === null) await options.secrets.remove(secretAccount(routeId));
      else await options.secrets.set(secretAccount(routeId), apiKey);
    } catch (error) {
      if (error instanceof SecretStoreError) throw new HttpError(400, error.message);
      throw error;
    }
    return providers();
  }

  async function sendTurn(req: IncomingMessage, res: ServerResponse, threadId: string): Promise<void> {
    const body = (await readJson(req)) as Partial<SendTurnRequest> | null;
    if (typeof body?.route !== "string" || typeof body.model !== "string" || typeof body.text !== "string" || body.model === "") {
      throw new HttpError(400, "route, model and text are required");
    }
    const { fs, receipts, lock } = await vault();
    const { path } = await findThread(fs, threadId);
    const chosen = await provider(body.route);
    if (busy.has(threadId)) throw new HttpError(409, "a reply is still arriving in this thread; wait for it to finish");

    busy.add(threadId);
    const abort = new AbortController();
    res.on("close", () => abort.abort());
    securityHeaders(res);
    res.writeHead(200, { "Content-Type": "text/event-stream; charset=utf-8", Connection: "keep-alive" });
    const send = (frame: TurnFrame) => res.write(`data: ${JSON.stringify(frame)}\n\n`);
    try {
      for await (const event of runTurn({ fs, receipts, lock, provider: chosen }, { path, model: body.model, text: body.text, signal: abort.signal })) {
        if (abort.signal.aborted) break;
        if (event.type === "saved") {
          send({ type: "saved", threadId: event.threadId, userTurn: event.userTurn, assistantTurn: event.assistantTurn, receipt: event.receipt });
        } else {
          send(event);
        }
      }
    } finally {
      busy.delete(threadId);
      res.end();
    }
  }

  /** Handles a chat route, or returns false so the caller can 404. The token is already checked. */
  return async function handle(req: IncomingMessage, res: ServerResponse, path: string): Promise<boolean> {
    const method = req.method ?? "GET";
    const parts = path.split("/").slice(2).map(decodeURIComponent); // ["providers", id?, "models"?]

    if (parts[0] === "providers") {
      if (method === "GET" && parts.length === 1) return sendJson(res, 200, await providers()), true;
      if (method === "PUT" && parts.length === 2 && parts[1] !== undefined) return sendJson(res, 200, await putProvider(parts[1], await readJson(req))), true;
      if (method === "GET" && parts.length === 3 && parts[2] === "models" && parts[1] !== undefined) {
        const list: ModelList = { models: await (await provider(parts[1])).listModels() };
        return sendJson(res, 200, list), true;
      }
      return false;
    }

    if (parts[0] === "threads") {
      if (method === "GET" && parts.length === 1) {
        const { fs } = await vault();
        const threads: ThreadSummary[] = [];
        for (const file of await threadFiles(fs)) {
          try {
            threads.push(summary(parseThread((await fs.read(file)) ?? "")));
          } catch {
            // A file that does not parse is left alone and left out; it is still the user's file.
          }
        }
        threads.sort((a, b) => (b.updated ?? b.started ?? "").localeCompare(a.updated ?? a.started ?? ""));
        return sendJson(res, 200, threads), true;
      }
      if (method === "POST" && parts.length === 1) {
        const { fs } = await vault();
        const raw = ((await readJson(req)) as Partial<NewThreadRequest> | null)?.title;
        const title = typeof raw === "string" && raw.trim() !== "" ? raw.trim().slice(0, 200) : "New conversation";
        const { thread } = await createThread(fs, { title });
        return sendJson(res, 201, summary(thread)), true;
      }
      const id = parts[1];
      if (id === undefined) return false;
      if (method === "GET" && parts.length === 2) {
        const { fs } = await vault();
        return sendJson(res, 200, detail((await findThread(fs, id)).thread)), true;
      }
      if (method === "POST" && parts.length === 3 && parts[2] === "turns") return await sendTurn(req, res, id), true;
    }
    return false;
  };
}
