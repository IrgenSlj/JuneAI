// The chat API: routes and keys, threads, and the streamed turn (PLAN.md,
// "HTTP API"). Thin: the engine runs the turn, providers speak to models, and
// this file only maps HTTP onto them.
//
// Keys go to the secret store (the macOS keychain) as "route:<id>" and never
// come back out through the API. A turn streams as SSE over POST, one JSON
// object per data: line, in the order label, text*, usage, saved | error, done.
import type { IncomingMessage, ServerResponse } from "node:http";
import { realpath, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, dirname, isAbsolute, join, relative, sep } from "node:path";
import {
  type ReceiptLog,
  type SecretStore,
  SecretStoreError,
  type Thread,
  type VaultFs,
  type VaultLock,
  createReceiptLog,
  findClaudeData,
  isId,
  parseClaudeExport,
  parseThread,
  rawPathFor,
  writeImported,
} from "@june/core";
import { branchThread, buildLabel, createThread, runTurn } from "@june/engine";
import { type ExportDocument, ExportReadError, type IndexedThread, type ThreadIndex, documentsIn, nodeFileLock, nodeVaultFs, openThreadIndex, readExportPath } from "@june/node";
import { type FetchLike, ROUTE_PRESETS, makeProvider } from "@june/providers";
import type {
  BranchRequest,
  ExportToPathRequest,
  ImportReport,
  ImportRequest,
  LabelRequest,
  ModelList,
  NewThreadRequest,
  ProviderInfo,
  PutProviderRequest,
  SendTurnRequest,
  Settings,
  SettingsChange,
  ThreadDetail,
  ThreadSummary,
  TurnFrame,
  TurnInfo,
} from "@june/ui/api/types";
import { HttpError, readBody, readJson, securityHeaders, sendJson } from "./http.js";
import { exportFilename, threadAsJson, threadAsMarkdown } from "./thread-export.js";
import { relocate } from "./local-routes.js";

interface VaultSession {
  root: string;
  fs: VaultFs;
  lock: VaultLock;
  receipts: ReceiptLog;
  index: ThreadIndex;
  /** When the index last compared itself with the files (ms). */
  syncedAt: number;
}

const MAX_UPLOAD = 1_500_000_000;
/** How often a list request re-checks the files for edits made outside June. */
const RESYNC_MS = 3_000;

export interface ChatApiOptions {
  /** The open vault's absolute path, or null. */
  vaultPath(): string | null;
  secrets: SecretStore;
  fetch: FetchLike;
  /** Ollama's base URL when OLLAMA_HOST moved it (loopback only). */
  ollamaUrl?: string;
  /** The daemon's settings, for GET /api/settings. */
  settings(): Promise<Settings>;
  /** Apply a validated change from PUT /api/settings. */
  saveSettings(change: SettingsChange): Promise<Settings>;
  /** Called after a reply is saved, with the route and model that produced it. */
  onSaved(choice: { route: string; model: string }): Promise<void>;
}

const secretAccount = (routeId: string): string => `route:${routeId}`;

function indexedSummary(thread: IndexedThread): ThreadSummary {
  const { path: _path, ...rest } = thread;
  return rest;
}

function summary(thread: Thread): ThreadSummary {
  return {
    id: thread.id,
    title: thread.title,
    source: thread.source,
    ...(thread.started === undefined ? {} : { started: thread.started }),
    ...(thread.updated === undefined ? {} : { updated: thread.updated }),
    ...(thread.parent === undefined ? {} : { parent: thread.parent }),
    ...(thread.lane === undefined ? {} : { lane: thread.lane }),
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
  const presets = ROUTE_PRESETS.map((p) => (p.route.id === "ollama" ? relocate(p, options.ollamaUrl) : p));
  const preset = (id: string) => presets.find((p) => p.route.id === id);
  /** Threads with a reply streaming right now, so a second send can be told so plainly. */
  const busy = new Set<string>();

  async function vault(): Promise<VaultSession> {
    const root = options.vaultPath();
    if (root === null) throw new HttpError(409, "open a vault first");
    if (session?.root !== root) {
      session?.index.close();
      session = undefined;
      const fs = await nodeVaultFs(root);
      const lock = nodeFileLock(root);
      const index = await openThreadIndex(root, fs);
      await index.sync();
      session = { root, fs, lock, receipts: createReceiptLog(fs, { lock }), index, syncedAt: Date.now() };
    }
    return session;
  }

  /** The index, re-checked against the files at most every few seconds. */
  async function freshIndex(): Promise<ThreadIndex> {
    const current = await vault();
    if (Date.now() - current.syncedAt > RESYNC_MS) {
      await current.index.sync();
      current.syncedAt = Date.now();
    }
    return current.index;
  }

  /** The documents an import request carries: a path on this Mac (JSON body) or an uploaded file. */
  async function importDocuments(req: IncomingMessage): Promise<ExportDocument[]> {
    const type = String(req.headers["content-type"] ?? "");
    try {
      if (type.startsWith("application/json") && req.headers["x-june-filename"] === undefined) {
        const path = ((await readJson(req)) as Partial<ImportRequest> | null)?.path;
        if (typeof path !== "string" || path.trim() === "") throw new HttpError(400, "path is required");
        return await readExportPath(path.trim().replace(/^~(?=\/|$)/, homedir()));
      }
      const name = decodeURIComponent(String(req.headers["x-june-filename"] ?? "upload"));
      return documentsIn(name, new Uint8Array(await readBody(req, MAX_UPLOAD)));
    } catch (error) {
      if (error instanceof ExportReadError) throw new HttpError(400, error.message);
      throw error;
    }
  }

  async function importExport(req: IncomingMessage): Promise<ImportReport> {
    const documents = await importDocuments(req);
    const found = findClaudeData(documents);
    if (found.manifestOnly) {
      throw new HttpError(400, "This is the export's list of files, not the conversations. Download the zip files it lists (at least conversations-000.zip) and import those.");
    }
    if (found.conversations.length === 0) throw new HttpError(400, "No Claude conversations found. Choose the export zip, or the folder you unzipped it into.");
    const { fs, receipts, lock, index } = await vault();
    const { threads, report } = await parseClaudeExport(found);
    const written = await lock.exclusive("import", () => writeImported(fs, receipts, threads, { source: "claude-export" }));
    await index.sync();
    return { source: "claude-export", ...report, ...written, ignored: found.ignored };
  }

  /**
   * Delete, recoverably: the thread file and its raw JSON move to .trash/ in
   * the vault under the same path (Obsidian uses the same folder), so Finder
   * or Obsidian can bring them back. Nothing is erased.
   */
  async function trashThread(id: string): Promise<{ trashedTo: string }> {
    if (busy.has(id)) throw new HttpError(409, "a reply is still arriving in this thread; wait for it to finish");
    const { fs, index } = await vault();
    const { path } = await findThread(fs, id);
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    let trashedTo = "";
    for (const from of [rawPathFor(path), path]) {
      const text = await fs.read(from);
      if (text === undefined) continue;
      let to = `.trash/${from}`;
      if ((await fs.read(to)) !== undefined) to = to.replace(/(\.raw\.json|\.md)$/, `-${stamp}$1`);
      await fs.write(to, text);
      await fs.remove(from);
      if (from === path) trashedTo = to;
    }
    await index.refresh(path);
    return { trashedTo };
  }

  /** The label for a draft, in `thread` or in a conversation not created yet. Sends nothing, writes nothing. */
  async function previewLabel(request: unknown, thread?: Thread) {
    const body = request as Partial<LabelRequest> | null;
    if (typeof body?.route !== "string" || typeof body.model !== "string" || typeof body.text !== "string") throw new HttpError(400, "route, model and text are required");
    const chosen = await provider(body.route);
    const empty: Thread = { id: "th_new", title: "", source: "june", turns: [] };
    return buildLabel(thread ?? empty, chosen.route, body.model, body.text).label;
  }

  /** One thread rendered for export. */
  async function exported(id: string, format: string): Promise<{ thread: Thread; body: string; format: "md" | "json" }> {
    if (format !== "md" && format !== "json") throw new HttpError(400, "format must be md or json");
    const { fs } = await vault();
    const { thread } = await findThread(fs, id);
    return { thread, body: format === "md" ? threadAsMarkdown(thread) : threadAsJson(thread), format };
  }

  async function exportReceipt(thread: Thread, body: string, format: string, destination: "file" | "download"): Promise<void> {
    await (await vault()).receipts.append({ kind: "export", actor: "user", targets: [thread.id], payload: body, meta: { format, bytes: Buffer.byteLength(body), to: destination } });
  }

  /** The Mac app's save dialog chose `path`; write the export there. */
  async function exportToPath(id: string, request: unknown): Promise<{ path: string }> {
    const { format, path } = (request ?? {}) as Partial<ExportToPathRequest>;
    if (typeof path !== "string" || !isAbsolute(path)) throw new HttpError(400, "path must be an absolute path");
    if (!path.toLowerCase().endsWith(`.${String(format)}`)) throw new HttpError(400, `the file name must end in .${String(format)}`);
    const folder = await stat(dirname(path)).catch(() => undefined);
    if (folder?.isDirectory() !== true) throw new HttpError(400, "that folder does not exist");
    // Never into the vault: an export there could overwrite a thread, a note or the index.
    const real = await realpath(dirname(path));
    const fromVault = relative(await realpath((await vault()).root), real);
    if (fromVault === "" || (!fromVault.startsWith(`..${sep}`) && fromVault !== ".." && !isAbsolute(fromVault))) {
      throw new HttpError(400, "choose a folder outside your vault; the vault already holds this conversation");
    }
    const { thread, body } = await exported(id, String(format));
    // The file first: a receipt must not claim an export that failed to write.
    await writeFile(join(real, basename(path)), body, "utf8");
    await exportReceipt(thread, body, String(format), "file");
    return { path };
  }

  async function exportThread(res: ServerResponse, id: string, requested: string): Promise<void> {
    const { thread, body, format } = await exported(id, requested);
    await exportReceipt(thread, body, format, "download");
    securityHeaders(res);
    res.writeHead(200, {
      "Content-Type": format === "md" ? "text/markdown; charset=utf-8" : "application/json; charset=utf-8",
      "Content-Disposition": `attachment; filename="${exportFilename(thread, format)}"`,
    });
    res.end(body);
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

  /** Whether a local server answers. A closed port fails in a millisecond; the timeout covers a hung one. */
  async function answers(baseUrl: string): Promise<boolean> {
    try {
      const res = await options.fetch(`${baseUrl}/models`, { method: "GET", headers: {}, signal: AbortSignal.timeout(800) });
      return res.ok;
    } catch {
      return false;
    }
  }

  async function providers(): Promise<ProviderInfo[]> {
    return Promise.all(
      presets.map(async (p): Promise<ProviderInfo> => {
        if (p.route.kind === "local") {
          const running = await answers(p.baseUrl);
          return { route: p.route, needsKey: false, ready: running, running };
        }
        return { route: p.route, needsKey: p.needsKey, ready: !p.needsKey || (await options.secrets.get(secretAccount(p.route.id))) !== undefined };
      }),
    );
  }

  async function provider(routeId: string) {
    const chosen = preset(routeId);
    if (chosen === undefined) throw new HttpError(404, `no route called ${routeId}`);
    const apiKey = chosen.needsKey ? await options.secrets.get(secretAccount(routeId)) : undefined;
    try {
      return makeProvider(chosen, { fetch: options.fetch, ...(apiKey === undefined ? {} : { apiKey }) });
    } catch (error) {
      throw new HttpError(400, (error as Error).message);
    }
  }

  async function putProvider(routeId: string, body: unknown): Promise<ProviderInfo[]> {
    if (preset(routeId) === undefined) throw new HttpError(404, `no route called ${routeId}`);
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
    const { fs, receipts, lock, index } = await vault();
    const { path, thread } = await findThread(fs, threadId);
    if (thread.source !== "june") throw new HttpError(409, "This conversation was imported, so it stays as it was. Continue it as a new branch.");
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
          await options.onSaved({ route: chosen.route.id, model: body.model }).catch((error: unknown) => console.error(error));
          send({ type: "saved", threadId: event.threadId, userTurn: event.userTurn, assistantTurn: event.assistantTurn, receipt: event.receipt });
        } else {
          send(event);
        }
      }
    } finally {
      busy.delete(threadId);
      await index.refresh(path).catch(() => undefined);
      res.end();
    }
  }

  /** Handles a chat route, or returns false so the caller can 404. The token is already checked. */
  return async function handle(req: IncomingMessage, res: ServerResponse, path: string, query = new URLSearchParams()): Promise<boolean> {
    const method = req.method ?? "GET";

    if (path === "/api/settings" && method === "GET") return sendJson(res, 200, await options.settings()), true;
    if (path === "/api/settings" && method === "PUT") {
      const { transparency } = ((await readJson(req)) ?? {}) as Partial<SettingsChange>;
      if (transparency !== undefined && transparency !== "quiet" && transparency !== "glance" && transparency !== "full") {
        throw new HttpError(400, "transparency must be quiet, glance or full");
      }
      return sendJson(res, 200, await options.saveSettings(transparency === undefined ? {} : { transparency })), true;
    }
    if (path === "/api/receipts" && method === "GET") {
      const after = Number(query.get("after") ?? "0");
      if (!Number.isInteger(after) || after < 0) throw new HttpError(400, "after must be a whole number");
      return sendJson(res, 200, await (await vault()).receipts.list({ after })), true;
    }
    if (path === "/api/receipts/verify" && method === "POST") return sendJson(res, 200, await (await vault()).receipts.verify()), true;
    if (path === "/api/import" && method === "POST") return sendJson(res, 200, await importExport(req)), true;
    if (path === "/api/label" && method === "POST") return sendJson(res, 200, await previewLabel(await readJson(req))), true;
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
        // Files that do not parse are left alone and left out of the index; they are still the user's files.
        const index = await freshIndex();
        const q = query.get("q")?.trim() ?? "";
        const limit = Number(query.get("limit") ?? "200");
        if (!Number.isInteger(limit) || limit < 1) throw new HttpError(400, "limit must be a whole number");
        if (q !== "") {
          const hits = index.search(q, { limit: Math.min(limit, 200) });
          return sendJson(res, 200, hits.map(({ turn, snippet, ...thread }): ThreadSummary => ({ ...indexedSummary(thread), match: { turn, snippet } }))), true;
        }
        const before = query.get("before") ?? undefined;
        const beforeId = query.get("beforeId") ?? undefined;
        const lane = query.get("lane") ?? undefined;
        return sendJson(res, 200, index.list({ limit, ...(before === undefined ? {} : { before }), ...(beforeId === undefined ? {} : { beforeId }), ...(lane === undefined ? {} : { lane }) }).map(indexedSummary)), true;
      }
      if (method === "POST" && parts.length === 1) {
        const { fs, index } = await vault();
        const raw = ((await readJson(req)) as Partial<NewThreadRequest> | null)?.title;
        const title = typeof raw === "string" && raw.trim() !== "" ? raw.trim().slice(0, 200) : "New conversation";
        const { path, thread } = await createThread(fs, { title });
        await index.refresh(path);
        return sendJson(res, 201, summary(thread)), true;
      }
      const id = parts[1];
      if (id === undefined) return false;
      if (method === "GET" && parts.length === 2) {
        const { fs } = await vault();
        return sendJson(res, 200, detail((await findThread(fs, id)).thread)), true;
      }
      if (method === "DELETE" && parts.length === 2) return sendJson(res, 200, await trashThread(id)), true;
      if (method === "POST" && parts.length === 3 && parts[2] === "turns") return await sendTurn(req, res, id), true;
      if (method === "POST" && parts.length === 3 && parts[2] === "label") {
        const { fs } = await vault();
        const { thread } = await findThread(fs, id);
        return sendJson(res, 200, await previewLabel(await readJson(req), thread)), true;
      }
      if (method === "GET" && parts.length === 3 && parts[2] === "export") return await exportThread(res, id, query.get("format") ?? "md"), true;
      if (method === "POST" && parts.length === 3 && parts[2] === "export") return sendJson(res, 200, await exportToPath(id, await readJson(req))), true;
      if (method === "POST" && parts.length === 3 && parts[2] === "branch") {
        const turn = ((await readJson(req)) as Partial<BranchRequest> | null)?.turn;
        if (typeof turn !== "string") throw new HttpError(400, "turn is required");
        const { fs, index } = await vault();
        const { path } = await findThread(fs, id);
        try {
          const branched = await branchThread(fs, { path, turnId: turn });
          await index.refresh(branched.path);
          return sendJson(res, 201, summary(branched.thread)), true;
        } catch (error) {
          throw new HttpError(400, (error as Error).message);
        }
      }
    }
    return false;
  };
}
