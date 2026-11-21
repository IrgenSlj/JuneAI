// The thread index: .june/index.sqlite, derived from the thread files and
// nothing else, so deleting .june/ loses nothing (PLAN.md rule 3). It answers
// the thread list and search without parsing every file on every request.
//
// Search uses SQLite FTS5 with the trigram tokenizer (lab 004): "budget"
// finds "Küchenbudget", accents fold, and Japanese works without word breaks.
// Trigrams need three characters, so shorter queries scan with LIKE, which is
// still fast at the size of one person's history.
//
// sync() compares each file's size and modification time with what was
// indexed, so a restart re-reads only files that changed (including edits
// made in Obsidian while June was closed).
import { mkdir, stat } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { type Thread, type VaultFs, parseThread } from "@june/core";

const SCHEMA_VERSION = 1;

export interface IndexedThread {
  id: string;
  path: string;
  title: string;
  source: string;
  started?: string;
  updated?: string;
  parent?: string;
  lane?: string;
  turns: number;
}

export interface SearchHit extends IndexedThread {
  /** The best matching turn, or "title" when only the title matched. */
  turn: string;
  /** Text around the match; the matched span is wrapped in \u0002 and \u0003. */
  snippet: string;
}

export interface SyncReport {
  indexed: number;
  removed: number;
  /** Files that did not parse: left alone on disk, left out of the index. */
  unreadable: string[];
}

export interface ThreadIndex {
  sync(): Promise<SyncReport>;
  /** Re-index one thread file after June wrote it (or forget it when it is gone). */
  refresh(path: string): Promise<void>;
  get(id: string): IndexedThread | undefined;
  /** Newest first, by last update. `before` pages on that same timestamp. */
  list(options?: { before?: string; limit?: number; lane?: string }): IndexedThread[];
  search(query: string, options?: { limit?: number }): SearchHit[];
  close(): void;
}

/** Lowercase without diacritics, with a map from each output position back to the input. */
function fold(text: string): { folded: string; at: number[] } {
  let folded = "";
  const at: number[] = [];
  for (let i = 0; i < text.length; i++) {
    const plain = (text[i] ?? "").normalize("NFKD").replace(/\p{M}/gu, "").toLowerCase();
    for (const ch of plain) {
      folded += ch;
      at.push(i);
    }
  }
  return { folded, at };
}

/** A window of about `width` characters around the first term found, with the term marked. */
export function makeSnippet(text: string, terms: string[], width = 160): string {
  const { folded, at } = fold(text);
  let hit: { start: number; end: number } | undefined;
  for (const term of terms) {
    const f = fold(term).folded;
    const index = f === "" ? -1 : folded.indexOf(f);
    if (index !== -1 && (hit === undefined || at[index]! < hit.start)) hit = { start: at[index]!, end: (at[index + f.length - 1] ?? text.length - 1) + 1 };
  }
  const flat = (s: string) => s.replace(/\s+/g, " ");
  if (hit === undefined) return flat(text.slice(0, width)) + (text.length > width ? "…" : "");
  const from = Math.max(0, hit.start - Math.floor((width - (hit.end - hit.start)) / 2));
  const to = Math.min(text.length, from + width);
  return `${from > 0 ? "…" : ""}${flat(text.slice(from, hit.start))}\u0002${flat(text.slice(hit.start, hit.end))}\u0003${flat(text.slice(hit.end, to))}${to < text.length ? "…" : ""}`;
}

/** Query words, deduplicated; quotes keep a phrase together. */
export function queryTerms(query: string): string[] {
  const terms: string[] = [];
  for (const match of query.matchAll(/"([^"]+)"|(\S+)/g)) {
    const term = (match[1] ?? match[2] ?? "").trim();
    if (term !== "" && !terms.includes(term)) terms.push(term);
  }
  return terms.slice(0, 12);
}

type Row = { [key: string]: unknown };

function toThread(row: Row): IndexedThread {
  const out: IndexedThread = { id: String(row.id), path: String(row.path), title: String(row.title), source: String(row.source), turns: Number(row.turns) };
  for (const key of ["started", "updated", "parent", "lane"] as const) if (typeof row[key] === "string") out[key] = row[key];
  return out;
}

export async function openThreadIndex(root: string, fs: VaultFs, options: { file?: string } = {}): Promise<ThreadIndex> {
  const file = options.file ?? join(root, ".june", "index.sqlite");
  if (file !== ":memory:") await mkdir(join(root, ".june"), { recursive: true });
  const db = new DatabaseSync(file);
  db.exec("PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;");
  const version = Number((db.prepare("PRAGMA user_version").get() as Row).user_version);
  if (version !== SCHEMA_VERSION) {
    // Derived data: on any schema change, drop it and rebuild from the files.
    db.exec("DROP TABLE IF EXISTS threads; DROP TABLE IF EXISTS text_fts;");
    db.exec(`
      CREATE TABLE threads (
        id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, title TEXT NOT NULL, source TEXT NOT NULL,
        started TEXT, updated TEXT, sort TEXT NOT NULL, parent TEXT, lane TEXT, turns INTEGER NOT NULL,
        size INTEGER NOT NULL, mtime REAL NOT NULL
      );
      CREATE INDEX threads_sort ON threads (sort DESC);
      CREATE VIRTUAL TABLE text_fts USING fts5(body, thread_id UNINDEXED, turn_id UNINDEXED, tokenize = 'trigram remove_diacritics 1');
      PRAGMA user_version = ${SCHEMA_VERSION};
    `);
  }

  const known = db.prepare("SELECT path, size, mtime FROM threads");
  const removeByPath = db.prepare("SELECT id FROM threads WHERE path = ?");
  const deleteThread = db.prepare("DELETE FROM threads WHERE id = ?");
  const deleteText = db.prepare("DELETE FROM text_fts WHERE thread_id = ?");
  const insertThread = db.prepare(
    `INSERT OR REPLACE INTO threads (id, path, title, source, started, updated, sort, parent, lane, turns, size, mtime)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  );
  const insertText = db.prepare("INSERT INTO text_fts (body, thread_id, turn_id) VALUES (?, ?, ?)");

  function forget(path: string): void {
    const row = removeByPath.get(path) as Row | undefined;
    if (row === undefined) return;
    deleteText.run(String(row.id));
    deleteThread.run(String(row.id));
  }

  function put(path: string, thread: Thread, size: number, mtime: number): void {
    forget(path);
    // A copied file with the same id: the newer path wins, the other is dropped from the index.
    deleteText.run(thread.id);
    deleteThread.run(thread.id);
    const sort = thread.turns.at(-1)?.at ?? thread.updated ?? thread.started ?? "";
    insertThread.run(thread.id, path, thread.title, thread.source, thread.started ?? null, thread.updated ?? null, sort, thread.parent ?? null, thread.lane ?? null, thread.turns.length, size, mtime);
    insertText.run(thread.title, thread.id, "title");
    for (const turn of thread.turns) if (turn.text !== "") insertText.run(turn.text, thread.id, turn.id);
  }

  async function index(path: string): Promise<boolean> {
    const info = await stat(join(root, path)).catch(() => undefined);
    if (info === undefined) {
      forget(path);
      return true;
    }
    const text = await fs.read(path);
    if (text === undefined) return true;
    put(path, parseThread(text), info.size, info.mtimeMs);
    return true;
  }

  const transaction = <T>(work: () => T): T => {
    db.exec("BEGIN");
    try {
      const out = work();
      db.exec("COMMIT");
      return out;
    } catch (error) {
      db.exec("ROLLBACK");
      throw error;
    }
  };

  return {
    async sync() {
      const report: SyncReport = { indexed: 0, removed: 0, unreadable: [] };
      const files = (await fs.list("threads")).filter((p) => p.endsWith(".md"));
      const seen = new Set(files);
      const before = new Map((known.all() as Row[]).map((r) => [String(r.path), { size: Number(r.size), mtime: Number(r.mtime) }]));
      // Read changed files first (async), then write them in one transaction.
      const changed: { path: string; thread: Thread; size: number; mtime: number }[] = [];
      for (const path of files) {
        const info = await stat(join(root, path)).catch(() => undefined);
        if (info === undefined) continue;
        const old = before.get(path);
        if (old !== undefined && old.size === info.size && old.mtime === info.mtimeMs) continue;
        try {
          changed.push({ path, thread: parseThread((await fs.read(path)) ?? ""), size: info.size, mtime: info.mtimeMs });
        } catch {
          report.unreadable.push(path);
        }
      }
      transaction(() => {
        for (const path of before.keys()) {
          if (!seen.has(path)) {
            forget(path);
            report.removed++;
          }
        }
        for (const { path, thread, size, mtime } of changed) put(path, thread, size, mtime);
      });
      report.indexed = changed.length;
      return report;
    },

    async refresh(path) {
      await index(path).catch(() => {
        forget(path);
      });
    },

    get(id) {
      const row = db.prepare("SELECT * FROM threads WHERE id = ?").get(id) as Row | undefined;
      return row === undefined ? undefined : toThread(row);
    },

    list({ before, limit = 200, lane } = {}) {
      const rows = db
        .prepare(`SELECT * FROM threads WHERE (? IS NULL OR sort < ?) AND (? IS NULL OR lane = ?) ORDER BY sort DESC, id DESC LIMIT ?`)
        .all(before ?? null, before ?? null, lane ?? null, lane ?? null, Math.min(Math.max(limit, 1), 1000)) as Row[];
      return rows.map(toThread);
    },

    search(query, { limit = 50 } = {}) {
      const terms = queryTerms(query);
      if (terms.length === 0) return [];
      const long = terms.filter((t) => [...t].length >= 3);
      let rows: Row[];
      if (long.length === terms.length) {
        const match = terms.map((t) => `"${t.replace(/"/g, '""')}"`).join(" AND ");
        // Rank turns by bm25 (FTS5's rank column), then keep each thread's best turn.
        const ranked = db.prepare("SELECT rowid, thread_id FROM text_fts WHERE text_fts MATCH ? ORDER BY rank LIMIT 5000").all(match) as Row[];
        const best = new Map<string, unknown>();
        for (const row of ranked) {
          if (best.size >= limit) break;
          if (!best.has(String(row.thread_id))) best.set(String(row.thread_id), row.rowid);
        }
        const byRowid = db.prepare("SELECT thread_id, turn_id, body FROM text_fts WHERE rowid = ?");
        rows = [...best.values()].map((rowid) => byRowid.get(rowid as number) as Row);
      } else {
        // A term under three characters: every term as a substring, newest threads first.
        const where = terms.map(() => "body LIKE ? ESCAPE '\\'").join(" AND ");
        const args = terms.map((t) => `%${t.replace(/[\\%_]/g, (c) => `\\${c}`)}%`);
        rows = db
          .prepare(
            `SELECT f.thread_id, MIN(f.turn_id) AS turn_id, f.body FROM text_fts f JOIN threads t ON t.id = f.thread_id
             WHERE ${where} GROUP BY f.thread_id ORDER BY MAX(t.sort) DESC LIMIT ?`,
          )
          .all(...args, limit) as Row[];
      }
      const getThread = db.prepare("SELECT * FROM threads WHERE id = ?");
      return rows.flatMap((row) => {
        const thread = getThread.get(String(row.thread_id)) as Row | undefined;
        if (thread === undefined) return [];
        return [{ ...toThread(thread), turn: String(row.turn_id), snippet: makeSnippet(String(row.body), terms) }];
      });
    },

    close() {
      db.close();
    },
  };
}
