// The thread file: one conversation as Markdown that stays readable in any
// editor and parses back to exactly what was written.
//
//   ---
//   id: "th_01JD3..."
//   title: "Kitchen quotes"
//   source: "claude-export"
//   ---
//
//   <!-- june:turn {"id":"m1","role":"user","at":"2026-09-02T18:04:11Z"} -->
//   ## User
//
//   What should a 12 m2 kitchen cost?
//
// Frontmatter values are JSON, which is also valid YAML, so Obsidian reads
// them. Each turn starts at a marker comment (invisible when rendered); the
// heading under it is for human readers and is ignored when parsing.
import type { Json } from "./hash.js";

export type Role = "user" | "assistant" | "system" | "tool";
export type ThreadSource = "june" | "claude-export" | "chatgpt-export" | "lechat-export" | "claude-code" | "capture";

export interface Turn {
  /** Stable within the thread. For imports, derived from the provider's message id. */
  id: string;
  role: Role;
  /** ISO 8601, the provider's original time. Never the import time. */
  at?: string;
  /** Model that produced this turn, exactly as the source states it. */
  model?: string;
  /** Anything else worth keeping (route, receipt seq, attached note ids). */
  meta?: { [key: string]: Json };
  text: string;
}

export interface Thread {
  id: string;
  title: string;
  source: ThreadSource;
  /** The provider's own conversation id; with `source`, the dedupe key. */
  sourceId?: string;
  lane?: string;
  started?: string;
  updated?: string;
  /** "th_...#m7" when this thread branches from a turn of another. */
  parent?: string;
  turns: Turn[];
}

const MARKER = "<!-- june:turn ";
const MARKER_END = " -->";
const HEADINGS: Record<Role, string> = { user: "User", assistant: "Assistant", system: "System", tool: "Tool" };
const FRONTMATTER_KEYS = ["id", "title", "source", "sourceId", "lane", "started", "updated", "parent"] as const;

export class ThreadFormatError extends Error {
  constructor(message: string) {
    super(`thread file: ${message}`);
    this.name = "ThreadFormatError";
  }
}

// ">" is escaped so a value containing "-->" cannot close the marker comment.
const markerJson = (value: Json): string => JSON.stringify(value).replace(/>/g, "\\u003e");

// A body line that looks like a marker gets one leading backslash; parsing
// removes one. Applying it to already-escaped lines keeps it reversible.
const ESCAPABLE = /^\\*<!-- june:turn /;
const escapeLine = (line: string): string => (ESCAPABLE.test(line) ? `\\${line}` : line);
const unescapeLine = (line: string): string => (/^\\+<!-- june:turn /.test(line) ? line.slice(1) : line);

export function serializeThread(thread: Thread): string {
  const out: string[] = ["---"];
  for (const key of FRONTMATTER_KEYS) {
    const value = thread[key];
    if (value !== undefined) out.push(`${key}: ${JSON.stringify(value)}`);
  }
  out.push("---", "");
  for (const turn of thread.turns) {
    const head: { [key: string]: Json } = { id: turn.id, role: turn.role };
    if (turn.at !== undefined) head.at = turn.at;
    if (turn.model !== undefined) head.model = turn.model;
    if (turn.meta !== undefined) head.meta = turn.meta;
    out.push(`${MARKER}${markerJson(head)}${MARKER_END}`, `## ${HEADINGS[turn.role]}`, "");
    out.push(...turn.text.split("\n").map(escapeLine), "");
  }
  return `${out.join("\n")}\n`;
}

export function parseThread(file: string): Thread {
  const lines = (file.endsWith("\n") ? file.slice(0, -1) : file).split("\n");
  if (lines[0] !== "---") throw new ThreadFormatError("missing frontmatter");
  const end = lines.indexOf("---", 1);
  if (end === -1) throw new ThreadFormatError("unterminated frontmatter");

  const front: { [key: string]: Json } = {};
  for (const line of lines.slice(1, end)) {
    const split = line.indexOf(": ");
    if (split === -1) throw new ThreadFormatError(`bad frontmatter line: ${line}`);
    try {
      front[line.slice(0, split)] = JSON.parse(line.slice(split + 2)) as Json;
    } catch {
      throw new ThreadFormatError(`frontmatter value is not JSON: ${line}`);
    }
  }
  const text = (key: string): string | undefined => {
    const value = front[key];
    if (value === undefined) return undefined;
    if (typeof value !== "string") throw new ThreadFormatError(`${key} must be a string`);
    return value;
  };
  const id = text("id");
  const title = text("title");
  const source = text("source");
  if (id === undefined || title === undefined || source === undefined) {
    throw new ThreadFormatError("id, title and source are required");
  }

  const turns: Turn[] = [];
  let current: { turn: Turn; body: string[] } | undefined;
  const close = (): void => {
    if (current === undefined) return;
    const body = current.body;
    // Drop the cosmetic heading and its blank line, and the separator line.
    const start = /^## (User|Assistant|System|Tool)$/.test(body[0] ?? "") && body[1] === "" ? 2 : 0;
    const stop = body.at(-1) === "" ? body.length - 1 : body.length;
    current.turn.text = body.slice(start, stop).map(unescapeLine).join("\n");
    turns.push(current.turn);
  };
  for (const line of lines.slice(end + 1)) {
    if (line.startsWith(MARKER) && line.endsWith(MARKER_END)) {
      close();
      let head: { [key: string]: Json };
      try {
        head = JSON.parse(line.slice(MARKER.length, -MARKER_END.length)) as { [key: string]: Json };
      } catch {
        throw new ThreadFormatError(`bad turn marker: ${line}`);
      }
      const { id: turnId, role, at, model, meta } = head;
      if (typeof turnId !== "string" || typeof role !== "string" || !(role in HEADINGS)) {
        throw new ThreadFormatError(`turn marker needs id and role: ${line}`);
      }
      const turn: Turn = { id: turnId, role: role as Role, text: "" };
      if (typeof at === "string") turn.at = at;
      if (typeof model === "string") turn.model = model;
      if (meta !== undefined && meta !== null && typeof meta === "object" && !Array.isArray(meta)) turn.meta = meta;
      current = { turn, body: [] };
    } else if (current !== undefined) {
      current.body.push(line);
    }
  }
  close();

  const thread: Thread = { id, title, source: source as ThreadSource, turns };
  for (const key of ["sourceId", "lane", "started", "updated", "parent"] as const) {
    const value = text(key);
    if (value !== undefined) thread[key] = value;
  }
  return thread;
}

/** ASCII filename slug for a thread title; the real title lives in frontmatter. */
export function slugify(title: string, maxLength = 60): string {
  const slug = title
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, maxLength)
    .replace(/-+$/g, "");
  return slug === "" ? "untitled" : slug;
}

/** threads/2026/09/2026-09-02-kitchen-quotes-5T1WB.md; the id tail keeps same-day, same-title threads apart. */
export function threadPath(thread: Pick<Thread, "id" | "title" | "started">): string {
  const day = /^\d{4}-\d{2}-\d{2}/.exec(thread.started ?? "")?.[0] ?? "undated";
  const [year, month] = day === "undated" ? ["undated", "00"] : [day.slice(0, 4), day.slice(5, 7)];
  return `threads/${year}/${month}/${day}-${slugify(thread.title)}-${thread.id.slice(-5).toLowerCase()}.md`;
}
