// The ChatGPT export (Settings, Data controls, Export data) to threads.
//
// conversations.json is an array of conversations. Each holds a tree of
// nodes in `mapping` ({ id: { message, parent, children } }) and
// `current_node`, the leaf the user last saw. A message has author.role
// (user, assistant, system, tool), create_time in Unix seconds, content with
// content_type (text, multimodal_text, code, execution_output, thoughts, ...)
// and parts, metadata.model_slug, and recipient ("all", or a tool's name when
// the model is calling one).
//
// As with Claude: the thread follows the current path; regenerated and
// edited branches, hidden system messages, tool output and reasoning stay in
// the raw JSON kept beside the thread.
import { stableId } from "../stable-ids.js";
import { type Thread, type Turn } from "../thread.js";
import type { ClaudeImportReport, ImportedThread } from "./claude.js";

type Obj = { [key: string]: unknown };
const isObj = (value: unknown): value is Obj => typeof value === "object" && value !== null && !Array.isArray(value);
const str = (value: unknown): string | undefined => (typeof value === "string" && value !== "" ? value : undefined);
const iso = (seconds: unknown): string | undefined => (typeof seconds === "number" && Number.isFinite(seconds) && seconds > 0 ? new Date(seconds * 1000).toISOString() : undefined);

/** The same counts as a Claude import, so the report reads the same. */
export type ImportCounts = ClaudeImportReport;

/** True for an array whose objects look like ChatGPT conversations. */
export function looksLikeChatGptConversations(value: unknown): value is unknown[] {
  return Array.isArray(value) && value.length > 0 && value.slice(0, 20).some((c) => isObj(c) && isObj(c.mapping));
}

function fenced(text: string, lang = ""): string {
  const longest = Math.max(2, ...[...text.matchAll(/`+/g)].map((m) => m[0].length));
  const fence = "`".repeat(longest + 1);
  return `${fence}${lang}\n${text}\n${fence}`;
}

/** The node ids from root to the current leaf. Without current_node, the newest child at each step. */
function pathOf(conversation: Obj, mapping: Obj): string[] {
  let leaf = str(conversation.current_node);
  if (leaf === undefined || !isObj(mapping[leaf])) {
    const root = Object.keys(mapping).find((id) => {
      const node = mapping[id];
      return isObj(node) && (node.parent === null || node.parent === undefined || !isObj(mapping[String(node.parent)]));
    });
    leaf = root;
    for (let guard = 0; leaf !== undefined && guard < 100_000; guard++) {
      const node = mapping[leaf];
      const children = isObj(node) && Array.isArray(node.children) ? node.children.filter((c): c is string => typeof c === "string" && isObj(mapping[c])) : [];
      if (children.length === 0) break;
      leaf = children.at(-1);
    }
  }
  const path: string[] = [];
  const seen = new Set<string>();
  for (let id = leaf; id !== undefined && !seen.has(id) && isObj(mapping[id]); id = str((mapping[id] as Obj).parent)) {
    seen.add(id);
    path.push(id);
  }
  return path.reverse();
}

/** Readable text for one message, or undefined when it is not part of the conversation as the user saw it. */
function messageText(message: Obj, counts: { filesNotIncluded: number }): string | undefined {
  const metadata = isObj(message.metadata) ? message.metadata : {};
  if (metadata.is_visually_hidden_from_conversation === true) return undefined;
  const content = isObj(message.content) ? message.content : {};
  const type = str(content.content_type) ?? "text";
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const recipient = str(message.recipient) ?? "all";
  if (recipient !== "all") return `*Used ${recipient.split(".")[0]}.*`;
  if (type === "code") {
    const code = str(content.text);
    return code === undefined ? undefined : fenced(code, str(content.language) === "unknown" ? "" : (str(content.language) ?? ""));
  }
  if (type !== "text" && type !== "multimodal_text") return undefined; // thoughts, reasoning, tool output, browsing quotes
  const out: string[] = [];
  for (const part of parts) {
    if (typeof part === "string") {
      if (part.trim() !== "") out.push(part);
    } else if (isObj(part)) {
      counts.filesNotIncluded++;
      out.push(`*Attached ${part.content_type === "image_asset_pointer" ? "an image" : "a file"} (not included in the conversation text).*`);
    }
  }
  return out.length === 0 ? undefined : out.join("\n\n");
}

function titleFor(conversation: Obj, turns: Turn[]): string {
  const title = str(conversation.title)?.trim();
  if (title !== undefined && title !== "" && title !== "New chat") return title.slice(0, 200);
  const first = turns.find((t) => t.role === "user")?.text.split("\n").find((line) => line.trim() !== "")?.trim() ?? "";
  return first === "" ? "Untitled" : first.length > 80 ? `${first.slice(0, 79)}…` : first;
}

export async function parseChatGptExport(conversations: unknown[]): Promise<{ threads: ImportedThread[]; report: ImportCounts }> {
  const report: ImportCounts = { conversations: 0, threads: 0, empty: 0, unreadable: 0, messages: 0, offPath: 0, attachments: 0, filesNotIncluded: 0 };
  const threads: ImportedThread[] = [];
  for (const conversation of conversations) {
    report.conversations++;
    const id = isObj(conversation) ? (str(conversation.conversation_id) ?? str(conversation.id)) : undefined;
    const created = isObj(conversation) ? iso(conversation.create_time) : undefined;
    if (!isObj(conversation) || !isObj(conversation.mapping) || id === undefined || created === undefined) {
      report.unreadable++;
      continue;
    }
    const mapping = conversation.mapping;
    const path = pathOf(conversation, mapping);
    const counts = { filesNotIncluded: 0 };
    const fallbackModel = str(conversation.default_model_slug);
    const turns: Turn[] = [];
    for (const nodeId of path) {
      const node = mapping[nodeId] as Obj;
      const message = isObj(node.message) ? node.message : undefined;
      const role = message !== undefined && isObj(message.author) ? message.author.role : undefined;
      if (message === undefined || (role !== "user" && role !== "assistant")) continue;
      const text = messageText(message, counts);
      if (text === undefined) continue;
      const previous = turns.at(-1);
      // Consecutive assistant pieces (a tool call, then the answer) read as one reply.
      if (role === "assistant" && previous?.role === "assistant") {
        previous.text = `${previous.text}\n\n${text}`;
        continue;
      }
      const turn: Turn = { id: `m${turns.length + 1}`, role, text };
      const at = iso(message.create_time);
      if (at !== undefined) turn.at = at;
      const metadata = isObj(message.metadata) ? message.metadata : {};
      const model = str(metadata.model_slug) ?? fallbackModel;
      if (role === "assistant" && model !== undefined) turn.model = model;
      const sourceId = str(message.id);
      if (sourceId !== undefined) turn.meta = { sourceId };
      turns.push(turn);
    }
    if (turns.length === 0) {
      report.empty++;
      continue;
    }
    const withMessages = Object.values(mapping).filter((n) => isObj(n) && isObj(n.message)).length;
    const onPath = path.filter((nodeId) => isObj((mapping[nodeId] as Obj).message)).length;
    report.threads++;
    report.messages += turns.length;
    report.offPath += Math.max(0, withMessages - onPath);
    report.filesNotIncluded += counts.filesNotIncluded;

    const thread: Thread = {
      id: await stableId("thread", Date.parse(created), "chatgpt-export", id),
      title: titleFor(conversation, turns),
      source: "chatgpt-export",
      sourceId: id,
      started: created,
      turns,
    };
    const updated = iso(conversation.update_time);
    if (updated !== undefined) thread.updated = updated;
    threads.push({ thread, raw: `${JSON.stringify(conversation, null, 2)}\n` });
  }
  return { threads, report };
}
