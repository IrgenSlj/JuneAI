// The Claude export (claude.ai: Settings, Privacy, Export data) to threads.
//
// The export is untyped JSON that changes over time, so every field is read
// defensively and anything unexpected is counted, never thrown. What we know
// of the shape comes from lab 001: a conversation has uuid, name, created_at,
// updated_at, chat_messages and sometimes model, project_uuid and
// current_leaf_message_uuid; a message has uuid, sender ("human" or
// "assistant"), created_at, text, content blocks (text, thinking, tool_use,
// tool_result), attachments with extracted_content, files, and sometimes
// parent_message_uuid.
//
// When messages carry parent links, the thread follows the conversation's
// current path (leaf back to root); edited and regenerated alternatives stay
// in the raw JSON kept beside the thread, so nothing is lost.
import { stableId } from "../stable-ids.js";
import { type Thread, type Turn, slugify } from "../thread.js";

type Obj = { [key: string]: unknown };
const isObj = (value: unknown): value is Obj => typeof value === "object" && value !== null && !Array.isArray(value);
const str = (value: unknown): string | undefined => (typeof value === "string" && value !== "" ? value : undefined);
const list = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

export interface ImportedThread {
  thread: Thread;
  /** The provider's original conversation object, kept beside the thread as .raw.json. */
  raw: string;
}

export interface ClaudeImportReport {
  /** Conversation objects found in the export. Equals threads + empty + unreadable. */
  conversations: number;
  threads: number;
  /** Conversations with no messages; nothing to import. */
  empty: number;
  /** Objects without a uuid or created_at; counted so the totals still match. */
  unreadable: number;
  /** Messages on the imported path. */
  messages: number;
  /** Messages on edited or regenerated branches, kept in the raw JSON only. */
  offPath: number;
  /** Attachments whose extracted text was included in a turn. */
  attachments: number;
  /** Files named by the export whose bytes the export does not contain. */
  filesNotIncluded: number;
}

/** True for an array whose objects look like Claude conversations. */
export function looksLikeClaudeConversations(value: unknown): value is unknown[] {
  return Array.isArray(value) && value.length > 0 && value.slice(0, 20).some((c) => isObj(c) && typeof c.uuid === "string" && Array.isArray(c.chat_messages));
}

/** True for an array whose objects look like Claude projects. */
export function looksLikeClaudeProjects(value: unknown): value is unknown[] {
  return Array.isArray(value) && value.length > 0 && value.slice(0, 5).every((p) => isObj(p) && typeof p.uuid === "string" && typeof p.name === "string" && !("chat_messages" in p));
}

/** A fence longer than any backtick run in the text, so attached code cannot close it. */
function fenced(text: string, lang = ""): string {
  const longest = Math.max(2, ...[...text.matchAll(/`+/g)].map((m) => m[0].length));
  const fence = "`".repeat(longest + 1);
  return `${fence}${lang}\n${text}\n${fence}`;
}

/** The readable text of one message: its text blocks, artifacts as code, tools as one line, attachments inlined. */
function messageText(message: Obj, counts: { attachments: number; filesNotIncluded: number }): string {
  const parts: string[] = [];
  for (const block of list(message.content)) {
    if (!isObj(block)) continue;
    const type = block.type;
    if (type === "text") {
      const text = str(block.text);
      if (text !== undefined) parts.push(text);
    } else if (type === "tool_use") {
      const input = isObj(block.input) ? block.input : {};
      const name = str(block.name) ?? "a tool";
      const body = str(input.content) ?? str(input.file_text) ?? str(input.code);
      if (body !== undefined && (name === "artifacts" || name === "create_file")) {
        const title = str(input.title) ?? str(input.path) ?? "Artifact";
        parts.push(`**${title}**\n\n${fenced(body, str(input.language) ?? "")}`);
      } else {
        parts.push(`*Used ${name}.*`);
      }
    }
    // thinking and tool_result blocks stay in the raw JSON only.
  }
  if (parts.length === 0) {
    const text = str(message.text);
    if (text !== undefined) parts.push(text);
  }
  for (const attachment of list(message.attachments)) {
    if (!isObj(attachment)) continue;
    const content = str(attachment.extracted_content);
    const name = str(attachment.file_name) ?? "attachment";
    if (content === undefined) {
      counts.filesNotIncluded++;
      parts.push(`*Attached ${name} (not included in the export).*`);
    } else {
      counts.attachments++;
      parts.push(`**Attached: ${name}**\n\n${fenced(content)}`);
    }
  }
  const inlined = new Set(list(message.attachments).flatMap((a) => (isObj(a) && str(a.file_name) !== undefined ? [str(a.file_name)] : [])));
  for (const file of list(message.files)) {
    const name = isObj(file) ? str(file.file_name) : undefined;
    if (name === undefined || inlined.has(name)) continue;
    counts.filesNotIncluded++;
    parts.push(`*Attached ${name} (not included in the export).*`);
  }
  return parts.join("\n\n");
}

/** The messages to import, in order: the current path when parent links exist, else the array order. */
function pathOf(conversation: Obj, messages: Obj[]): Obj[] {
  const byId = new Map(messages.flatMap((m) => (typeof m.uuid === "string" ? [[m.uuid, m] as const] : [])));
  const hasLinks = messages.some((m) => typeof m.parent_message_uuid === "string" && byId.has(m.parent_message_uuid));
  if (!hasLinks) return messages;
  // The leaf the user last saw, or failing that the newest message.
  let leaf = byId.get(str(conversation.current_leaf_message_uuid) ?? "");
  leaf ??= [...messages].sort((a, b) => String(a.created_at ?? "").localeCompare(String(b.created_at ?? ""))).at(-1);
  const path: Obj[] = [];
  const seen = new Set<unknown>();
  for (let node = leaf; node !== undefined && !seen.has(node.uuid); node = byId.get(str(node.parent_message_uuid) ?? "")) {
    seen.add(node.uuid);
    path.push(node);
  }
  return path.reverse();
}

function titleFor(conversation: Obj, turns: Turn[]): string {
  const name = str(conversation.name)?.trim();
  if (name !== undefined && name !== "") return name.slice(0, 200);
  const first = turns.find((t) => t.role === "user")?.text.split("\n").find((line) => line.trim() !== "")?.trim() ?? "";
  return first === "" ? "Untitled" : first.length > 80 ? `${first.slice(0, 79)}…` : first;
}

export interface ClaudeImportInput {
  conversations: unknown[];
  /** projects.json, when present: project names become lanes. */
  projects?: unknown[];
}

export async function parseClaudeExport(input: ClaudeImportInput): Promise<{ threads: ImportedThread[]; report: ClaudeImportReport }> {
  const report: ClaudeImportReport = { conversations: 0, threads: 0, empty: 0, unreadable: 0, messages: 0, offPath: 0, attachments: 0, filesNotIncluded: 0 };
  const lanes = new Map<string, string>();
  for (const project of input.projects ?? []) {
    if (isObj(project) && typeof project.uuid === "string" && str(project.name) !== undefined) lanes.set(project.uuid, slugify(String(project.name)));
  }
  const threads: ImportedThread[] = [];
  for (const conversation of input.conversations) {
    report.conversations++;
    const uuid = isObj(conversation) ? str(conversation.uuid) : undefined;
    const created = isObj(conversation) ? str(conversation.created_at) : undefined;
    if (!isObj(conversation) || uuid === undefined || created === undefined || !Number.isFinite(Date.parse(created))) {
      report.unreadable++;
      continue;
    }
    const messages = list(conversation.chat_messages).filter(isObj);
    const path = pathOf(conversation, messages);
    const counts = { attachments: 0, filesNotIncluded: 0 };
    const model = str(conversation.model);
    const turns: Turn[] = [];
    for (const message of path) {
      const role = message.sender === "human" ? "user" : message.sender === "assistant" ? "assistant" : undefined;
      if (role === undefined) continue;
      const text = messageText(message, counts);
      const turn: Turn = { id: `m${turns.length + 1}`, role, text };
      const at = str(message.created_at);
      if (at !== undefined) turn.at = at;
      if (role === "assistant" && model !== undefined) turn.model = model;
      const sourceId = str(message.uuid);
      if (sourceId !== undefined) turn.meta = { sourceId };
      turns.push(turn);
    }
    if (turns.length === 0) {
      report.empty++;
      continue;
    }
    report.threads++;
    report.messages += turns.length;
    report.offPath += messages.length - path.length;
    report.attachments += counts.attachments;
    report.filesNotIncluded += counts.filesNotIncluded;

    const thread: Thread = {
      id: await stableId("thread", Date.parse(created), "claude-export", uuid),
      title: titleFor(conversation, turns),
      source: "claude-export",
      sourceId: uuid,
      started: created,
      turns,
    };
    const updated = str(conversation.updated_at);
    if (updated !== undefined) thread.updated = updated;
    const lane = lanes.get(str(conversation.project_uuid) ?? "");
    if (lane !== undefined) thread.lane = lane;
    threads.push({ thread, raw: `${JSON.stringify(conversation, null, 2)}\n` });
  }
  return { threads, report };
}
