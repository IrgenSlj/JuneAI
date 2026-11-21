// Lab 001: the shape of a real Claude export, without its content.
// Prints file lists, counts, date ranges, which fields appear how often, and
// the kinds of content blocks. Never prints message text, names or emails.
//
//   node lab/scripts/claude-shape.mjs lab/private/claude
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { unzipSync } from "../../packages/node/node_modules/fflate/esm/index.mjs";

const root = process.argv[2] ?? "lab/private/claude";
const docs = [];

function take(name, bytes) {
  if (bytes[0] === 0x50 && bytes[1] === 0x4b) {
    const entries = unzipSync(bytes, { filter: (f) => /\.(json|jsonl|zip)$/i.test(f.name) && !f.name.startsWith("__MACOSX/") });
    console.log(`${name}: zip`);
    for (const [entry, data] of Object.entries(entries)) console.log(`  ${entry}  ${(data.length / 1e6).toFixed(2)} MB`);
    for (const [entry, data] of Object.entries(entries)) take(`${name}/${entry}`, data);
    return;
  }
  const text = new TextDecoder().decode(bytes);
  try {
    docs.push({ name, data: /\.jsonl$/i.test(name) ? text.split("\n").filter(Boolean).map((l) => JSON.parse(l)) : JSON.parse(text) });
  } catch {
    console.log(`${name}: not JSON (${bytes.length} bytes)`);
  }
}

for (const file of readdirSync(root).sort()) {
  const path = join(root, file);
  if (statSync(path).isFile() && /\.(json|jsonl|zip)$/i.test(file)) take(file, new Uint8Array(readFileSync(path)));
}

/** Keys and value types, with how often each key appears, across up to 2000 objects. */
function shape(objects) {
  const seen = new Map();
  for (const o of objects.slice(0, 2000)) {
    if (o === null || typeof o !== "object" || Array.isArray(o)) continue;
    for (const [k, v] of Object.entries(o)) {
      const type = Array.isArray(v) ? "array" : v === null ? "null" : typeof v;
      const entry = seen.get(k) ?? { count: 0, types: new Set() };
      entry.count++;
      entry.types.add(type);
      seen.set(k, entry);
    }
  }
  const n = Math.min(objects.length, 2000);
  return [...seen].map(([k, e]) => `    ${k}: ${[...e.types].join("|")}  (${Math.round((100 * e.count) / n)}%)`).join("\n");
}

const count = (items, key) => {
  const out = {};
  for (const item of items) out[item[key]] = (out[item[key]] ?? 0) + 1;
  return out;
};

for (const { name, data } of docs) {
  console.log(`\n== ${name}: ${Array.isArray(data) ? `array of ${data.length}` : `object with keys ${Object.keys(data ?? {}).join(", ")}`}`);
  if (!Array.isArray(data) || data.length === 0) continue;
  console.log(`  item fields:\n${shape(data)}`);
  if (!data.some((c) => Array.isArray(c?.chat_messages))) continue;
  const messages = data.flatMap((c) => c.chat_messages ?? []);
  const dates = data.map((c) => c.created_at).filter(Boolean).sort();
  console.log(`  conversations: ${data.length}, empty: ${data.filter((c) => (c.chat_messages ?? []).length === 0).length}, messages: ${messages.length}`);
  console.log(`  created from ${dates[0]} to ${dates.at(-1)}`);
  console.log(`  with current_leaf_message_uuid: ${data.filter((c) => c.current_leaf_message_uuid).length}, with model: ${data.filter((c) => c.model).length}, with project_uuid: ${data.filter((c) => c.project_uuid).length}`);
  console.log(`  message fields:\n${shape(messages)}`);
  console.log(`  senders: ${JSON.stringify(count(messages, "sender"))}`);
  const blocks = messages.flatMap((m) => m.content ?? []);
  console.log(`  content block types: ${JSON.stringify(count(blocks, "type"))}`);
  console.log(`  tool_use names: ${JSON.stringify(count(blocks.filter((b) => b.type === "tool_use"), "name"))}`);
  console.log(`  content block fields:\n${shape(blocks)}`);
  const attachments = messages.flatMap((m) => m.attachments ?? []);
  console.log(`  attachments: ${attachments.length} (with text: ${attachments.filter((a) => a.extracted_content).length}), files: ${messages.flatMap((m) => m.files ?? []).length}`);
  const branched = data.filter((c) => new Set((c.chat_messages ?? []).map((m) => m.parent_message_uuid)).size < (c.chat_messages ?? []).length).length;
  console.log(`  conversations where a parent has two children (edited or regenerated): ${branched}`);
}
