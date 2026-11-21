// One thread out of the vault (PLAN.md decision 8: "per-thread Markdown and
// JSON"). The vault file is already Markdown, but its turn markers are for
// June; this is the version to read, send or paste elsewhere.
import type { Thread } from "@june/core";

const day = (iso: string | undefined): string => (iso === undefined ? "" : iso.slice(0, 16).replace("T", " "));

export function threadAsMarkdown(thread: Thread): string {
  const lines = [`# ${thread.title}`, ""];
  const about = [thread.started === undefined ? undefined : `Started ${day(thread.started)}`, thread.source === "june" ? undefined : `imported from ${thread.source}`]
    .filter((part) => part !== undefined)
    .join(", ");
  if (about !== "") lines.push(`*${about}.*`, "");
  for (const turn of thread.turns) {
    const who = turn.role === "user" ? "You" : turn.role === "assistant" ? (turn.model ?? "Assistant") : turn.role === "system" ? "System" : "Tool";
    lines.push(`## ${who}${turn.at === undefined ? "" : ` · ${day(turn.at)}`}`, "", turn.text, "");
  }
  return `${lines.join("\n").trimEnd()}\n`;
}

export function threadAsJson(thread: Thread): string {
  return `${JSON.stringify(thread, null, 2)}\n`;
}

/** An ASCII filename for the download. */
export function exportFilename(thread: Thread, extension: "md" | "json"): string {
  const slug = thread.title
    .normalize("NFKD")
    .replace(/[^\x20-\x7e]/g, "")
    .replace(/[^A-Za-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60);
  return `${slug === "" ? "conversation" : slug}.${extension}`;
}
