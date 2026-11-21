// Pure helpers for the thread list: date groups like a mail app, and search
// snippets split into plain and matched parts (rendered as text and <mark>,
// never as HTML).

export interface Dated {
  activity?: string;
  updated?: string;
  started?: string;
}

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

/** The group a time falls in, seen from `now` in local time: Today, Yesterday, Previous 7 days, Previous 30 days, then "March 2025". */
export function dayGroup(iso: string | undefined, now: Date): string {
  const at = iso === undefined ? Number.NaN : Date.parse(iso);
  if (!Number.isFinite(at)) return "Undated";
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const day = 86_400_000;
  if (at >= startOfToday) return "Today";
  if (at >= startOfToday - day) return "Yesterday";
  if (at >= startOfToday - 7 * day) return "Previous 7 days";
  if (at >= startOfToday - 30 * day) return "Previous 30 days";
  const date = new Date(at);
  return `${MONTHS[date.getMonth()]} ${date.getFullYear()}`;
}

/** Consecutive groups, in the list's own order (the daemon sorts newest first). */
export function groupByDay<T extends Dated>(items: T[], now: Date): { name: string; items: T[] }[] {
  const groups: { name: string; items: T[] }[] = [];
  for (const item of items) {
    const name = dayGroup(item.activity ?? item.updated ?? item.started, now);
    const last = groups.at(-1);
    if (last?.name === name) last.items.push(item);
    else groups.push({ name, items: [item] });
  }
  return groups;
}

/** A snippet from the daemon split on its \u0002 ... \u0003 markers. */
export function snippetParts(snippet: string): { text: string; match: boolean }[] {
  const parts: { text: string; match: boolean }[] = [];
  for (const [i, piece] of snippet.split(/[\u0002\u0003]/).entries()) if (piece !== "") parts.push({ text: piece, match: i % 2 === 1 });
  return parts;
}

/** Display name of a thread's source. */
export function sourceName(source: string): string | undefined {
  return { "claude-export": "Claude", "chatgpt-export": "ChatGPT", "lechat-export": "Le Chat", "claude-code": "Claude Code" }[source];
}
