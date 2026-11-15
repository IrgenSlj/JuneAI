// Markdown to a plain tree, for model output and imported text. Nothing here
// produces HTML: Markdown.svelte renders the tree as ordinary elements, so a
// reply containing <script> or an onerror attribute is shown as text, by
// construction rather than by filtering (PLAN.md: "sanitised Markdown, no raw
// HTML"). Links survive only to absolute http, https and mailto addresses.
//
// A deliberate subset of CommonMark and GFM: what models actually write.

export type Inline =
  | { type: "text"; value: string }
  | { type: "break" }
  | { type: "code"; value: string }
  | { type: "strong"; children: Inline[] }
  | { type: "em"; children: Inline[] }
  | { type: "link"; href: string; children: Inline[] };

export type Block =
  | { type: "paragraph"; children: Inline[] }
  | { type: "heading"; level: number; children: Inline[] }
  | { type: "code"; lang: string; value: string }
  | { type: "quote"; children: Block[] }
  | { type: "list"; ordered: boolean; start: number; items: Block[][] }
  | { type: "table"; head: Inline[][]; rows: Inline[][][] }
  | { type: "rule" };

const FENCE = /^\s{0,3}(`{3,}|~{3,})\s*([\w+#.-]*)\s*$/;
const HEADING = /^\s{0,3}(#{1,6})\s+(.*?)(?:\s+#+)?\s*$/;
const RULE = /^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/;
const QUOTE = /^\s{0,3}>\s?(.*)$/;
const ITEM = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/;
const TABLE_DIVIDER = /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$/;

const indentOf = (line: string): number => (/^\s*/.exec(line)?.[0] ?? "").replace(/\t/g, "    ").length;

/** A line that starts a new block, so a paragraph ends before it. */
function startsBlock(line: string): boolean {
  return FENCE.test(line) || HEADING.test(line) || RULE.test(line) || QUOTE.test(line) || ITEM.test(line);
}

function cells(row: string): string[] {
  return row
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

export function parseMarkdown(source: string): Block[] {
  return parseLines(source.replace(/\r\n?/g, "\n").split("\n"));
}

function parseLines(lines: string[]): Block[] {
  const blocks: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i] ?? "";
    if (line.trim() === "") {
      i++;
      continue;
    }

    const fence = FENCE.exec(line);
    if (fence !== null) {
      const marker = fence[1] ?? "```";
      const body: string[] = [];
      i++;
      while (i < lines.length && !(lines[i] ?? "").trim().startsWith(marker)) body.push(lines[i++] ?? "");
      i++; // the closing fence, or past the end when it was never closed
      blocks.push({ type: "code", lang: fence[2] ?? "", value: body.join("\n") });
      continue;
    }

    const heading = HEADING.exec(line);
    if (heading !== null) {
      blocks.push({ type: "heading", level: (heading[1] ?? "#").length, children: parseInline(heading[2] ?? "") });
      i++;
      continue;
    }

    if (RULE.test(line)) {
      blocks.push({ type: "rule" });
      i++;
      continue;
    }

    if (QUOTE.test(line)) {
      const body: string[] = [];
      while (i < lines.length && QUOTE.test(lines[i] ?? "")) body.push(QUOTE.exec(lines[i++] ?? "")?.[1] ?? "");
      blocks.push({ type: "quote", children: parseLines(body) });
      continue;
    }

    if (line.includes("|") && TABLE_DIVIDER.test(lines[i + 1] ?? "")) {
      const head = cells(line).map(parseInline);
      const rows: Inline[][][] = [];
      i += 2;
      while (i < lines.length && (lines[i] ?? "").includes("|") && (lines[i] ?? "").trim() !== "") rows.push(cells(lines[i++] ?? "").map(parseInline));
      blocks.push({ type: "table", head, rows });
      continue;
    }

    const item = ITEM.exec(line);
    if (item !== null) {
      i = parseList(lines, i, blocks);
      continue;
    }

    const body: string[] = [];
    while (i < lines.length && (lines[i] ?? "").trim() !== "" && (body.length === 0 || !startsBlock(lines[i] ?? ""))) body.push((lines[i++] ?? "").trim());
    blocks.push({ type: "paragraph", children: joinLines(body) });
  }
  return blocks;
}

/** Parses one list starting at `start`; returns the index after it. */
function parseList(lines: string[], start: number, out: Block[]): number {
  const first = ITEM.exec(lines[start] ?? "");
  if (first === null) return start + 1;
  const base = indentOf(lines[start] ?? "");
  const ordered = /\d/.test(first[2] ?? "");
  const items: Block[][] = [];
  let i = start;
  while (i < lines.length) {
    const match = ITEM.exec(lines[i] ?? "");
    if (match === null || indentOf(lines[i] ?? "") !== base || /\d/.test(match[2] ?? "") !== ordered) break;
    const body = [match[3] ?? ""];
    i++;
    // Continuation: deeper-indented lines, including nested lists, and blank lines followed by one.
    while (i < lines.length) {
      const next = lines[i] ?? "";
      if (next.trim() === "") {
        if (indentOf(lines[i + 1] ?? "") > base && (lines[i + 1] ?? "").trim() !== "") {
          body.push("");
          i++;
          continue;
        }
        break;
      }
      if (indentOf(next) <= base) break;
      body.push(next.slice(Math.min(indentOf(next), base + 2)));
      i++;
    }
    items.push(parseLines(body));
  }
  out.push({ type: "list", ordered, start: ordered ? Number.parseInt(first[2] ?? "1", 10) : 1, items });
  return i;
}

function joinLines(lines: string[]): Inline[] {
  const out: Inline[] = [];
  lines.forEach((line, index) => {
    if (index > 0) out.push({ type: "break" });
    out.push(...parseInline(line));
  });
  return out;
}

/** An href June will make clickable, or undefined. Only absolute http, https and mailto. */
export function safeHref(raw: string): string | undefined {
  const href = raw.trim();
  if (href === "" || /[\s\u0000-\u001f]/.test(href)) return undefined;
  try {
    const url = new URL(href);
    return url.protocol === "http:" || url.protocol === "https:" || url.protocol === "mailto:" ? href : undefined;
  } catch {
    return undefined;
  }
}

const isWord = (ch: string | undefined): boolean => ch !== undefined && /[\p{L}\p{N}]/u.test(ch);

export function parseInline(source: string): Inline[] {
  const out: Inline[] = [];
  let buffer = "";
  const flush = () => {
    if (buffer !== "") out.push({ type: "text", value: buffer });
    buffer = "";
  };
  let i = 0;
  while (i < source.length) {
    const ch = source[i] ?? "";
    const rest = source.slice(i);

    if (ch === "`") {
      const run = /^`+/.exec(rest)?.[0] ?? "`";
      const close = source.indexOf(run, i + run.length);
      if (close !== -1) {
        flush();
        out.push({ type: "code", value: source.slice(i + run.length, close).replace(/^ (.*) $/, "$1") });
        i = close + run.length;
        continue;
      }
    }

    if ((ch === "*" || ch === "_") && source[i + 1] === ch) {
      const marker = ch + ch;
      const close = source.indexOf(marker, i + 2);
      const inner = close === -1 ? "" : source.slice(i + 2, close);
      if (close !== -1 && inner.trim() !== "" && !/^\s/.test(inner) && !/\s$/.test(inner)) {
        flush();
        out.push({ type: "strong", children: parseInline(inner) });
        i = close + 2;
        continue;
      }
    }

    if (ch === "*" || ch === "_") {
      const opensWord = ch === "_" && isWord(source[i - 1]);
      const next = source[i + 1];
      if (!opensWord && next !== undefined && !/\s/.test(next) && next !== ch) {
        let close = source.indexOf(ch, i + 1);
        while (close !== -1 && (/\s/.test(source[close - 1] ?? " ") || (ch === "_" && isWord(source[close + 1])) || source[close + 1] === ch)) {
          close = source.indexOf(ch, close + 1);
        }
        if (close !== -1) {
          flush();
          out.push({ type: "em", children: parseInline(source.slice(i + 1, close)) });
          i = close + 1;
          continue;
        }
      }
    }

    if (ch === "[") {
      const textEnd = source.indexOf("](", i + 1);
      if (textEnd !== -1 && !source.slice(i + 1, textEnd).includes("[")) {
        let depth = 1;
        let j = textEnd + 2;
        while (j < source.length && depth > 0) {
          if (source[j] === "(") depth++;
          else if (source[j] === ")") depth--;
          if (depth > 0) j++;
        }
        if (depth === 0) {
          const href = safeHref(source.slice(textEnd + 2, j));
          if (href !== undefined) {
            flush();
            out.push({ type: "link", href, children: parseInline(source.slice(i + 1, textEnd)) });
          } else {
            buffer += source.slice(i, j + 1);
          }
          i = j + 1;
          continue;
        }
      }
    }

    if (ch === "h" && /^https?:\/\//.test(rest) && !isWord(source[i - 1])) {
      const url = (/^https?:\/\/[^\s<>()[\]"']+/.exec(rest)?.[0] ?? "").replace(/[.,;:!?]+$/, "");
      const href = safeHref(url);
      if (href !== undefined && url.length > "https://".length) {
        flush();
        out.push({ type: "link", href, children: [{ type: "text", value: url }] });
        i += url.length;
        continue;
      }
    }

    buffer += ch;
    i++;
  }
  flush();
  return out;
}
