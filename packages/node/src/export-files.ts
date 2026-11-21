// Reads what a provider export arrives as (a zip, a folder of zips, a
// conversations.json, or JSON Lines) into parsed JSON documents, leaving the
// interpretation to core. Claude's 2026 export is a manifest plus several
// zips (conversations-000.zip, projects-000.zip, ...); older exports are one
// zip with conversations.json inside. Both arrive here as "JSON documents".
//
// Only .json and .jsonl entries are decompressed, and an entry claiming to be
// larger than the limit is refused before it is inflated (a zip bomb guard).
import { readFile, readdir, stat } from "node:fs/promises";
import { basename, extname, join } from "node:path";
import { unzipSync } from "fflate";

export interface ExportDocument {
  /** Where it came from, for the report: "conversations-000.zip/conversations.json". */
  name: string;
  data: unknown;
}

export class ExportReadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ExportReadError";
  }
}

const MAX_ENTRY_BYTES = 2_000_000_000;
const WANTED = /\.(json|jsonl|zip)$/i;

function parseDocument(name: string, bytes: Uint8Array): ExportDocument {
  const text = new TextDecoder("utf-8").decode(bytes).replace(/^﻿/, "");
  if (/\.jsonl$/i.test(name)) {
    const data = text
      .split("\n")
      .filter((line) => line.trim() !== "")
      .map((line, i) => {
        try {
          return JSON.parse(line) as unknown;
        } catch {
          throw new ExportReadError(`${name}: line ${i + 1} is not JSON`);
        }
      });
    return { name, data };
  }
  try {
    return { name, data: JSON.parse(text) as unknown };
  } catch {
    throw new ExportReadError(`${name} is not valid JSON`);
  }
}

/** The JSON documents inside one file's bytes, opening zips (and zips inside zips). */
export function documentsIn(name: string, bytes: Uint8Array, depth = 0): ExportDocument[] {
  const isZip = bytes[0] === 0x50 && bytes[1] === 0x4b;
  if (!isZip) return /\.(json|jsonl)$/i.test(name) || extname(name) === "" ? [parseDocument(name, bytes)] : [];
  if (depth > 2) return [];
  let entries: Record<string, Uint8Array>;
  try {
    entries = unzipSync(bytes, {
      filter: (file) => {
        if (!WANTED.test(file.name) || file.name.startsWith("__MACOSX/") || basename(file.name).startsWith("._")) return false;
        if (file.originalSize > MAX_ENTRY_BYTES) throw new ExportReadError(`${name}/${file.name} is too large to import`);
        return true;
      },
    });
  } catch (error) {
    if (error instanceof ExportReadError) throw error;
    throw new ExportReadError(`${name} is not a readable zip`);
  }
  return Object.keys(entries)
    .sort()
    .flatMap((entry) => documentsIn(`${name}/${entry}`, entries[entry] ?? new Uint8Array(), depth + 1));
}

/** Every JSON document in a file or folder on disk (folders: two levels deep, sorted). */
export async function readExportPath(path: string): Promise<ExportDocument[]> {
  const info = await stat(path).catch(() => undefined);
  if (info === undefined) throw new ExportReadError(`nothing found at ${path}`);
  if (info.isFile()) return documentsIn(basename(path), new Uint8Array(await readFile(path)));
  const out: ExportDocument[] = [];
  const walk = async (dir: string, prefix: string, depth: number): Promise<void> => {
    for (const entry of (await readdir(dir, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
      if (entry.name.startsWith(".")) continue;
      const full = join(dir, entry.name);
      if (entry.isDirectory() && depth < 2) await walk(full, `${prefix}${entry.name}/`, depth + 1);
      else if (entry.isFile() && WANTED.test(entry.name)) out.push(...documentsIn(`${prefix}${entry.name}`, new Uint8Array(await readFile(full))));
    }
  };
  await walk(path, "", 0);
  return out;
}
