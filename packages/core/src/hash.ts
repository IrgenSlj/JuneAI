// Hashing and canonical JSON, shared by receipts and deterministic ids.
// Web Crypto only, so it runs wherever the core runs.

export type Json = string | number | boolean | null | Json[] | { [key: string]: Json };

const encoder = new TextEncoder();

/** SHA-256 of a string (UTF-8), as raw bytes. */
export async function sha256Bytes(text: string): Promise<Uint8Array> {
  return new Uint8Array(await crypto.subtle.digest("SHA-256", encoder.encode(text)));
}

/** SHA-256 of a string (UTF-8), as lowercase hex. */
export async function sha256Hex(text: string): Promise<string> {
  let out = "";
  for (const byte of await sha256Bytes(text)) out += byte.toString(16).padStart(2, "0");
  return out;
}

/**
 * JSON with object keys sorted at every level, so the same value always
 * produces the same bytes. Hashes are only meaningful over canonical text.
 */
export function canonicalJson(value: Json): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const keys = Object.keys(value).sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalJson(value[k] as Json)}`).join(",")}}`;
}
