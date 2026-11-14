// Deterministic ids for imported material. An imported thread must get the
// same id on every machine and every re-import, or links from notes break.
// The id is still a valid prefixed ULID: its time part is the provider's own
// creation time, its "random" part is a hash of where the item came from.
import { ID_PREFIXES, type IdKind } from "./ids.js";
import { sha256Bytes } from "./hash.js";

const ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
const MAX_TIME = 2 ** 48 - 1;

function encode(value: bigint, length: number): string {
  let out = "";
  for (let i = 0; i < length; i++) {
    out = ALPHABET[Number(value & 31n)] + out;
    value >>= 5n;
  }
  return out;
}

/**
 * @param createdMs the provider's creation time in ms; clamped into ULID range.
 * @param source    e.g. "claude-export"
 * @param sourceId  the provider's own id for the item
 */
export async function stableId(kind: IdKind, createdMs: number, source: string, sourceId: string): Promise<string> {
  const time = Math.min(Math.max(Math.trunc(Number.isFinite(createdMs) ? createdMs : 0), 0), MAX_TIME);
  const digest = await sha256Bytes(`${source}\u0000${sourceId}`);
  let random = 0n;
  for (const byte of digest.subarray(0, 10)) random = (random << 8n) | BigInt(byte);
  return `${ID_PREFIXES[kind]}_${encode(BigInt(time), 10)}${encode(random, 16)}`;
}
