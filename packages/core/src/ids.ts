// Vault ids: a type prefix plus a ULID, e.g. "th_01JD3K9V2Q8XN4M6P0R7S5T1WB".
// ULIDs sort by creation time, so ids in a folder listing are in time order.
// Spec: https://github.com/ulid/spec (48-bit ms timestamp + 80 random bits,
// Crockford base32, monotonic within the same millisecond).

export const ID_PREFIXES = {
  thread: "th",
  note: "nt",
  plan: "pl",
} as const;

export type IdKind = keyof typeof ID_PREFIXES;
export type IdPrefix = (typeof ID_PREFIXES)[IdKind];

const ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
const TIME_LEN = 10;
const RANDOM_LEN = 16;
const MAX_TIME = 2 ** 48 - 1;
const MAX_RANDOM = (1n << 80n) - 1n;
const ID_PATTERN = /^(th|nt|pl)_[0-9A-HJKMNP-TV-Z]{26}$/;

export interface IdSource {
  now(): number;
  random80(): bigint;
}

const systemSource: IdSource = {
  now: () => Date.now(),
  random80: () => {
    const bytes = crypto.getRandomValues(new Uint8Array(10));
    let value = 0n;
    for (const byte of bytes) value = (value << 8n) | BigInt(byte);
    return value;
  },
};

function encode(value: bigint, length: number): string {
  let out = "";
  for (let i = 0; i < length; i++) {
    out = ALPHABET[Number(value & 31n)] + out;
    value >>= 5n;
  }
  return out;
}

/** Make an id generator. Pass a source in tests to control time and randomness. */
export function createIdGenerator(source: IdSource = systemSource) {
  let lastTime = -1;
  let lastRandom = 0n;

  return function newId(kind: IdKind): string {
    const time = source.now();
    if (!Number.isInteger(time) || time < 0 || time > MAX_TIME) {
      throw new RangeError(`time out of ULID range: ${time}`);
    }
    if (time <= lastTime) {
      // Same millisecond (or a clock step back): keep order by incrementing.
      if (lastRandom === MAX_RANDOM) throw new RangeError("ULID random overflow");
      lastRandom += 1n;
    } else {
      lastTime = time;
      lastRandom = source.random80() & MAX_RANDOM;
    }
    return `${ID_PREFIXES[kind]}_${encode(BigInt(lastTime), TIME_LEN)}${encode(lastRandom, RANDOM_LEN)}`;
  };
}

export const newId = createIdGenerator();

export function isId(value: string, kind?: IdKind): boolean {
  if (!ID_PATTERN.test(value)) return false;
  return kind === undefined || value.startsWith(`${ID_PREFIXES[kind]}_`);
}

/** The creation time embedded in an id, in ms since the epoch. */
export function idTime(id: string): number {
  if (!isId(id)) throw new TypeError(`not a vault id: ${id}`);
  let value = 0;
  for (const char of id.slice(3, 3 + TIME_LEN)) value = value * 32 + ALPHABET.indexOf(char);
  return value;
}
