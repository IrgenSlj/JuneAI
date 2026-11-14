// Providers are platform-agnostic like core: no Node or DOM types. These are
// the only runtime globals they use; Node 22+, browsers and web views have them.
declare class TextDecoder {
  constructor(label?: string);
  decode(input?: Uint8Array, options?: { stream?: boolean }): string;
}
declare class TextEncoder {
  encode(input: string): Uint8Array;
}
