// The engine is platform-agnostic like core and providers, and compiles their
// source, so it declares the same runtime globals they rely on.
declare const crypto: {
  getRandomValues<T extends Uint8Array>(array: T): T;
  subtle: {
    digest(algorithm: "SHA-256", data: Uint8Array): Promise<ArrayBuffer>;
  };
};
declare class TextDecoder {
  constructor(label?: string);
  decode(input?: Uint8Array, options?: { stream?: boolean }): string;
}
declare class TextEncoder {
  encode(input: string): Uint8Array;
}
