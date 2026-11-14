// The core is platform-agnostic, so it compiles without Node or DOM types.
// These are the only runtime globals it relies on; every target (Node 22+,
// Electron, browsers, iOS/Android web views) provides them.
declare const crypto: {
  getRandomValues<T extends Uint8Array>(array: T): T;
};
