// Web backend — browser-only capabilities. Native-only methods throw
// UnsupportedError; calling code is expected to feature-detect and fall back.

import {
  type HotkeyId,
  type NotifyOptions,
  type OllamaProgress,
  type PickedFile,
  type PickFileOptions,
  type Platform,
  type TrayMenuItem,
  UnsupportedError,
} from "./types.js";

const RUNTIME = "web" as const;

const unsupported = (capability: string): never => {
  throw new UnsupportedError(capability, RUNTIME);
};

async function notify(options: NotifyOptions): Promise<void> {
  if (typeof Notification === "undefined") return; // SSR / non-secure context
  let permission = Notification.permission;
  if (permission === "default") permission = await Notification.requestPermission();
  if (permission !== "granted") return;
  new Notification(options.title, { body: options.body, tag: options.tag });
}

async function openExternal(url: string): Promise<void> {
  if (typeof window === "undefined") return;
  window.open(url, "_blank", "noopener,noreferrer");
}

async function pickFile(options?: PickFileOptions): Promise<PickedFile[] | null> {
  if (typeof document === "undefined") return null;
  return await new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = options?.multiple ?? false;
    if (options?.accept?.length) input.accept = options.accept.join(",");
    input.onchange = async () => {
      if (!input.files || input.files.length === 0) {
        resolve(null);
        return;
      }
      const files = await Promise.all(
        Array.from(input.files).map(async (f) => ({
          name: f.name,
          path: null,
          bytes: new Uint8Array(await f.arrayBuffer()),
        })),
      );
      resolve(files);
    };
    // If the user cancels, the change event never fires. We resolve null after
    // a focus-back tick so the caller can move on.
    const onFocus = () => {
      window.removeEventListener("focus", onFocus);
      setTimeout(() => {
        if (!input.files || input.files.length === 0) resolve(null);
      }, 200);
    };
    window.addEventListener("focus", onFocus);
    input.click();
  });
}

export const webPlatform: Platform = {
  runtime: RUNTIME,
  notify,
  openExternal,
  pickFile,

  // Phase 3 — Ollama supervision is unreachable from the browser. The PWA
  // continues to render the existing /help/ollama text instructions.
  isOllamaInstalled: () => Promise.resolve(false),
  bootstrapOllama: (_onProgress?: (p: OllamaProgress) => void) =>
    Promise.reject(new UnsupportedError("bootstrapOllama", RUNTIME)),
  startOllama: () => Promise.reject(new UnsupportedError("startOllama", RUNTIME)),
  isModelPulled: (_tag: string) => Promise.resolve(false),
  pullModel: (_tag: string, _onProgress?: (p: OllamaProgress) => void) =>
    Promise.reject(new UnsupportedError("pullModel", RUNTIME)),

  // No loopback token in the browser — the server middleware is a pass-through
  // there, so undefined (not an error) is the correct, non-throwing answer.
  getApiToken: () => Promise.resolve(undefined),

  // Phase 4 — native affordances absent in the browser.
  registerHotkey: (_combo: string, _handler: () => void): Promise<HotkeyId> =>
    Promise.reject(new UnsupportedError("registerHotkey", RUNTIME)),
  unregisterHotkey: (_id: HotkeyId) =>
    Promise.reject(new UnsupportedError("unregisterHotkey", RUNTIME)),
  setTrayMenu: (_items: TrayMenuItem[], _onSelect: (id: string) => void) =>
    Promise.reject(new UnsupportedError("setTrayMenu", RUNTIME)),
  getAutostart: () => Promise.resolve(false),
  setAutostart: (_enabled: boolean) =>
    Promise.reject(new UnsupportedError("setAutostart", RUNTIME)),
};

// Re-export for tests and for callers that want to assert against the symbol.
export { unsupported as _webUnsupported };
