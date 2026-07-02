// Tauri backend — implemented for the desktop shell. The methods that need a
// Rust command go through `tauriInvoke<T>(cmd, args)` whose first argument is
// a TauriCommand (a closed string union), so a typo at the call site is a
// compile error — that is the wiring contract the type system enforces.
//
// Only `notify` is wired end-to-end in Phase 2; Phase 3 fills in the Ollama
// methods, Phase 4 fills in the hotkey / tray / autostart methods. Until then
// they invoke commands the Rust side does not yet register, which throws at
// runtime — acceptable because the UI should feature-detect via runtime.

import {
  type HotkeyId,
  type NotifyOptions,
  type OllamaProgress,
  type PickedFile,
  type PickFileOptions,
  type Platform,
  type TauriCommand,
  type TrayMenuItem,
} from "./types.js";

const RUNTIME = "tauri" as const;

interface TauriCoreApi {
  invoke: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
}

interface TauriEventApi {
  listen: <T>(event: string, handler: (e: { payload: T }) => void) => Promise<() => void>;
}

interface TauriNotificationApi {
  isPermissionGranted: () => Promise<boolean>;
  requestPermission: () => Promise<"granted" | "denied" | "default">;
  sendNotification: (options: { title: string; body: string }) => void;
}

const coreImport = () =>
  import(/* @vite-ignore */ "@tauri-apps/api/core") as Promise<TauriCoreApi>;
const eventImport = () =>
  import(/* @vite-ignore */ "@tauri-apps/api/event") as Promise<TauriEventApi>;
const notificationImport = () =>
  import(/* @vite-ignore */ "@tauri-apps/plugin-notification") as Promise<TauriNotificationApi>;

async function tauriInvoke<T>(cmd: TauriCommand, args?: Record<string, unknown>): Promise<T> {
  const { invoke } = await coreImport();
  return invoke<T>(cmd, args);
}

async function notify(options: NotifyOptions): Promise<void> {
  const api = await notificationImport();
  let granted = await api.isPermissionGranted();
  if (!granted) granted = (await api.requestPermission()) === "granted";
  if (!granted) return;
  api.sendNotification({ title: options.title, body: options.body });
}

async function openExternal(url: string): Promise<void> {
  // Use the shell plugin's default `open` permission (allowed in capabilities).
  const mod = (await import(/* @vite-ignore */ "@tauri-apps/plugin-shell")) as {
    open: (path: string) => Promise<void>;
  };
  await mod.open(url);
}

async function pickFile(options?: PickFileOptions): Promise<PickedFile[] | null> {
  // Delegate to dialog plugin once it is added (Phase 3); until then fall back
  // to invoking a Rust command we have not yet registered. Callers should
  // prefer the web pickFile until the dialog plugin is wired.
  return tauriInvoke<PickedFile[] | null>("pick_file", { options: options ?? null });
}

async function withProgress(
  event: string,
  promise: Promise<void>,
  onProgress?: (p: OllamaProgress) => void,
): Promise<void> {
  if (!onProgress) return promise;
  const { listen } = await eventImport();
  const unlisten = await listen<OllamaProgress>(event, (e) => onProgress(e.payload));
  try {
    await promise;
  } finally {
    unlisten();
  }
}

export const tauriPlatform: Platform = {
  runtime: RUNTIME,
  notify,
  openExternal,
  pickFile,

  isOllamaInstalled: () => tauriInvoke<boolean>("is_ollama_installed"),
  bootstrapOllama: (onProgress) =>
    withProgress("ollama-install-progress", tauriInvoke<void>("bootstrap_ollama"), onProgress),
  startOllama: () => tauriInvoke<void>("start_ollama"),
  isModelPulled: (tag: string) => tauriInvoke<boolean>("is_model_pulled", { tag }),
  pullModel: (tag, onProgress) =>
    withProgress("ollama-pull-progress", tauriInvoke<void>("pull_model", { tag }), onProgress),

  // Fetch the loopback token the Rust shell generated and set on the sidecar
  // env. Return undefined on any error so a token-fetch failure degrades to the
  // no-header path rather than breaking the app — but log loudly: this runs
  // only in the Tauri platform, so a failure here means the webview cannot
  // authenticate to the sidecar and every request will 401. Silent degradation
  // would make that undiagnosable.
  getApiToken: async () => {
    try {
      return await tauriInvoke<string>("get_api_token");
    } catch (err) {
      console.error(
        "June: failed to fetch the loopback API token from the desktop shell; " +
          "requests to the local API will be unauthenticated and may be rejected.",
        err,
      );
      return undefined;
    }
  },

  registerHotkey: async (combo, handler) => {
    const { listen } = await eventImport();
    const id = await tauriInvoke<HotkeyId>("register_hotkey", { combo });
    await listen<string>(`hotkey:${id}`, () => handler());
    return id;
  },
  unregisterHotkey: (id) => tauriInvoke<void>("unregister_hotkey", { id }),
  setTrayMenu: async (items, onSelect) => {
    const { listen } = await eventImport();
    await listen<string>("tray:select", (e) => onSelect(e.payload));
    await tauriInvoke<void>("set_tray_menu", { items });
  },
  getAutostart: () => tauriInvoke<boolean>("get_autostart"),
  setAutostart: (enabled) => tauriInvoke<void>("set_autostart", { enabled }),
};
