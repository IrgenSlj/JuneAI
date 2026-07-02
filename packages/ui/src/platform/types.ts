// Platform capability layer — the typed surface every shell implements.
//
// Three runtime backends fulfil this interface:
//   - web.ts         — browser APIs only; throws Unsupported for native-only methods.
//   - tauri.ts       — Rust commands invoked through @tauri-apps/api.
//   - capacitor.ts   — stubs until the mobile shell ships.
//
// Shells never call backend modules directly; they import getPlatform() from
// ./index.ts which picks the right backend at module load via runtime detection.
//
// Adding a new method:
//   1. Add it to the Platform interface below.
//   2. Add a TauriCommand entry if the Tauri backend will invoke a Rust command.
//   3. Implement (or stub) it in all three backend files.
//   4. The TypeScript compiler enforces the rest.

export interface NotifyOptions {
  title: string;
  body: string;
  /** Optional ID so a subsequent notification with the same id replaces this one. */
  tag?: string;
}

export interface PickFileOptions {
  /** Accepted MIME types or extensions, e.g. ["application/pdf", ".pdf"]. */
  accept?: string[];
  multiple?: boolean;
}

export interface PickedFile {
  name: string;
  path: string | null;
  bytes: Uint8Array;
}

export interface OllamaProgress {
  /** 0..1, or null when the source does not report a percentage. */
  fraction: number | null;
  status: string;
}

export type HotkeyId = string;

export interface TrayMenuItem {
  id: string;
  label: string;
  /** When true, the item is rendered as a separator and `id`/`label` are ignored. */
  separator?: boolean;
  enabled?: boolean;
}

export type ShellRuntime = "web" | "tauri" | "capacitor";

/**
 * Every method the UI may call on its host shell.
 *
 * Methods marked "native-only" throw {@link UnsupportedError} on the web backend
 * and are stubbed on Capacitor until the mobile shell ships. Calling code is
 * expected to feature-detect via {@link Platform.runtime} or by catching
 * UnsupportedError and falling back to the existing web flow.
 */
export interface Platform {
  /** Which backend is active. Use to gate UI surfaces (e.g. show "Set up Ollama" button only on tauri). */
  readonly runtime: ShellRuntime;

  /** Show a notification to the user. Web uses Notifications API; native shells use the OS facility. */
  notify(options: NotifyOptions): Promise<void>;

  /** Open a URL in the user's default browser. Web uses window.open; native shells use the OS opener. */
  openExternal(url: string): Promise<void>;

  /** Prompt the user to pick one or more files. Returns null when the user cancels. */
  pickFile(options?: PickFileOptions): Promise<PickedFile[] | null>;

  // Native-only: Phase 3 (Ollama supervision) — see ADR 0008.
  isOllamaInstalled(): Promise<boolean>;
  bootstrapOllama(onProgress?: (p: OllamaProgress) => void): Promise<void>;
  startOllama(): Promise<void>;
  isModelPulled(tag: string): Promise<boolean>;
  pullModel(tag: string, onProgress?: (p: OllamaProgress) => void): Promise<void>;

  /**
   * Loopback shared-secret for the packaged sidecar. The desktop shell
   * generates it at startup and enforces it via `JUNE_API_TOKEN`; the webview
   * fetches it here and sends it as `X-June-Token`. Returns undefined off the
   * desktop (browser / mobile), where "no token" is the valid, expected state
   * — so implementations return undefined rather than throwing.
   */
  getApiToken(): Promise<string | undefined>;

  // Native-only: Phase 4 (native affordances).
  registerHotkey(combo: string, handler: () => void): Promise<HotkeyId>;
  unregisterHotkey(id: HotkeyId): Promise<void>;
  setTrayMenu(items: TrayMenuItem[], onSelect: (id: string) => void): Promise<void>;
  getAutostart(): Promise<boolean>;
  setAutostart(enabled: boolean): Promise<void>;
}

/**
 * Thrown by the web backend (and by Capacitor stubs) when a method is not
 * available in the current shell. Caller should fall back to whatever flow the
 * web app uses today (e.g. opening /help/ollama instead of bootstrapping).
 */
export class UnsupportedError extends Error {
  readonly capability: string;
  readonly runtime: ShellRuntime;
  constructor(capability: string, runtime: ShellRuntime) {
    super(`${capability} is not supported on ${runtime}`);
    this.name = "UnsupportedError";
    this.capability = capability;
    this.runtime = runtime;
  }
}

/**
 * The exhaustive set of Rust commands the Tauri backend may invoke. Keep this
 * union in lockstep with `apps/desktop/src-tauri/src/commands/`. The typed
 * wrapper in tauri.ts uses this union as the only allowed argument to invoke(),
 * which is the mechanism that makes "type system catches a typo'd command name"
 * true (Phase 2 acceptance criterion).
 */
export type TauriCommand =
  | "pick_file"
  | "is_ollama_installed"
  | "bootstrap_ollama"
  | "start_ollama"
  | "is_model_pulled"
  | "pull_model"
  | "get_api_token"
  | "register_hotkey"
  | "unregister_hotkey"
  | "set_tray_menu"
  | "get_autostart"
  | "set_autostart";
