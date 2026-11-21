// Platform capability layer: what the UI may ask of its host shell.
//
//   web.ts    browser APIs only; native-only methods throw UnsupportedError.
//   tauri.ts  the Mac app, through Tauri's JS API.
//
// The UI never imports a backend directly; getPlatform() in ./index.ts picks
// one at runtime. Adding a method: add it here, implement it in both backends.

export type ShellRuntime = "web" | "tauri";

export interface Platform {
  readonly runtime: ShellRuntime;

  /** Open a URL in the user's default browser, outside June's window. */
  openExternal(url: string): Promise<void>;

  /**
   * Native folder picker for choosing a vault. Returns an absolute path, or
   * null when the user cancels. Native-only: the browser cannot hand a real
   * path to a page, so the web UI asks for a typed path instead.
   */
  pickFolder(options?: { title?: string; defaultPath?: string }): Promise<string | null>;

  /** Native file picker; an absolute path or null. Native-only: the web UI uploads the file instead. */
  pickFile(options: { title?: string; extensions: string[] }): Promise<string | null>;

  /** Native save dialog; the chosen absolute path or null. Native-only: the web UI downloads instead. */
  pickSavePath(options: { title?: string; defaultName: string; extensions: string[] }): Promise<string | null>;
}

/** Thrown when a capability does not exist in the current shell. Catch it and fall back. */
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
