// Capacitor backend — stubs only. The mobile shell is trigger-gated and not
// yet shipped (see docs/product/roadmap.md). Every method throws
// UnsupportedError so the UI can feature-detect today and the implementation
// can drop in here without touching callers when the mobile shell is built.

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

const RUNTIME = "capacitor" as const;
const reject = (capability: string) =>
  Promise.reject(new UnsupportedError(capability, RUNTIME));

export const capacitorPlatform: Platform = {
  runtime: RUNTIME,
  notify: (_o: NotifyOptions) => reject("notify"),
  openExternal: (_u: string) => reject("openExternal"),
  pickFile: (_o?: PickFileOptions) =>
    reject("pickFile") as Promise<PickedFile[] | null>,

  isOllamaInstalled: () => Promise.resolve(false),
  bootstrapOllama: (_o?: (p: OllamaProgress) => void) => reject("bootstrapOllama"),
  startOllama: () => reject("startOllama"),
  isModelPulled: (_t: string) => Promise.resolve(false),
  pullModel: (_t: string, _o?: (p: OllamaProgress) => void) => reject("pullModel"),

  registerHotkey: (_c: string, _h: () => void) =>
    reject("registerHotkey") as Promise<HotkeyId>,
  unregisterHotkey: (_id: HotkeyId) => reject("unregisterHotkey"),
  setTrayMenu: (_i: TrayMenuItem[], _s: (id: string) => void) => reject("setTrayMenu"),
  getAutostart: () => Promise.resolve(false),
  setAutostart: (_e: boolean) => reject("setAutostart"),
};
