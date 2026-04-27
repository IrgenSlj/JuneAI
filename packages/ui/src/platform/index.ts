// Public entry for the platform capability layer.
//
// Usage:
//   import { getPlatform } from "@june/ui/platform";
//   const platform = getPlatform();
//   await platform.notify({ title: "June", body: "Hello." });
//
// The right backend is selected at module load via runtime detection on
// `window`. SSR (no window) falls back to the web backend so importing the
// module never throws; calls that require a browser will no-op or reject.
//
// Adding a method goes through types.ts; see the comment at the top of that file.

import { capacitorPlatform } from "./capacitor.js";
import { tauriPlatform } from "./tauri.js";
import {
  type NotifyOptions,
  type PickedFile,
  type PickFileOptions,
  type Platform,
  type ShellRuntime,
  UnsupportedError,
} from "./types.js";
import { webPlatform } from "./web.js";

declare global {
  interface Window {
    __TAURI__?: unknown;
    __TAURI_INTERNALS__?: unknown;
    Capacitor?: { isNativePlatform?: () => boolean };
  }
}

function detectRuntime(): ShellRuntime {
  if (typeof window === "undefined") return "web";
  if (window.__TAURI__ !== undefined || window.__TAURI_INTERNALS__ !== undefined) return "tauri";
  if (window.Capacitor && window.Capacitor.isNativePlatform?.()) return "capacitor";
  return "web";
}

let cached: Platform | null = null;

export function getPlatform(): Platform {
  if (cached) return cached;
  switch (detectRuntime()) {
    case "tauri":
      cached = tauriPlatform;
      break;
    case "capacitor":
      cached = capacitorPlatform;
      break;
    default:
      cached = webPlatform;
  }
  return cached;
}

/**
 * Reset the cached platform. Tests only — never call from app code.
 * @internal
 */
export function _resetPlatformCache(): void {
  cached = null;
}

export {
  type NotifyOptions,
  type PickedFile,
  type PickFileOptions,
  type Platform,
  type ShellRuntime,
  UnsupportedError,
};
