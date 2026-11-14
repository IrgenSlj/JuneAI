// Usage: import { getPlatform } from "@june/ui/platform";
import { tauriPlatform } from "./tauri.js";
import { type Platform, type ShellRuntime, UnsupportedError } from "./types.js";
import { webPlatform } from "./web.js";

declare global {
  interface Window {
    __TAURI_INTERNALS__?: unknown;
  }
}

export function detectRuntime(): ShellRuntime {
  if (typeof window !== "undefined" && window.__TAURI_INTERNALS__ !== undefined) return "tauri";
  return "web";
}

let cached: Platform | null = null;

export function getPlatform(): Platform {
  cached ??= detectRuntime() === "tauri" ? tauriPlatform : webPlatform;
  return cached;
}

export { type Platform, type ShellRuntime, UnsupportedError };
