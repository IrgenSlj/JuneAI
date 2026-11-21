// The Mac app backend. Tauri's JS modules are imported lazily so the browser
// build never evaluates them.
import type { Platform } from "./types.js";

export const tauriPlatform: Platform = {
  runtime: "tauri",
  async openExternal(url) {
    const { openUrl } = await import("@tauri-apps/plugin-opener");
    await openUrl(url);
  },
  async pickFolder(options) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const picked = await open({
      directory: true,
      multiple: false,
      ...(options?.title === undefined ? {} : { title: options.title }),
      ...(options?.defaultPath === undefined ? {} : { defaultPath: options.defaultPath }),
    });
    return typeof picked === "string" ? picked : null;
  },
  async pickFile(options) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const picked = await open({
      directory: false,
      multiple: false,
      filters: [{ name: "Export", extensions: options.extensions }],
      ...(options.title === undefined ? {} : { title: options.title }),
    });
    return typeof picked === "string" ? picked : null;
  },
  async pickSavePath(options) {
    const { save } = await import("@tauri-apps/plugin-dialog");
    const picked = await save({
      defaultPath: options.defaultName,
      filters: [{ name: options.extensions.join(", "), extensions: options.extensions }],
      ...(options.title === undefined ? {} : { title: options.title }),
    });
    return typeof picked === "string" ? picked : null;
  },
};
