import { type Platform, UnsupportedError } from "./types.js";

export const webPlatform: Platform = {
  runtime: "web",
  async openExternal(url) {
    window.open(url, "_blank", "noopener,noreferrer");
  },
  async pickFolder() {
    throw new UnsupportedError("pickFolder", "web");
  },
  async pickFile() {
    throw new UnsupportedError("pickFile", "web");
  },
  async pickSavePath() {
    throw new UnsupportedError("pickSavePath", "web");
  },
};
