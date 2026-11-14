// Route presets hold only facts that were checked. A preset with no checked
// data practice says "unknown" rather than guess. Remote presets arrive with
// their checked sources in a later M1 slice.
import type { Route } from "./types.js";

export const OLLAMA: Route = {
  id: "ollama",
  name: "Ollama (this Mac)",
  kind: "local",
  host: "localhost:11434",
  leavesDevice: false,
  billing: "none (local)",
  dataNote: "Runs on this Mac. Nothing is sent anywhere.",
};

export const LM_STUDIO: Route = {
  id: "lm-studio",
  name: "LM Studio (this Mac)",
  kind: "local",
  host: "localhost:1234",
  leavesDevice: false,
  billing: "none (local)",
  dataNote: "Runs on this Mac. Nothing is sent anywhere.",
};

/** Base URLs of the OpenAI-compatible endpoints the local presets use. */
export const LOCAL_BASE_URLS: Record<string, string> = {
  ollama: "http://localhost:11434/v1",
  "lm-studio": "http://localhost:1234/v1",
};
