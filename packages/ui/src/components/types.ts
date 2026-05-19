import type { RecallHit } from "../api/index.js";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  toolName?: string;
  /** Memories June drew on to compose this message. Only set on assistant messages. */
  recallHits?: RecallHit[];
  /** Model provenance for this assistant turn. */
  provenance?: { provider?: string; model?: string; tier?: string; latency_ms?: number };
}
