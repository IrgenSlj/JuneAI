import type { RecallHit } from "../api/index.js";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  toolName?: string;
  /** Memories June drew on to compose this message. Only set on assistant messages. */
  recallHits?: RecallHit[];
  /** Model provenance for this assistant turn. */
  provenance?: { provider?: string; model?: string; tier?: string; latency_ms?: number; cloud_call?: boolean; cloud_payload_summary?: string; memories_recalled?: number; skills_called?: string[]; rationale?: string };
}

export interface ActivityStep {
  id: string;
  ts: number;            // Date.now()
  kind: "recall" | "tool" | "tool_result" | "provenance" | "done" | "error" | "reasoning";
  label: string;         // short, June-voiced, lower-case, no emoji
  detail?: string;       // optional secondary line (args, rationale, snippet)
  cloud?: boolean;       // set on provenance steps: true=cloud, false=local
}
