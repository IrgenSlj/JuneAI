import type { RecallHit } from "../api/index.js";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  toolName?: string;
  /** Memories June drew on to compose this message. Only set on assistant messages. */
  recallHits?: RecallHit[];
  /** Model provenance for this assistant turn. */
  provenance?: { provider?: string; model?: string; tier?: string; latency_ms?: number; cloud_call?: boolean; cloud_payload_summary?: string; memories_recalled?: number; skills_called?: string[]; rationale?: string; difficulty?: string; difficulty_source?: string };
}

export interface ActivityStep {
  id: string;
  ts: number;            // Date.now()
  kind:
    | "recall"
    | "tool"
    | "tool_result"
    | "provenance"
    | "done"
    | "error"
    | "reasoning"
    | "prompt"        // the rendered "LLM factory" input for an iteration
    | "iteration"     // a loop pass + its intermediate model output
    | "compaction"    // conversation compacted into the pinned-state anchor
    | "tool_blocked"; // a tool withheld: Local-only mode, or pending user approval
  label: string;         // short, June-voiced, lower-case, no emoji
  detail?: string;       // full expandable body (prompt, args, rationale, output)
  cloud?: boolean;       // set on provenance steps: true=cloud, false=local
  network?: boolean;     // set on tool steps: true when the tool reaches the network (egress)
}
