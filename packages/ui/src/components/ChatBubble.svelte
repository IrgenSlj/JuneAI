<script lang="ts">
  /**
   * One chat bubble — user, assistant, or a tool marker.
   *
   * Tool bubbles collapse the raw JSON args/result into a single line
   * so the transcript stays readable; clicking expands the detail.
   * Assistant bubbles can carry a "memories used" disclosure listing
   * the recall hits that fed this turn.
   */
  import type { RecallHit } from "../api/index.js";

  interface Props {
    role: "user" | "assistant" | "tool";
    content: string;
    toolName?: string;
    /** True for the tail of an active stream, so an empty assistant bubble can pulse. */
    pending?: boolean;
    /** Memories June drew on to compose this message. */
    recallHits?: RecallHit[];
    /** Base path of the memory browser, for deep-linking recall hits. */
    memoryHref?: string;
    /**
     * Called when the user thumbs-up / thumbs-down a recalled memory.
     * Vote 'clear' fires when the user re-clicks the active button.
     * The component does not own the API call — the consumer wires it
     * through to the server and is responsible for any error UI.
     */
    onVote?: (ref: string, vote: "up" | "down" | "clear") => void | Promise<void>;
    /** Model provenance for this assistant turn. */
    provenance?: { provider?: string; model?: string; tier?: string; latency_ms?: number; cloud_call?: boolean; cloud_payload_summary?: string; memories_recalled?: number; skills_called?: string[]; rationale?: string };
  }

  const {
    role,
    content,
    toolName = "",
    pending = false,
    recallHits = [],
    memoryHref = "/memory",
    onVote,
    provenance,
  }: Props = $props();

  /** Per-bubble vote overrides applied locally so the button reflects the user's
   * latest click before the next recall pulls the persisted state. */
  let voteOverrides: Record<string, "up" | "down" | ""> = $state({});

  function currentVote(hit: RecallHit): "up" | "down" | "" {
    const ref = hit.ref ?? "";
    if (ref in voteOverrides) return voteOverrides[ref];
    const fb = hit.feedback;
    return fb === "up" || fb === "down" ? fb : "";
  }

  async function handleVote(hit: RecallHit, intent: "up" | "down") {
    if (!onVote || !hit.ref) return;
    const active = currentVote(hit);
    const next: "up" | "down" | "clear" = active === intent ? "clear" : intent;
    voteOverrides = { ...voteOverrides, [hit.ref]: next === "clear" ? "" : next };
    try {
      await onVote(hit.ref, next);
    } catch {
      // Roll back the local override; keep behaviour silent — the consumer
      // can surface its own error UI if the call fails.
      const { [hit.ref]: _drop, ...rest } = voteOverrides;
      voteOverrides = rest;
    }
  }

  const hasRecall = $derived(role === "assistant" && recallHits.length > 0);

  const SOURCE_LABELS: Record<string, string> = {
    vector: "fact",
    graph: "entity",
    sqlite: "structured",
  };

  function sourceLabel(hit: RecallHit): string {
    return SOURCE_LABELS[hit.source] ?? hit.source ?? "memory";
  }
  const showThinking = $derived(
    pending && role === "assistant" && content.length === 0,
  );
  const showCopy = $derived(
    role === "assistant" && !pending && content.length > 0,
  );

  const hasProvenance = $derived(role === "assistant" && !!provenance?.model);

  let copied = $state(false);
  async function copy() {
    if (typeof navigator === "undefined" || !navigator.clipboard) return;
    try {
      await navigator.clipboard.writeText(content);
      copied = true;
      setTimeout(() => (copied = false), 1500);
    } catch {
      // Clipboard write can fail in insecure contexts; stay silent.
    }
  }
</script>

<article class="bubble" data-role={role}>
  {#if role === "tool"}
    <header class="tool-header">
      <span class="tool-dot" aria-hidden="true"></span>
      <span class="tool-label">{toolName || "tool"}</span>
    </header>
  {/if}
  <div class="body" class:tool={role === "tool"}>
    {#if showThinking}
      <span class="thinking" aria-label="June is thinking">
        <span class="pulse"></span>
        <span class="pulse"></span>
        <span class="pulse"></span>
      </span>
    {:else}
      {content}
    {/if}
  </div>
  {#if showCopy}
    <button
      type="button"
      class="copy"
      onclick={copy}
      aria-label={copied ? "Copied" : "Copy message"}
      title={copied ? "Copied" : "Copy"}
    >
      {copied ? "Copied" : "Copy"}
    </button>
  {/if}
  {#if hasRecall}
    <details class="recall">
      <summary>
        {recallHits.length} memor{recallHits.length === 1 ? "y" : "ies"} used
      </summary>
      <ul class="recall-list">
        {#each recallHits as hit (hit.ref || hit.text)}
          {@const vote = currentVote(hit)}
          <li class="recall-item">
            <a class="recall-link" href={memoryHref}>
              <span class="recall-source">{sourceLabel(hit)}</span>
              <span class="recall-text">{hit.text}</span>
            </a>
            {#if onVote && hit.ref}
              <div class="recall-vote">
                <button
                  type="button"
                  class="vote up"
                  class:active={vote === "up"}
                  onclick={() => handleVote(hit, "up")}
                  aria-label="Helpful"
                  aria-pressed={vote === "up"}
                  title={vote === "up" ? "Marked helpful — click to clear" : "Helpful"}
                >
                  Helpful
                </button>
                <button
                  type="button"
                  class="vote down"
                  class:active={vote === "down"}
                  onclick={() => handleVote(hit, "down")}
                  aria-label="Wrong"
                  aria-pressed={vote === "down"}
                  title={vote === "down" ? "Marked wrong — click to clear" : "Wrong"}
                >
                  Wrong
                </button>
              </div>
            {/if}
          </li>
        {/each}
      </ul>
    </details>
  {/if}
  {#if hasProvenance}
    <div class="provenance-block">
      <div class="provenance-chip">
        {#if provenance!.cloud_call}
          <span
            class="provenance-cloud-badge"
            title={provenance!.cloud_payload_summary ?? "Data sent to cloud"}
          >Cloud</span>
          <span class="provenance-sep">·</span>
        {:else}
          <span class="provenance-local-badge">Local</span>
          <span class="provenance-sep">·</span>
        {/if}
        <span class="provenance-tier">{provenance!.tier}</span>
        <span class="provenance-sep">·</span>
        <span class="provenance-model">{provenance!.model}</span>
        {#if provenance!.latency_ms}
          <span class="provenance-sep">·</span>
          <span class="provenance-latency">{provenance!.latency_ms}ms</span>
        {/if}
      </div>
      {#if provenance!.rationale}
        <div class="provenance-rationale">{provenance!.rationale}</div>
      {/if}
    </div>
  {/if}
</article>

<style>
  .bubble {
    position: relative;
    max-width: 72ch;
    padding: var(--space-3) var(--space-4);
    border-radius: var(--radius-lg);
    font-size: var(--size-md);
    line-height: var(--leading-relaxed);
    white-space: pre-wrap;
    word-wrap: break-word;
    box-shadow: var(--shadow-sm);
  }

  .copy {
    position: absolute;
    top: var(--space-2);
    right: var(--space-2);
    padding: 2px 8px;
    font-size: var(--size-xs);
    font-family: var(--font-sans);
    color: var(--color-fg-subtle);
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    cursor: pointer;
    opacity: 0;
    transition: opacity 120ms ease;
  }
  .bubble:hover .copy,
  .copy:focus-visible {
    opacity: 1;
  }
  .copy:hover {
    color: var(--color-fg-primary);
    border-color: var(--color-border-strong);
  }
  .bubble[data-role="assistant"] .copy {
    top: 0;
    right: 0;
  }

  .bubble[data-role="user"] {
    align-self: flex-end;
    background: var(--color-bg-raised);
    color: var(--color-fg-primary);
    border: 1px solid var(--color-border);
  }

  .bubble[data-role="assistant"] {
    align-self: flex-start;
    background: transparent;
    color: var(--color-fg-primary);
    padding-left: 0;
    padding-right: 0;
    box-shadow: none;
  }

  .bubble[data-role="tool"] {
    align-self: flex-start;
    background: var(--color-bg-sunken);
    color: var(--color-fg-muted);
    border: 1px dashed var(--color-border);
    font-family: var(--font-mono);
    font-size: var(--size-xs);
  }

  .tool-header {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: var(--space-1);
  }

  .tool-dot {
    width: 6px;
    height: 6px;
    border-radius: var(--radius-pill);
    background: var(--color-accent);
  }

  .tool-label {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-accent);
    letter-spacing: 0.02em;
  }

  .body.tool {
    color: var(--color-fg-muted);
  }

  .thinking {
    display: inline-flex;
    gap: 4px;
    padding: 2px 0;
  }
  .pulse {
    width: 6px;
    height: 6px;
    border-radius: var(--radius-pill);
    background: var(--color-fg-subtle);
    animation: june-pulse 1.1s ease-in-out infinite;
  }
  .pulse:nth-child(2) {
    animation-delay: 0.15s;
  }
  .pulse:nth-child(3) {
    animation-delay: 0.3s;
  }
  @keyframes june-pulse {
    0%, 80%, 100% {
      opacity: 0.25;
      transform: scale(0.85);
    }
    40% {
      opacity: 1;
      transform: scale(1);
    }
  }

  .recall {
    margin-top: var(--space-2);
    font-size: var(--size-xs);
  }

  .recall summary {
    cursor: pointer;
    color: var(--color-fg-subtle);
    font-family: var(--font-mono);
    list-style: none;
    padding: var(--space-1) 0;
  }
  .recall summary::-webkit-details-marker {
    display: none;
  }
  .recall summary::before {
    content: "▸ ";
    font-size: 0.85em;
  }
  .recall[open] summary::before {
    content: "▾ ";
  }
  .recall summary:hover {
    color: var(--color-fg-muted);
  }

  .recall-list {
    list-style: none;
    padding: var(--space-2) 0 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }

  .recall-item {
    display: flex;
    gap: var(--space-2);
    align-items: center;
  }

  .recall-link {
    display: flex;
    gap: var(--space-2);
    align-items: baseline;
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius-sm);
    color: inherit;
    text-decoration: none;
    border: 1px solid transparent;
    flex: 1;
    min-width: 0;
  }
  .recall-link:hover {
    background: var(--color-bg-raised);
    border-color: var(--color-border);
  }

  .recall-vote {
    display: flex;
    gap: var(--space-1);
    flex-shrink: 0;
  }

  .vote {
    background: transparent;
    color: var(--color-fg-subtle);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: 2px 8px;
    font: inherit;
    font-size: var(--size-xs);
    cursor: pointer;
  }
  .vote:hover {
    color: var(--color-fg-primary);
    border-color: var(--color-border-strong);
  }
  .vote.up.active {
    color: var(--color-accent);
    border-color: var(--color-accent);
    background: color-mix(in srgb, var(--color-accent) 10%, transparent);
  }
  .vote.down.active {
    color: var(--color-danger);
    border-color: var(--color-danger);
    background: color-mix(in srgb, var(--color-danger) 10%, transparent);
  }

  .recall-source {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    flex-shrink: 0;
    min-width: 5em;
  }

  .recall-text {
    color: var(--color-fg-muted);
    line-height: var(--leading-normal);
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .provenance-block {
    margin-top: var(--space-2);
    display: flex;
    flex-direction: column;
    gap: 0.25em;
  }

  .provenance-chip {
    display: flex;
    align-items: center;
    gap: 0.3em;
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
  }

  .provenance-sep {
    opacity: 0.4;
  }

  .provenance-tier,
  .provenance-model,
  .provenance-latency {
    color: var(--color-fg-subtle);
  }

  .provenance-cloud-badge {
    display: inline-block;
    padding: 0 5px;
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    font-weight: 600;
    color: var(--color-accent);
    border: 1px solid var(--color-accent);
    background: color-mix(in srgb, var(--color-accent) 10%, transparent);
    cursor: default;
  }

  .provenance-local-badge {
    display: inline-block;
    padding: 0 5px;
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
    border: 1px solid var(--color-border);
  }

  .provenance-rationale {
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
    font-family: var(--font-sans);
    line-height: var(--leading-normal);
  }
</style>
