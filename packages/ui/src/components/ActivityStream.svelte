<script lang="ts">
  import type { ActivityStep } from "./types.js";

  interface Props {
    steps: ActivityStep[];
    open: boolean;
  }

  const { steps, open }: Props = $props();

  let scrollEl: HTMLDivElement | undefined = $state();
  // Step ids whose full detail is expanded. Reassigned (not mutated) so Svelte
  // tracks the change.
  let expanded = $state<Set<string>>(new Set());

  function toggle(id: string): void {
    const next = new Set(expanded);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    expanded = next;
  }

  function formatTime(ts: number): string {
    const d = new Date(ts);
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
  }

  function lineCount(detail: string | undefined): number {
    if (!detail) return 0;
    return detail.split("\n").length;
  }

  // Auto-scroll to newest step whenever steps change (but not when the user is
  // expanding an older row).
  $effect(() => {
    void steps.length;
    if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;
  });
</script>

{#if open}
  <div class="terminal">
    <div class="term-header">
      <span class="term-title">activity</span>
      <span class="term-live" aria-hidden="true"></span>
      <span class="term-count">
        {steps.length} step{steps.length === 1 ? "" : "s"}
      </span>
    </div>
    <div class="stream" bind:this={scrollEl} role="log" aria-label="June activity">
      {#each steps as step (step.id)}
        {@const hasDetail = !!step.detail}
        {@const isOpen = expanded.has(step.id)}
        {@const isBoundary = step.kind === "provenance"}
        <div
          class="row"
          class:boundary={isBoundary}
          class:cloud={isBoundary && step.cloud}
          class:local={isBoundary && !step.cloud}
          class:egress={step.network}
          data-kind={step.kind}
        >
          <button
            type="button"
            class="line"
            class:clickable={hasDetail}
            disabled={!hasDetail}
            aria-expanded={hasDetail ? isOpen : undefined}
            onclick={() => hasDetail && toggle(step.id)}
          >
            <span class="ts">{formatTime(step.ts)}</span>
            {#if isBoundary}
              <span class="boundary-tag">{step.cloud ? "cloud" : "local"}</span>
            {/if}
            <span class="label">{step.label}</span>
            {#if hasDetail}
              <span class="expand">
                {#if isOpen}
                  collapse
                {:else}
                  +{lineCount(step.detail)} line{lineCount(step.detail) === 1 ? "" : "s"}
                {/if}
              </span>
            {/if}
          </button>
          {#if hasDetail && isOpen}
            <pre
              class="detail"
              class:reasoning={step.kind === "reasoning"}>{step.detail}</pre>
          {/if}
        </div>
      {/each}
    </div>
  </div>
{/if}

<style>
  .terminal {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--color-term-bg);
    border-top: 1px solid var(--color-border);
  }

  .term-header {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: 40px;
    padding: 0 var(--space-4);
    font-family: var(--font-mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--color-fg-subtle);
  }
  .term-title {
    color: var(--color-fg-muted);
  }
  .term-live {
    width: 6px;
    height: 6px;
    border-radius: var(--radius-pill);
    background: var(--color-accent);
    animation: june-term-live var(--motion-pulse, 1100ms) var(--ease, ease) infinite;
  }
  @keyframes june-term-live {
    0%, 100% { opacity: 0.2; }
    50% { opacity: 1; }
  }
  .term-count {
    margin-left: auto;
  }

  .stream {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: var(--space-2) var(--space-4) var(--space-3);
    display: flex;
    flex-direction: column;
    gap: 2px;
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    line-height: 1.5;
  }

  .stream::-webkit-scrollbar {
    width: 6px;
  }
  .stream::-webkit-scrollbar-thumb {
    background: var(--color-border);
    border-radius: var(--radius-pill);
  }

  .row {
    display: flex;
    flex-direction: column;
    min-width: 0;
  }

  .line {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
    width: 100%;
    min-width: 0;
    border: none;
    background: transparent;
    padding: 1px 0;
    margin: 0;
    text-align: left;
    font: inherit;
    color: inherit;
    cursor: default;
    border-radius: var(--radius-sm);
  }
  .line.clickable {
    cursor: pointer;
  }
  .line.clickable:hover {
    background: color-mix(in srgb, var(--color-fg-muted) 8%, transparent);
  }

  .ts {
    color: var(--color-fg-subtle);
    flex-shrink: 0;
    opacity: 0.6;
  }

  .label {
    color: var(--color-fg-muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
    min-width: 0;
  }

  .expand {
    flex-shrink: 0;
    color: var(--color-accent);
    opacity: 0.8;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .row[data-kind="error"] .label {
    color: var(--color-danger);
  }

  /* Egress — a tool reached the network. Surfaced amber so a query leaving the
     machine in local-only mode is never silent. */
  .row.egress:not(.boundary) {
    border-left: 2px solid var(--color-warn);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding-left: var(--space-2);
    background: color-mix(in srgb, var(--color-warn) 9%, transparent);
  }
  .row.egress .label {
    color: var(--color-warn);
  }
  .row[data-kind="done"] .label {
    color: var(--color-fg-subtle);
  }
  .row[data-kind="prompt"] .label,
  .row[data-kind="iteration"] .label {
    color: var(--color-fg-subtle);
  }
  .row[data-kind="reasoning"] .label {
    color: var(--color-fg-subtle);
    font-style: italic;
  }

  .detail {
    margin: var(--space-1) 0 var(--space-2);
    padding: var(--space-2) var(--space-3);
    background: color-mix(in srgb, var(--color-fg-muted) 7%, transparent);
    border-left: 2px solid var(--color-border-strong);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    max-height: 320px;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--color-fg-secondary);
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    line-height: 1.5;
  }
  .detail.reasoning {
    font-family: var(--font-sans);
    font-style: italic;
    color: var(--color-fg-muted);
  }

  /* Boundary line — the cloud/local trust anchor. */
  .row.boundary {
    margin: 2px 0;
    padding: var(--space-1) var(--space-3);
    border-left: 2px solid var(--color-success);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    background: color-mix(in srgb, var(--color-success) 10%, transparent);
  }
  .row.boundary.cloud {
    border-left-color: var(--color-warn);
    background: var(--color-accent-soft);
  }
  .boundary-tag {
    flex-shrink: 0;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 10px;
    color: var(--color-success);
  }
  .row.boundary.cloud .boundary-tag {
    color: var(--color-warn);
  }
  .row.boundary .label {
    color: var(--color-fg-secondary);
  }
</style>
