<script lang="ts">
  import type { ActivityStep } from "./types.js";

  interface Props {
    steps: ActivityStep[];
    open: boolean;
  }

  const { steps, open }: Props = $props();

  let scrollEl: HTMLDivElement | undefined = $state();

  function formatTime(ts: number): string {
    const d = new Date(ts);
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
  }

  // Auto-scroll to newest step whenever steps change.
  $effect(() => {
    void steps.length;
    if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;
  });
</script>

{#if open}
  <div
    class="stream"
    bind:this={scrollEl}
    role="log"
    aria-label="June activity"
  >
    {#each steps as step (step.id)}
      <div class="step" data-kind={step.kind}>
        <span class="ts">{formatTime(step.ts)}</span>
        {#if step.kind === "provenance"}
          <span
            class="provenance-dot"
            class:cloud={step.cloud}
            class:local={!step.cloud}
            aria-label={step.cloud ? "cloud" : "local"}
          ></span>
        {/if}
        <span class="label">{step.label}</span>
        {#if step.kind === "reasoning" && step.detail}
          <span class="reasoning-detail">{step.detail}</span>
        {:else if step.detail}
          <span class="detail">{step.detail}</span>
        {/if}
      </div>
    {/each}
  </div>
{/if}

<style>
  .stream {
    overflow-y: auto;
    height: 100%;
    background: var(--color-bg-sunken);
    border-top: 1px solid var(--color-border);
    padding: var(--space-2) var(--space-4);
    display: flex;
    flex-direction: column;
    gap: 2px;
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    line-height: 1.4;
  }

  .stream::-webkit-scrollbar {
    width: 6px;
  }
  .stream::-webkit-scrollbar-thumb {
    background: var(--color-border);
    border-radius: var(--radius-pill);
  }

  .step {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
    min-width: 0;
    flex-wrap: wrap;
  }

  .ts {
    color: var(--color-fg-subtle);
    flex-shrink: 0;
    opacity: 0.6;
  }

  .provenance-dot {
    width: 6px;
    height: 6px;
    border-radius: var(--radius-pill);
    flex-shrink: 0;
    align-self: center;
  }

  .provenance-dot.cloud {
    background: var(--color-accent);
  }

  .provenance-dot.local {
    background: var(--color-success);
  }

  .label {
    color: var(--color-fg-muted);
    flex-shrink: 0;
  }

  .step[data-kind="error"] .label {
    color: var(--color-danger);
  }

  .step[data-kind="done"] .label {
    color: var(--color-fg-subtle);
  }

  .detail {
    color: var(--color-fg-subtle);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
    min-width: 0;
    padding-left: var(--space-3);
  }

  .step[data-kind="reasoning"] .label {
    color: var(--color-fg-subtle);
    font-style: italic;
  }

  .reasoning-detail {
    color: var(--color-fg-subtle);
    font-style: italic;
    font-family: var(--font-sans);
    white-space: pre-wrap;
    word-break: break-word;
    overflow-wrap: break-word;
    flex: 1;
    min-width: 0;
    padding-left: var(--space-3);
    opacity: 0.75;
  }
</style>
