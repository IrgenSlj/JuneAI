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
        {#if step.kind === "provenance"}
          <!-- The cloud/local boundary — the trust anchor of the terminal. -->
          <div class="boundary" class:cloud={step.cloud} class:local={!step.cloud}>
            <span class="boundary-tag">{step.cloud ? "cloud" : "local"}</span>
            <span class="ts">{formatTime(step.ts)}</span>
            <span class="label">{step.label}</span>
            {#if step.detail}
              <span class="detail">{step.detail}</span>
            {/if}
          </div>
        {:else}
          <div class="step" data-kind={step.kind}>
            <span class="ts">{formatTime(step.ts)}</span>
            <span class="label">{step.label}</span>
            {#if step.kind === "reasoning" && step.detail}
              <span class="reasoning-detail">{step.detail}</span>
            {:else if step.detail}
              <span class="detail">{step.detail}</span>
            {/if}
          </div>
        {/if}
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
    gap: 3px;
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

  /* Boundary line — the cloud/local trust anchor. */
  .boundary {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
    flex-wrap: wrap;
    margin: 2px 0;
    padding: var(--space-2) var(--space-3);
    border-left: 2px solid var(--color-success);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    background: color-mix(in srgb, var(--color-success) 10%, transparent);
  }
  .boundary.cloud {
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
  .boundary.cloud .boundary-tag {
    color: var(--color-warn);
  }

  .boundary .label {
    color: var(--color-fg-secondary);
  }
</style>
