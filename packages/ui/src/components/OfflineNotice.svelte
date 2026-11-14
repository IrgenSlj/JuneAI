<script lang="ts">
  interface Props {
    /** What the user was trying to load — shapes the copy. */
    kind?: "memory" | "skills" | "settings" | "system" | "generic";
    /** Extra detail from the caught error. Optional. */
    detail?: string;
    /** If set, shows a retry button that calls this. */
    onRetry?: () => void | Promise<void>;
    /** Whether a retry is currently in flight. */
    retrying?: boolean;
  }

  const {
    kind = "generic",
    detail,
    onRetry,
    retrying = false,
  }: Props = $props();

  const titles: Record<NonNullable<Props["kind"]>, string> = {
    memory: "Can't reach June's brain",
    skills: "Can't reach June's brain",
    settings: "Can't reach June's brain",
    system: "Can't reach June's brain",
    generic: "Can't reach June's brain",
  };

  const bodies: Record<NonNullable<Props["kind"]>, string> = {
    memory: "Memories are stored in the API, so the list is read-only until the brain comes back online.",
    skills: "The skills registry lives in the API. Toggles and status return when the brain is reachable again.",
    settings: "Settings persist in the brain. Start the API to load and save your configuration.",
    system: "The chat surface needs the brain for streaming replies.",
    generic: "This page loads data from the API and the brain isn't answering right now.",
  };
</script>

<div class="offline" role="status" aria-live="polite">
  <div class="row">
    <span class="dot" aria-hidden="true"></span>
    <div class="text">
      <p class="title">{titles[kind]}</p>
      <p class="body">{bodies[kind]}</p>
      {#if detail}
        <p class="detail"><code>{detail}</code></p>
      {/if}
    </div>
  </div>
  <p class="hint">
    Start the brain with <code>./tools/run.sh</code>, then retry.
  </p>
  {#if onRetry}
    <button type="button" onclick={onRetry} disabled={retrying}>
      {retrying ? "Retrying…" : "Retry"}
    </button>
  {/if}
</div>

<style>
  .offline {
    border: 1px solid var(--color-border);
    background: var(--color-bg-raised);
    border-radius: var(--radius-lg);
    padding: var(--space-4) var(--space-5);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    max-width: 640px;
  }
  .row {
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
  }
  .dot {
    flex-shrink: 0;
    width: 8px;
    height: 8px;
    margin-top: 0.55em;
    border-radius: var(--radius-pill);
    background: var(--color-accent);
  }
  .text {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
  .title {
    margin: 0;
    font-weight: 500;
    color: var(--color-fg-primary);
  }
  .body {
    margin: 0;
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
    line-height: var(--leading-relaxed);
  }
  .detail {
    margin: var(--space-1) 0 0;
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
  }
  .hint {
    margin: 0;
    padding-left: calc(var(--space-3) + 8px);
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
  }
  code {
    font-family: var(--font-mono);
    font-size: 0.92em;
    padding: 0.1em 0.35em;
    background: var(--color-bg-sunken);
    border-radius: var(--radius-sm);
  }
  button {
    align-self: flex-start;
    background: transparent;
    color: var(--color-fg-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-2) var(--space-4);
    font-size: var(--size-sm);
    cursor: pointer;
  }
  button:hover:not(:disabled) {
    color: var(--color-fg-primary);
    border-color: var(--color-border-strong);
  }
  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
