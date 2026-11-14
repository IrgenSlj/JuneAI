<script lang="ts">
  /**
   * EgressLine — the persistent, calm trust signal (ADR 0022). Two truthful
   * states: clean ("nothing left today") and touched ("N cloud calls today").
   * Calm in both; never a red badge. `as="card"` renders the fuller home/rail
   * version with a "view receipts" action.
   */
  interface Props {
    /** egress_today from the ledger summary. */
    calls?: number;
    as?: "line" | "card";
    onView?: () => void;
  }
  const { calls = 0, as = "line", onView }: Props = $props();
  const clean = $derived(!calls);
  const text = $derived(
    clean
      ? "Nothing has left this machine today"
      : `${calls} cloud call${calls > 1 ? "s" : ""} today`,
  );
</script>

{#if as === "card"}
  <div class="card">
    <div class="head" class:touched={!clean}>
      <svg class="shield" width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
        <path d="M8 1.6l5 1.9v4.1c0 3-2.1 5-5 6.8-2.9-1.8-5-3.8-5-6.8V3.5L8 1.6z"
          stroke="currentColor" stroke-width="1.3" stroke-linejoin="round" />
      </svg>
      <span class="headline">{text}</span>
    </div>
    <p class="sub">
      {clean
        ? "Recall and chat stayed on the machine. This is the ordinary day."
        : "Each one was shown before it was sent, and is written down for good."}
    </p>
    {#if onView}
      <button type="button" class="view-btn" onclick={onView}>
        {clean ? "Open receipts" : "View receipts"}
      </button>
    {/if}
  </div>
{:else}
  <div class="line" class:touched={!clean}>
    <svg class="shield" width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="M8 1.6l5 1.9v4.1c0 3-2.1 5-5 6.8-2.9-1.8-5-3.8-5-6.8V3.5L8 1.6z"
        stroke="currentColor" stroke-width="1.3" stroke-linejoin="round" />
    </svg>
    <span class="text">{text}</span>
    {#if !clean && onView}
      <button type="button" class="view-link" onclick={onView}>view</button>
    {/if}
  </div>
{/if}

<style>
  /* The shield carries the only color: success when clean, warn when touched. */
  .line,
  .head {
    color: var(--color-success);
  }
  .line.touched,
  .head.touched {
    color: var(--color-warn);
  }
  .shield {
    flex-shrink: 0;
  }

  .line {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    font-family: var(--font-sans);
    font-size: 12.5px;
  }
  .line .text {
    color: var(--color-fg-secondary);
  }
  .view-link {
    appearance: none;
    border: none;
    background: transparent;
    cursor: pointer;
    padding: 0;
    font-family: var(--font-sans);
    font-size: 12.5px;
    color: var(--color-warn);
    text-decoration: underline;
    text-underline-offset: 2px;
  }

  .card {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    background: var(--color-bg-raised);
    padding: 16px 18px;
  }
  .head {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .headline {
    font-family: var(--font-sans);
    font-size: 14.5px;
    color: var(--color-fg-primary);
    letter-spacing: -0.005em;
  }
  .sub {
    margin: 8px 0 0;
    font-family: var(--font-sans);
    font-size: 12.5px;
    color: var(--color-fg-muted);
    line-height: 1.55;
  }
  .view-btn {
    appearance: none;
    cursor: pointer;
    margin-top: 12px;
    border: 1px solid var(--color-border-strong);
    background: transparent;
    color: var(--color-fg-secondary);
    font-family: var(--font-sans);
    font-size: 12.5px;
    padding: 6px 12px;
    border-radius: var(--radius-md);
  }
  .view-btn:hover {
    border-color: var(--color-fg-muted);
  }
</style>
