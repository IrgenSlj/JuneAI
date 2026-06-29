<script lang="ts">
  /**
   * VerifyAffordance — the "prove it" moment for the Trust Ledger (ADR 0022).
   * Renders the hash chain as a calm, checkable claim — never raw hashes.
   * Controlled by `state`; clicking Verify calls `onVerify` (the parent runs
   * POST /system/ledger/verify and flips state to verifying → verified|tampered).
   */
  type VerifyState = "unknown" | "verified" | "verifying" | "tampered";
  interface Props {
    state?: VerifyState;
    /** Total entries in the chain. */
    count?: number;
    /** First seq where the chain broke, when tampered. */
    firstBrokenSeq?: number | null;
    onVerify?: () => void | Promise<void>;
  }
  const {
    state = "unknown",
    count = 0,
    firstBrokenSeq = null,
    onVerify,
  }: Props = $props();

  const busy = $derived(state === "verifying");
  const tampered = $derived(state === "tampered");

  const headline = $derived(
    busy
      ? "Checking every link…"
      : tampered
        ? "An entry was changed after it was written."
        : state === "verified"
          ? "Chain intact · verified just now"
          : "Not verified yet this session",
  );
  const sub = $derived(
    busy
      ? `Re-reading ${count} entries and matching each fingerprint to the one after it.`
      : tampered
        ? `Entry #${firstBrokenSeq ?? "?"} no longer matches the fingerprint recorded in the entry that follows it. The record is append-only — this should never happen. Treat everything from that point with suspicion.`
        : state === "verified"
          ? `All ${count} entries link cleanly, oldest to newest. Change any one and this check fails.`
          : `Tap Verify to re-read every entry and confirm the chain is intact, oldest to newest.`,
  );
  const buttonLabel = $derived(busy ? "Verifying…" : "Verify");
</script>

<div class="verify" class:tampered class:busy>
  <span class="chain" aria-hidden="true">
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <rect x="1.8" y="5.2" width="7.4" height="5.6" rx="2.8" stroke="currentColor" stroke-width="1.3" />
      <rect x="6.8" y="5.2" width="7.4" height="5.6" rx="2.8" stroke="currentColor" stroke-width="1.3" />
    </svg>
  </span>
  <div class="body">
    <div class="headline">
      {#if state === "verified"}<span class="ok-dot" aria-hidden="true"></span>{/if}
      {headline}
    </div>
    <div class="sub">{sub}</div>
    {#if busy}
      <div class="bar-track"><div class="bar"></div></div>
    {/if}
  </div>
  <button type="button" class="verify-btn" onclick={() => onVerify?.()} disabled={busy}>
    {buttonLabel}
  </button>
</div>

<style>
  .verify {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    border: 1px solid var(--color-border-strong);
    border-left: 3px solid var(--color-success);
    border-radius: var(--radius-lg);
    background: var(--color-bg-raised);
    padding: 16px 18px;
    color: var(--color-success);
  }
  .verify.busy {
    border-left-color: var(--color-accent);
    color: var(--color-accent);
  }
  .verify.tampered {
    border-color: var(--color-danger);
    border-left-color: var(--color-danger);
    background: var(--color-accent-soft);
    color: var(--color-danger);
    box-shadow: var(--shadow-md);
  }
  .chain {
    margin-top: 1px;
    flex-shrink: 0;
    display: flex;
  }
  .verify.busy .chain {
    animation: verify-pulse var(--motion-pulse) ease-in-out infinite;
  }
  .body {
    flex: 1;
    min-width: 0;
  }
  .headline {
    display: flex;
    align-items: center;
    gap: 9px;
    font-family: var(--font-sans);
    font-size: 14.5px;
    letter-spacing: -0.005em;
    color: var(--color-fg-primary);
  }
  .verify.tampered .headline {
    color: var(--color-danger);
    font-weight: 500;
  }
  .ok-dot {
    width: 6px;
    height: 6px;
    border-radius: var(--radius-pill);
    background: var(--color-success);
    flex-shrink: 0;
  }
  .sub {
    font-family: var(--font-sans);
    font-size: 12.5px;
    color: var(--color-fg-muted);
    line-height: 1.55;
    margin-top: 6px;
    max-width: 540px;
  }
  .verify.tampered .sub {
    color: var(--color-fg-secondary);
  }
  .bar-track {
    margin-top: 12px;
    height: 3px;
    border-radius: 3px;
    background: var(--color-border);
    overflow: hidden;
  }
  .bar {
    height: 100%;
    width: 30%;
    background: var(--color-accent);
    border-radius: 3px;
    animation: verify-bar 1500ms var(--ease) forwards;
  }
  .verify-btn {
    appearance: none;
    flex-shrink: 0;
    cursor: pointer;
    font-family: var(--font-sans);
    font-size: 13px;
    white-space: nowrap;
    padding: 8px 15px;
    border-radius: 9px;
    border: 1px solid var(--color-border-strong);
    background: transparent;
    color: var(--color-fg-secondary);
  }
  .verify-btn:hover:not(:disabled) {
    border-color: var(--color-fg-muted);
  }
  .verify.tampered .verify-btn {
    border-color: var(--color-danger);
    background: var(--color-danger);
    color: #fff;
  }
  .verify-btn:disabled {
    opacity: 0.6;
    cursor: default;
  }
  @keyframes verify-pulse {
    0%,
    100% {
      opacity: 0.45;
    }
    50% {
      opacity: 1;
    }
  }
  @keyframes verify-bar {
    from {
      width: 4%;
    }
    to {
      width: 100%;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .verify.busy .chain {
      animation: none;
    }
    .bar {
      animation: none;
      width: 100%;
    }
  }
</style>
