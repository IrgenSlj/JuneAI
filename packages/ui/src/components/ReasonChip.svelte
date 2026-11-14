<script lang="ts">
  /**
   * ReasonChip — a quiet, truthful token explaining a Silence decision at a
   * glance (ADR 0023). Deliberately calm: same flat token everywhere, never
   * colored for alarm. `tone="faint"` dims it for suppressed/low-salience items.
   */
  interface Props {
    /** The deciding-feature text, e.g. "you were mid-task" or "low salience". */
    text: string;
    tone?: "muted" | "faint";
    title?: string;
  }
  const { text, tone = "muted", title }: Props = $props();
</script>

<span class="chip" class:faint={tone === "faint"} {title}>
  <span class="dot" aria-hidden="true"></span>{text}
</span>

<style>
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.01em;
    line-height: 1.2;
    white-space: nowrap;
    color: var(--color-fg-muted);
    background: var(--color-bg-sunken);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-pill);
    padding: 3px 9px 3px 8px;
  }
  .chip.faint {
    color: var(--color-fg-subtle);
  }
  .dot {
    width: 3px;
    height: 3px;
    border-radius: 3px;
    background: currentColor;
    opacity: 0.65;
  }
</style>
