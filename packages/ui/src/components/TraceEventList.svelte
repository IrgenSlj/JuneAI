<script lang="ts">
  import type { TraceEventView } from "../api/client.js";

  interface Props {
    events: TraceEventView[];
  }

  const { events }: Props = $props();

  // Set of seq indices whose detail is expanded.
  let expanded = $state<Set<number>>(new Set());

  function toggle(seq: number): void {
    const next = new Set(expanded);
    if (next.has(seq)) next.delete(seq);
    else next.add(seq);
    expanded = next;
  }

  function formatTime(epochSeconds: number): string {
    const d = new Date(epochSeconds * 1000);
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
  }

  function lineCount(detail: string): number {
    if (!detail) return 0;
    return detail.split("\n").length;
  }

  // Provenance rows are the cloud/local trust anchor. TraceEventView carries no
  // typed `cloud` field, so we infer it from the writer's summary text, which
  // includes the word "cloud" for cloud turns (see brain loop/trace.py). If that
  // summary contract changes, update this together with it.
  function isCloudBoundary(event: TraceEventView): boolean {
    return event.kind === "provenance" && event.summary.includes("cloud");
  }
</script>

<div class="trace-list" role="log" aria-label="Turn trace events">
  {#each events as event (event.seq)}
    {@const hasDetail = !!event.detail}
    {@const isOpen = expanded.has(event.seq)}
    {@const isBoundary = event.kind === "provenance"}
    {@const isCloud = isCloudBoundary(event)}
    <div
      class="row"
      class:boundary={isBoundary}
      class:cloud={isCloud}
      class:local={isBoundary && !isCloud}
      data-kind={event.kind}
    >
      <button
        type="button"
        class="line"
        class:clickable={hasDetail}
        disabled={!hasDetail}
        aria-expanded={hasDetail ? isOpen : undefined}
        onclick={() => hasDetail && toggle(event.seq)}
      >
        <span class="ts">{formatTime(event.ts)}</span>
        <span class="kind">{event.kind}</span>
        {#if isBoundary}
          <span class="boundary-tag">{isCloud ? "cloud" : "local"}</span>
        {/if}
        <span class="label">{event.summary || event.kind}</span>
        {#if hasDetail}
          <span class="expand">
            {#if isOpen}
              collapse
            {:else}
              +{lineCount(event.detail)} line{lineCount(event.detail) === 1 ? "" : "s"}
            {/if}
          </span>
        {/if}
      </button>
      {#if hasDetail && isOpen}
        <pre
          class="detail"
          class:reasoning={event.kind === "reasoning"}>{event.detail}</pre>
      {/if}
    </div>
  {/each}
  {#if events.length === 0}
    <p class="empty">No events in this trace.</p>
  {/if}
</div>

<style>
  .trace-list {
    display: flex;
    flex-direction: column;
    gap: 2px;
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    line-height: 1.5;
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

  .kind {
    flex-shrink: 0;
    color: var(--color-fg-subtle);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 10px;
    min-width: 7em;
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
  .row[data-kind="tool_call"] .label,
  .row[data-kind="tool_result"] .label {
    color: var(--color-fg-muted);
  }
  .row[data-kind="model_call"] .label {
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

  /* Boundary line — cloud/local trust anchor. */
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

  .empty {
    margin: 0;
    color: var(--color-fg-subtle);
    font-family: var(--font-mono);
    font-size: var(--size-xs);
  }
</style>
