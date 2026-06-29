<script lang="ts">
  import { onMount } from "svelte";
  import { TraceEventList, type TraceEventView } from "@june/ui";
  import { glass, loadTurns, loadTurn } from "$lib/stores/glass.svelte.js";

  onMount(() => {
    void loadTurns();
  });

  // Track which turn_ids are expanded (open).
  let open = $state<Set<string>>(new Set());

  async function toggle(turn_id: string): Promise<void> {
    const next = new Set(open);
    if (next.has(turn_id)) {
      next.delete(turn_id);
    } else {
      next.add(turn_id);
      await loadTurn(turn_id);
    }
    open = next;
  }

  function formatTime(epochSeconds: number): string {
    const d = new Date(epochSeconds * 1000);
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
  }

  function shortId(turn_id: string): string {
    return turn_id.slice(-8);
  }

  // Derive a title from loaded events: summary of first iteration or prompt.
  function turnTitle(events: TraceEventView[] | null): string | null {
    if (!events) return null;
    const found = events.find((e) => e.kind === "iteration" || e.kind === "prompt");
    return found?.summary || null;
  }
</script>

<svelte:head>
  <title>Glass Box — June</title>
</svelte:head>

<main class="page" id="main-content">
  <header class="top">
    <a class="back" href="/system">← Trust</a>
    <div class="heading">
      <div class="title-row">
        <h1>Glass Box</h1>
        <span class="live-dot" aria-hidden="true" title="Idle — live streaming in GB-2"></span>
      </div>
      <p class="lede">
        Every step June takes — route, recall, prompt, model calls, tools, and what left the device.
      </p>
    </div>
  </header>

  {#if glass.error}
    <p class="inline-error">Could not load traces: {glass.error}</p>
  {:else if glass.loading && glass.turns.length === 0}
    <p class="hint">Loading turns…</p>
  {:else if glass.turns.length === 0}
    <p class="hint empty-state">
      No turns recorded yet. Chat with June and every step shows up here.
    </p>
  {:else}
    <ul class="timeline">
      {#each glass.turns as turn (turn.turn_id)}
        {@const isOpen = open.has(turn.turn_id)}
        {@const title = isOpen ? turnTitle(turn.events) : null}
        <li class="turn" class:expanded={isOpen}>
          <button
            type="button"
            class="turn-header"
            onclick={() => toggle(turn.turn_id)}
            aria-expanded={isOpen}
            aria-controls={isOpen ? `turn-panel-${turn.turn_id}` : undefined}
          >
            <span class="turn-time">{formatTime(turn.started_at)}</span>
            <span class="turn-count">{turn.event_count} step{turn.event_count === 1 ? "" : "s"}</span>
            {#if title}
              <span class="turn-title">{title}</span>
            {:else}
              <span class="turn-id">{shortId(turn.turn_id)}</span>
            {/if}
            <span class="turn-chevron" aria-hidden="true">{isOpen ? "▲" : "▼"}</span>
          </button>

          {#if isOpen}
            <div class="turn-panel" id={`turn-panel-${turn.turn_id}`}>
              {#if turn.loading}
                <p class="hint panel-hint">Loading trace…</p>
              {:else if turn.error}
                <p class="inline-error panel-hint">{turn.error}</p>
              {:else}
                <TraceEventList events={turn.events ?? []} />
              {/if}
            </div>
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</main>

<style>
  .page {
    max-width: 860px;
    margin: 0 auto;
    padding: var(--space-5) var(--space-4) var(--space-7);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-width: 0;
  }

  .top {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    padding-bottom: var(--space-3);
    border-bottom: 1px solid var(--color-border);
  }

  .back {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
    text-decoration: none;
  }
  .back:hover {
    color: var(--color-fg-muted);
  }

  .heading {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }

  .title-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }

  .heading h1 {
    margin: 0;
    font-size: var(--size-xl);
    font-weight: 400;
    letter-spacing: -0.01em;
  }

  /* Static idle dot — GB-2 will animate it when a stream is live. */
  .live-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: var(--radius-pill);
    background: var(--color-fg-subtle);
    opacity: 0.4;
    flex-shrink: 0;
  }

  .lede {
    margin: 0;
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
    line-height: var(--leading-relaxed);
    max-width: 60ch;
  }

  .hint {
    color: var(--color-fg-subtle);
    font-size: var(--size-sm);
    font-family: var(--font-mono);
    margin: 0;
  }

  .empty-state {
    font-family: var(--font-sans);
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
  }

  .inline-error {
    color: var(--color-danger);
    font-size: var(--size-sm);
    font-family: var(--font-mono);
    margin: 0;
    padding: var(--space-2) var(--space-3);
    border: 1px solid color-mix(in srgb, var(--color-danger) 30%, transparent);
    border-radius: var(--radius-sm);
    background: color-mix(in srgb, var(--color-danger) 6%, transparent);
  }

  .timeline {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }

  .turn {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-bg);
    overflow: hidden;
  }

  .turn.expanded {
    border-color: color-mix(in srgb, var(--color-accent) 35%, var(--color-border));
  }

  .turn-header {
    display: grid;
    grid-template-columns: 6.5em auto minmax(0, 1fr) 1.2em;
    gap: var(--space-3);
    width: 100%;
    padding: var(--space-2) var(--space-3);
    border: none;
    background: transparent;
    text-align: left;
    font: inherit;
    cursor: pointer;
    align-items: baseline;
    font-family: var(--font-mono);
    font-size: var(--size-xs);
  }

  .turn-header:hover {
    background: color-mix(in srgb, var(--color-fg-muted) 5%, transparent);
  }

  .turn-time {
    color: var(--color-fg-primary);
    flex-shrink: 0;
    letter-spacing: 0.02em;
  }

  .turn-count {
    color: var(--color-fg-subtle);
    flex-shrink: 0;
    white-space: nowrap;
  }

  .turn-title {
    color: var(--color-fg-muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-family: var(--font-sans);
    font-size: var(--size-xs);
  }

  .turn-id {
    color: var(--color-fg-subtle);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    opacity: 0.7;
  }

  .turn-chevron {
    font-size: 9px;
    color: var(--color-fg-subtle);
    text-align: right;
    justify-self: end;
  }

  .turn-panel {
    padding: var(--space-3) var(--space-4);
    border-top: 1px solid var(--color-border);
    background: var(--color-term-bg, var(--color-bg));
  }

  .panel-hint {
    padding: 0;
  }

  @media (max-width: 640px) {
    .page {
      padding-inline: var(--space-3);
    }
    .turn-header {
      grid-template-columns: 5.5em auto minmax(0, 1fr) 1em;
      gap: var(--space-2);
    }
  }
</style>
