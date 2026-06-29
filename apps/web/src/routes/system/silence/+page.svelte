<script lang="ts">
  import { onMount } from "svelte";
  import {
    OfflineNotice,
    ReasonChip,
    type SurfacingDecisionView,
  } from "@june/ui";
  import { client } from "$lib/api.js";
  import { formatRelative } from "$lib/dates.js";

  type Filter = "all" | "surfaced" | "batched" | "suppressed";

  let decisions: SurfacingDecisionView[] = $state([]);
  let nextCursor: string | null = $state(null);
  let filter: Filter = $state("all");
  let loading = $state(true);
  let loadingMore = $state(false);
  let loadError: string | null = $state(null);
  // Per-decision feedback result, keyed by id.
  let feedback: Record<string, "good" | "bad"> = $state({});
  let feedbackErr: string | null = $state(null);

  const FILTERS: [Filter, string][] = [
    ["all", "All"],
    ["surfaced", "Surfaced"],
    ["batched", "Batched"],
    ["suppressed", "Suppressed"],
  ];

  const STATE_META = {
    surfaced: { label: "surfaced now", blurb: "June chose to interrupt. Rare, and earned." },
    batched: { label: "batched", blurb: "Held calmly for the next time you look. The norm." },
    suppressed: { label: "suppressed", blurb: "Not shown to you. Visible here only, never pushed." },
  } as const;

  function stateOf(action: string): keyof typeof STATE_META {
    if (action === "now") return "surfaced";
    if (action === "suppress") return "suppressed";
    return "batched";
  }

  async function load() {
    loading = true;
    loadError = null;
    try {
      const page = await client.getSurfacing(null, 50);
      decisions = page.entries ?? [];
      nextCursor = page.next_cursor ?? null;
    } catch (err) {
      loadError = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  async function loadMore() {
    if (loadingMore || !nextCursor) return;
    loadingMore = true;
    try {
      const page = await client.getSurfacing(nextCursor, 50);
      decisions = [...decisions, ...(page.entries ?? [])];
      nextCursor = page.next_cursor ?? null;
    } catch (err) {
      loadError = err instanceof Error ? err.message : String(err);
    } finally {
      loadingMore = false;
    }
  }

  async function sendFeedback(id: string, verdict: "good" | "bad") {
    feedbackErr = null;
    try {
      await client.sendSurfacingFeedback(id, verdict);
      feedback = { ...feedback, [id]: verdict };
    } catch (err) {
      feedbackErr = err instanceof Error ? err.message : String(err);
    }
  }

  onMount(load);

  function humanizeKind(kind: string): string {
    const map: Record<string, string> = {
      deadline: "Deadline",
      contradiction: "Contradiction",
      promise_nudge: "Promise nudge",
    };
    return map[kind] ?? kind.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
  }

  // Derive a small set of truthful feature chips for a decision.
  function chips(d: SurfacingDecisionView): string[] {
    const f = (d.features ?? {}) as Record<string, unknown>;
    const out: string[] = [];
    const presence = f["presence_state"];
    if (typeof presence === "string" && presence) out.push(presence);
    const delta = f["deadline_delta_hours"];
    if (typeof delta === "number") {
      out.push(delta < 0 ? "deadline passed" : `deadline in ${Math.round(delta)}h`);
    }
    const dismissed = f["dismissals_for_similar"];
    if (typeof dismissed === "number" && dismissed >= 1) {
      out.push(`dismissed ${dismissed}×`);
    }
    return out.slice(0, 3);
  }

  const shown = $derived(
    filter === "all" ? decisions : decisions.filter((d) => stateOf(d.action) === filter),
  );
  const held = $derived(
    decisions.filter((d) => d.action === "batch" && !d.surfaced_at),
  );
</script>

<svelte:head><title>Silence — June</title></svelte:head>

<main class="page" id="main-content">
  <header class="top">
    <a class="back" href="/system">← Trust</a>
    <div class="heading">
      <h1>Silence</h1>
      <p class="lede">
        When June stays quiet, it's a choice — and you can see every one of them. Here's what
        she held for you, and the reasoning behind what she surfaced, batched, or kept back
        entirely.
      </p>
    </div>
  </header>

  {#if loadError && decisions.length === 0}
    <OfflineNotice kind="system" detail={loadError} onRetry={load} retrying={loading} />
  {/if}

  {#if feedbackErr}
    <p class="hint err">{feedbackErr}</p>
  {/if}

  <!-- Held digest — batched, not yet surfaced -->
  {#if (filter === "all" || filter === "batched") && held.length > 0}
    <section class="layer">
      <div class="layer-head">
        <span class="layer-label">Held for you</span>
        <span class="layer-count">{held.length} thing{held.length === 1 ? "" : "s"} · no rush</span>
      </div>
      <div class="rows">
        {#each held as d (d.id)}
          <div class="row">
            <div class="row-main">
              <span class="row-title">{humanizeKind(d.kind)}</span>
              <span class="row-line">{d.reason}</span>
            </div>
            <ReasonChip text={d.reason} />
          </div>
        {/each}
      </div>
    </section>
  {/if}

  <!-- State legend -->
  <div class="legend">
    {#each Object.entries(STATE_META) as [key, meta] (key)}
      <div class="legend-card">
        <div class="legend-head">
          <span class="dot {key}" aria-hidden="true"></span>
          <span class="legend-label">{meta.label}</span>
        </div>
        <p class="legend-blurb">{meta.blurb}</p>
      </div>
    {/each}
  </div>

  <div class="layer-head">
    <span class="layer-label">Decision inspector</span>
    <span class="layer-count">three ways a thing can land</span>
  </div>

  <div class="filter" role="tablist" aria-label="Filter decisions">
    {#each FILTERS as [value, label] (value)}
      <button
        type="button"
        role="tab"
        aria-selected={filter === value}
        class:on={filter === value}
        onclick={() => (filter = value)}
      >{label}</button>
    {/each}
  </div>

  {#if loading && decisions.length === 0}
    <p class="hint">Reading recent decisions…</p>
  {:else if shown.length === 0}
    <p class="hint">No decisions recorded yet. When June chooses to surface, batch, or stay quiet, each choice lands here with its reason.</p>
  {:else}
    <ul class="decisions">
      {#each shown as d (d.id)}
        {@const st = stateOf(d.action)}
        <li class="decision {st}">
          <div class="decision-top">
            <span class="decision-title">{humanizeKind(d.kind)}</span>
            <div class="decision-meta">
              <span class="state-badge">
                <span class="dot {st}" aria-hidden="true"></span>{STATE_META[st].label}
              </span>
              <time class="when" datetime={d.ts}>{formatRelative(d.ts)}</time>
            </div>
          </div>
          <p class="decision-line">{d.reason}</p>
          <div class="decision-foot">
            <div class="chips">
              {#each chips(d) as c (c)}
                <ReasonChip text={c} tone={st === "suppressed" ? "faint" : "muted"} />
              {/each}
            </div>
            <div class="feedback">
              {#if feedback[d.id]}
                <span class="noted">
                  {feedback[d.id] === "good" ? "noted — thanks" : "noted — I'll weigh this differently"}
                </span>
              {:else}
                <span class="feedback-q">was this the right call?</span>
                <button type="button" class="fb" onclick={() => sendFeedback(d.id, "good")}>yes</button>
                <button type="button" class="fb" onclick={() => sendFeedback(d.id, "bad")}>
                  {st === "suppressed" ? "I'd have wanted this" : "too much"}
                </button>
              {/if}
            </div>
          </div>
        </li>
      {/each}
    </ul>

    {#if nextCursor}
      <button type="button" class="load-more" onclick={loadMore} disabled={loadingMore}>
        {loadingMore ? "Loading…" : "Load older"}
      </button>
    {/if}
  {/if}

  <p class="footnote">
    The suppressed ones are here on purpose. A companion you can't audit is just a louder app —
    you should be able to see what June decided <em>not</em> to show you.
  </p>
</main>

<style>
  .page {
    max-width: 820px;
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
  .heading h1 {
    margin: 0 0 var(--space-1);
    font-size: var(--size-xl);
    font-weight: 400;
    letter-spacing: -0.01em;
  }
  .lede {
    margin: 0;
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
    line-height: var(--leading-relaxed);
    max-width: 62ch;
  }
  .hint {
    color: var(--color-fg-subtle);
    font-size: var(--size-sm);
  }
  .hint.err {
    color: var(--color-danger);
  }

  .layer-head {
    display: flex;
    align-items: baseline;
    gap: 10px;
  }
  .layer-label {
    font-family: var(--font-sans);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--color-fg-muted);
  }
  .layer-count {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--color-fg-subtle);
  }
  .layer {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .rows,
  .decisions {
    list-style: none;
    margin: 0;
    padding: 0;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }
  .row,
  .decision {
    background: var(--color-bg-raised);
    border-bottom: 1px solid var(--color-border);
  }
  .row:last-child,
  .decision:last-child {
    border-bottom: none;
  }

  .row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
    padding: 14px 18px;
  }
  .row-main {
    min-width: 0;
  }
  .row-title {
    font-family: var(--font-sans);
    font-size: 14.5px;
    color: var(--color-fg-primary);
  }
  .row-line {
    display: block;
    font-family: var(--font-sans);
    font-size: 13px;
    color: var(--color-fg-muted);
    line-height: 1.5;
    margin-top: 4px;
  }

  .legend {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: var(--space-3);
  }
  .legend-card {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-bg-raised);
    padding: 13px 15px;
  }
  .legend-head {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .legend-label {
    font-family: var(--font-sans);
    font-size: 13.5px;
    color: var(--color-fg-primary);
    font-weight: 500;
  }
  .legend-blurb {
    margin: 6px 0 0;
    font-family: var(--font-sans);
    font-size: 12px;
    color: var(--color-fg-muted);
    line-height: 1.5;
  }
  @media (max-width: 640px) {
    .legend {
      grid-template-columns: 1fr;
    }
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: var(--radius-pill);
    flex-shrink: 0;
  }
  .dot.surfaced {
    background: var(--color-accent);
  }
  .dot.batched {
    background: var(--color-success);
  }
  .dot.suppressed {
    background: var(--color-fg-subtle);
  }

  .filter {
    display: flex;
    gap: 2px;
    background: var(--color-bg-sunken);
    padding: 3px;
    border-radius: 9px;
    width: fit-content;
  }
  .filter button {
    appearance: none;
    border: none;
    cursor: pointer;
    padding: 5px 13px;
    border-radius: 7px;
    font-family: var(--font-sans);
    font-size: 12.5px;
    background: transparent;
    color: var(--color-fg-muted);
  }
  .filter button.on {
    background: var(--color-bg-raised);
    color: var(--color-fg-primary);
    box-shadow: var(--shadow-sm);
  }

  .decision {
    padding: 17px 18px;
    border-left: 3px solid transparent;
  }
  .decision.surfaced {
    border-left-color: var(--color-accent);
  }
  .decision.suppressed {
    opacity: 0.72;
  }
  .decision-top {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 14px;
  }
  .decision-title {
    font-family: var(--font-sans);
    font-size: 15px;
    color: var(--color-fg-primary);
    font-weight: 500;
  }
  .decision-meta {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
  }
  .state-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: var(--font-sans);
    font-size: 12px;
    color: var(--color-fg-muted);
  }
  .when {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--color-fg-subtle);
  }
  .decision-line {
    margin: 7px 0 0;
    font-family: var(--font-sans);
    font-size: 13.5px;
    color: var(--color-fg-secondary);
    line-height: 1.6;
    max-width: 64ch;
  }
  .decision-foot {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 13px;
    flex-wrap: wrap;
  }
  .chips {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .feedback {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .feedback-q {
    font-family: var(--font-sans);
    font-size: 11.5px;
    color: var(--color-fg-subtle);
  }
  .fb {
    appearance: none;
    cursor: pointer;
    border: 1px solid var(--color-border);
    background: transparent;
    color: var(--color-fg-muted);
    font-family: var(--font-sans);
    font-size: 11.5px;
    padding: 3px 10px;
    border-radius: 7px;
  }
  .fb:hover {
    border-color: var(--color-border-strong);
    color: var(--color-fg-primary);
  }
  .noted {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--color-fg-subtle);
  }

  .load-more {
    align-self: flex-start;
    appearance: none;
    cursor: pointer;
    border: 1px solid var(--color-border);
    background: transparent;
    color: var(--color-fg-muted);
    border-radius: var(--radius-md);
    padding: var(--space-2) var(--space-4);
    font: inherit;
    font-size: var(--size-sm);
  }
  .load-more:hover:not(:disabled) {
    border-color: var(--color-border-strong);
    color: var(--color-fg-primary);
  }
  .load-more:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .footnote {
    margin: var(--space-2) 0 0;
    font-family: var(--font-sans);
    font-size: 13px;
    color: var(--color-fg-muted);
    line-height: 1.65;
  }
  .footnote em {
    color: var(--color-fg-secondary);
    font-style: italic;
  }

  @media (max-width: 640px) {
    .page {
      padding-inline: var(--space-3);
    }
    .feedback {
      margin-left: 0;
    }
  }
</style>
