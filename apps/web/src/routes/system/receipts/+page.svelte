<script lang="ts">
  import { onMount } from "svelte";
  import {
    OfflineNotice,
    VerifyAffordance,
    type LedgerEntryView,
    type LedgerVerifyResponse,
    type SystemStatus,
  } from "@june/ui";
  import { client } from "$lib/api.js";
  import { formatRelative } from "$lib/dates.js";

  type VerifyState = "unknown" | "verified" | "verifying" | "tampered";
  type KindFilter = "all" | "egress" | "action" | "approval";

  let entries: LedgerEntryView[] = $state([]);
  let nextCursor: number | null = $state(null);
  let count = $state(0);
  let verifyState: VerifyState = $state("unknown");
  let firstBrokenSeq: number | null = $state(null);
  let filter: KindFilter = $state("all");
  let loading = $state(true);
  let loadingMore = $state(false);
  let loadError: string | null = $state(null);

  const KINDS: [KindFilter, string][] = [
    ["all", "Everything"],
    ["egress", "Egress"],
    ["action", "Actions"],
    ["approval", "Approvals"],
  ];

  async function load() {
    loading = true;
    loadError = null;
    try {
      const [page, sys] = await Promise.all([
        client.getLedger(null, 50),
        client.getSystem().catch(() => null as SystemStatus | null),
      ]);
      entries = page.entries ?? [];
      nextCursor = page.next_cursor ?? null;
      const summary = sys?.ledger_summary ?? null;
      count = summary?.count ?? entries.length;
      // Seed the verify claim from the cached last-verification (honest: null
      // until the user verifies this session).
      if (summary?.chain_verified === true) verifyState = "verified";
      else if (summary?.chain_verified === false) verifyState = "tampered";
      else verifyState = "unknown";
    } catch (err) {
      loadError = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  async function loadMore() {
    if (loadingMore || nextCursor == null) return;
    loadingMore = true;
    try {
      const page = await client.getLedger(nextCursor, 50);
      entries = [...entries, ...(page.entries ?? [])];
      nextCursor = page.next_cursor ?? null;
    } catch (err) {
      loadError = err instanceof Error ? err.message : String(err);
    } finally {
      loadingMore = false;
    }
  }

  async function runVerify() {
    verifyState = "verifying";
    try {
      const res: LedgerVerifyResponse = await client.verifyLedger();
      firstBrokenSeq = res.first_broken_seq ?? null;
      verifyState = res.ok ? "verified" : "tampered";
    } catch (err) {
      loadError = err instanceof Error ? err.message : String(err);
      verifyState = "unknown";
    }
  }

  onMount(load);

  const shown = $derived(
    filter === "all" ? entries : entries.filter((e) => e.kind === filter),
  );

  // ── derive human fields from a redacted ledger payload (no raw content) ──
  function payloadStr(p: Record<string, unknown> | undefined, key: string): string {
    const v = p?.[key];
    return v == null ? "" : String(v);
  }

  function isCloud(e: LedgerEntryView): boolean {
    return e.kind === "egress";
  }

  function title(e: LedgerEntryView): string {
    const p = (e.payload ?? {}) as Record<string, unknown>;
    if (e.kind === "egress") {
      const model = payloadStr(p, "model_id");
      return model ? `Cloud call · ${model}` : "Cloud call";
    }
    if (e.kind === "action") {
      if (payloadStr(p, "event") === "surfacing_decision") {
        return `Silence decision · ${payloadStr(p, "candidate_kind") || "candidate"}`;
      }
      return `Ran ${payloadStr(p, "tool") || "an action"}`;
    }
    if (e.kind === "approval") {
      const tool = payloadStr(p, "tool");
      const decision = payloadStr(p, "decision");
      return `${decision || "approval"}${tool ? ` · ${tool}` : ""}`;
    }
    return "System event";
  }

  function leftText(e: LedgerEntryView): string {
    const p = (e.payload ?? {}) as Record<string, unknown>;
    if (e.kind === "egress") {
      const summary = payloadStr(p, "summary");
      return summary || "A model request left the device.";
    }
    if (e.kind === "action") {
      if (payloadStr(p, "event") === "surfacing_decision") {
        return payloadStr(p, "reason") || "A surface-vs-defer decision, recorded locally.";
      }
      const cls = payloadStr(p, "action_class");
      const tainted = p["tainted"] === true ? " (used a prior tool result)" : "";
      return `Nothing left the device — a ${cls || "local"} action${tainted}, written to june.db.`;
    }
    if (e.kind === "approval") {
      return payloadStr(p, "reason") || "A decision you made, recorded locally.";
    }
    return "Recorded locally.";
  }

  function metaPairs(e: LedgerEntryView): [string, string][] {
    const p = (e.payload ?? {}) as Record<string, unknown>;
    const pairs: [string, string][] = [["actor", e.actor]];
    if (e.kind === "egress") {
      if (payloadStr(p, "model_id")) pairs.push(["model", payloadStr(p, "model_id")]);
      if (payloadStr(p, "privacy_dial")) pairs.push(["privacy", payloadStr(p, "privacy_dial")]);
    } else if (e.kind === "action") {
      if (payloadStr(p, "action_class")) pairs.push(["class", payloadStr(p, "action_class")]);
      if (payloadStr(p, "tool")) pairs.push(["tool", payloadStr(p, "tool")]);
    } else if (e.kind === "approval") {
      if (payloadStr(p, "surface")) pairs.push(["surface", payloadStr(p, "surface")]);
      if (payloadStr(p, "tool")) pairs.push(["tool", payloadStr(p, "tool")]);
    }
    return pairs;
  }

  const seqLabel = (seq: number) => `#${String(seq).padStart(4, "0")}`;
  const fp = (hash: string) => (hash ? hash.slice(0, 4) : "");
</script>

<svelte:head><title>Receipts — June</title></svelte:head>

<main class="page" id="main-content">
  <header class="top">
    <a class="back" href="/system">← Trust</a>
    <div class="heading">
      <h1>Receipts</h1>
      <p class="lede">
        Every time data left this device, and every consequential thing June did — written
        down the moment it happened, in an order that can't be quietly rewritten. Read it
        like a statement; check it like a skeptic.
      </p>
    </div>
  </header>

  {#if loadError && entries.length === 0}
    <OfflineNotice kind="system" detail={loadError} onRetry={load} retrying={loading} />
  {/if}

  <VerifyAffordance state={verifyState} {count} {firstBrokenSeq} onVerify={runVerify} />

  <div class="filter" role="tablist" aria-label="Filter receipts by kind">
    {#each KINDS as [value, label] (value)}
      <button
        type="button"
        role="tab"
        aria-selected={filter === value}
        class:on={filter === value}
        onclick={() => (filter = value)}
      >{label}</button>
    {/each}
  </div>

  {#if loading && entries.length === 0}
    <p class="hint">Reading the ledger…</p>
  {:else if shown.length === 0}
    <p class="hint">
      {filter === "all"
        ? "The ledger is empty. In local-only mode there are no egress rows at all — most days nothing leaves."
        : `No ${filter} entries recorded yet.`}
    </p>
  {:else}
    <ul class="receipts">
      {#each shown as e (e.seq)}
        {@const broke = verifyState === "tampered" && firstBrokenSeq === e.seq}
        <li class="receipt" class:cloud={isCloud(e)} class:broke>
          <div class="receipt-body">
            <div class="receipt-top">
              <div class="receipt-what">
                <span class="kind-tag">{e.kind}</span>
                <span class="what">{title(e)}</span>
              </div>
              <time class="when" datetime={e.ts}>{formatRelative(e.ts)}</time>
            </div>

            <div class="left" class:cloud={isCloud(e)}>
              <div class="left-head" class:cloud={isCloud(e)} class:broke>
                <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path d="M8 1.6l5 1.9v4.1c0 3-2.1 5-5 6.8-2.9-1.8-5-3.8-5-6.8V3.5L8 1.6z"
                    stroke="currentColor" stroke-width="1.3" stroke-linejoin="round" />
                </svg>
                <span>{isCloud(e) ? "left the device" : "stayed local"}</span>
              </div>
              <p class="left-text">{leftText(e)}</p>
            </div>

            <dl class="meta">
              {#each metaPairs(e) as [k, v] (k)}
                <div class="meta-row"><dt>{k}</dt><dd>{v}</dd></div>
              {/each}
            </dl>
          </div>

          <div class="chain" class:broke>
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <rect x="1.8" y="5.2" width="7.4" height="5.6" rx="2.8" stroke="currentColor" stroke-width="1.3" />
              <rect x="6.8" y="5.2" width="7.4" height="5.6" rx="2.8" stroke="currentColor" stroke-width="1.3" />
            </svg>
            <span>entry {seqLabel(e.seq)}</span>
            <span class="dim">·</span>
            <span>fingerprint {fp(e.entry_hash)}…</span>
            {#if e.sig}<span class="dim">·</span><span>signed</span>{/if}
            <span class="spacer"></span>
            {#if broke}
              <span class="break-note">chain breaks here — changed after writing</span>
            {:else}
              <span>{e.seq === 1 ? "genesis" : `linked to ${seqLabel(e.seq - 1)}`}</span>
            {/if}
          </div>
        </li>
      {/each}
    </ul>

    {#if nextCursor != null}
      <button type="button" class="load-more" onclick={loadMore} disabled={loadingMore}>
        {loadingMore ? "Loading…" : "Load older"}
      </button>
    {/if}
  {/if}
</main>

<style>
  .page {
    max-width: 800px;
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
    max-width: 60ch;
  }
  .hint {
    color: var(--color-fg-subtle);
    font-size: var(--size-sm);
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

  .receipts {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .receipt {
    border: 1px solid var(--color-border);
    border-left: 3px solid transparent;
    border-radius: var(--radius-lg);
    background: var(--color-bg-raised);
    overflow: hidden;
  }
  .receipt.cloud {
    border-left-color: var(--color-warn);
  }
  .receipt.broke {
    border-color: var(--color-danger);
    border-left-color: var(--color-danger);
    box-shadow: var(--shadow-md);
  }
  .receipt-body {
    padding: 16px 18px 14px;
  }
  .receipt-top {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 14px;
  }
  .receipt-what {
    display: flex;
    align-items: baseline;
    gap: 10px;
    min-width: 0;
  }
  .kind-tag {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--color-fg-muted);
    background: var(--color-bg-sunken);
    border: 1px solid var(--color-border);
    border-radius: 6px;
    padding: 2px 7px;
    flex-shrink: 0;
  }
  .what {
    font-family: var(--font-sans);
    font-size: 15.5px;
    color: var(--color-fg-primary);
    font-weight: 500;
    letter-spacing: -0.005em;
    overflow-wrap: anywhere;
  }
  .when {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--color-fg-subtle);
    white-space: nowrap;
    flex-shrink: 0;
  }

  .left {
    margin-top: 13px;
    border-radius: var(--radius-md);
    background: var(--color-bg-sunken);
    border: 1px solid var(--color-border);
    padding: 11px 14px;
  }
  .left.cloud {
    background: var(--color-accent-soft);
    border-color: color-mix(in srgb, var(--color-warn) 33%, transparent);
  }
  .left-head {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--color-success);
    font-family: var(--font-sans);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .left-head.cloud {
    color: var(--color-warn);
  }
  .left-head.broke {
    color: var(--color-danger);
  }
  .left-text {
    margin: 6px 0 0;
    font-family: var(--font-sans);
    font-size: 13.5px;
    color: var(--color-fg-primary);
    line-height: 1.5;
    overflow-wrap: anywhere;
  }

  .meta {
    margin: 13px 0 0;
    display: grid;
    grid-template-columns: 84px 1fr;
    column-gap: 14px;
    row-gap: 7px;
  }
  .meta-row {
    display: contents;
  }
  .meta dt {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--color-fg-subtle);
  }
  .meta dd {
    margin: 0;
    font-family: var(--font-sans);
    font-size: 13px;
    color: var(--color-fg-secondary);
    overflow-wrap: anywhere;
  }

  .chain {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 9px 18px;
    border-top: 1px solid var(--color-border);
    background: var(--color-bg-sunken);
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--color-fg-subtle);
  }
  .chain svg {
    color: var(--color-success);
  }
  .chain.broke {
    color: var(--color-danger);
    background: var(--color-accent-soft);
    border-top-color: color-mix(in srgb, var(--color-danger) 30%, transparent);
  }
  .chain.broke svg {
    color: var(--color-danger);
  }
  .chain .dim {
    opacity: 0.5;
  }
  .chain .spacer {
    flex: 1;
  }
  .break-note {
    color: var(--color-danger);
    font-weight: 500;
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

  @media (max-width: 640px) {
    .page {
      padding-inline: var(--space-3);
    }
    .meta {
      grid-template-columns: 1fr;
      row-gap: 2px;
    }
    .meta-row {
      display: flex;
      gap: var(--space-2);
      align-items: baseline;
    }
  }
</style>
