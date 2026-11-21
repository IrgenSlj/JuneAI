<script lang="ts">
  // The receipts log (PLAN.md M3): what June did on your behalf, newest first,
  // and a check of the hash chain. A receipt holds what, when and where to,
  // never the content itself.
  import { ApiError, type ReceiptInfo, type ReceiptsVerify, type ThreadSummary } from "@june/ui/api";
  import { onMount } from "svelte";
  import { session } from "#lib/session.svelte.js";

  let receipts: ReceiptInfo[] = $state([]);
  let titles = $state(new Map<string, string>());
  let check: ReceiptsVerify | undefined = $state();
  let checking = $state(false);
  let error: string | undefined = $state();

  onMount(async () => {
    try {
      const [all, threads] = await Promise.all([session.client.receipts(), session.client.threads({ limit: 1000 })]);
      receipts = all.toReversed();
      titles = new Map(threads.map((t: ThreadSummary) => [t.id, t.title]));
    } catch (e) {
      error = e instanceof ApiError ? e.reason : String(e);
    }
  });

  async function verify(): Promise<void> {
    checking = true;
    try {
      check = await session.client.verifyReceipts();
    } finally {
      checking = false;
    }
  }

  const KIND: Record<string, string> = { send: "Sent", import: "Imported", export: "Exported", "memory-read": "Read a note", "memory-write": "Wrote a note", "tool-call": "Tool call", "mcp-read": "Read by another assistant" };
  const when = (iso: string) => new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

  function describe(r: ReceiptInfo): string {
    const m = r.meta ?? {};
    if (r.kind === "send") return `${String(m.model ?? "")} via ${String(m.host ?? "")}${m.leavesDevice === false ? " (this Mac)" : ""}, ${Number(m.chars ?? 0).toLocaleString()} characters`;
    if (r.kind === "import") return `${Number(m.created ?? 0)} new, ${Number(m.updated ?? 0)} updated, ${Number(m.unchanged ?? 0)} unchanged`;
    if (r.kind === "export") return `as ${String(m.format ?? "")}, ${m.to === "file" ? "saved to a file" : "downloaded"}`;
    return "";
  }
  /** Targets name threads and turns ("th_...#m3"); show each thread once, by title. */
  function target(r: ReceiptInfo): string | undefined {
    const threads = [...new Set(r.targets.map((t) => t.split("#")[0] ?? t))];
    if (threads.length === 0) return undefined;
    const first = titles.get(threads[0] ?? "") ?? threads[0];
    return threads.length === 1 ? first : `${first} and ${threads.length - 1} more`;
  }
</script>

<main class="page">
  <h1>Receipts</h1>
  <p class="intro">A line for everything June did on your behalf. Each line includes a fingerprint of the one before it, so a removed or edited line shows up when you check.</p>
  <div class="check">
    <button type="button" class="primary" onclick={verify} disabled={checking}>Check the chain</button>
    {#if check}
      <span class:ok={check.ok} class:broken={!check.ok} role="status">
        {check.ok ? (check.count === 1 ? "The 1 receipt checks out." : `All ${check.count.toLocaleString()} receipts check out.`) : `Broken at ${check.brokenAt?.file} line ${check.brokenAt?.line}: ${check.brokenAt?.reason}`}
      </span>
    {/if}
  </div>
  {#if error}<p class="error" role="alert">{error}</p>{/if}
  {#if receipts.length === 0 && !error}
    <p class="intro">No receipts yet. Send a message, import or export, and it appears here.</p>
  {/if}
  <ol class="list" aria-label="Receipts">
    {#each receipts as r (r.seq)}
      <li>
        <span class="seq">#{r.seq}</span>
        <span class="kind">{KIND[r.kind] ?? r.kind}</span>
        <span class="what">
          {#if target(r)}<span class="target">{target(r)}</span>{/if}
          <span class="detail">{describe(r)}</span>
        </span>
        <time datetime={r.at}>{when(r.at)}</time>
      </li>
    {/each}
  </ol>
</main>

<style>
  .page {
    max-width: 48rem;
    width: 100%;
    margin: 0 auto;
    padding: 1.5rem 1.25rem 4rem;
    overflow-y: auto;
  }
  .intro {
    color: var(--color-fg-muted);
  }
  .check {
    display: flex;
    gap: var(--space-3);
    align-items: center;
    margin: var(--space-4) 0;
  }
  .ok {
    color: var(--color-success);
  }
  .broken {
    color: var(--color-danger);
  }
  .ok::before {
    content: "✓ ";
  }
  .broken::before {
    content: "✕ ";
  }
  .list {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  .list li {
    display: grid;
    grid-template-columns: 3.5rem 6.5rem 1fr auto;
    gap: var(--space-3);
    align-items: baseline;
    padding: var(--space-2) 0;
    border-bottom: 1px solid var(--color-border);
    font-size: var(--size-sm);
  }
  .seq,
  time {
    color: var(--color-fg-subtle);
    font-variant-numeric: tabular-nums;
  }
  .what {
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .target {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .detail {
    color: var(--color-fg-muted);
    font-size: var(--size-xs);
  }
  .error {
    border-left: 3px solid var(--color-danger);
    padding-left: var(--space-2);
  }
  @media (max-width: 40rem) {
    .list li {
      grid-template-columns: 3rem 1fr;
    }
    time {
      grid-column: 2;
    }
  }
</style>
