<script lang="ts">
  // The send label (PLAN.md, Transparency, Full level; design brief frame 3):
  // what is about to leave, where to, and under which data practice, before
  // it leaves. Built by the daemon from the exact request; nothing here is a
  // model's account of itself.
  import type { SendLabelInfo } from "@june/ui/api";
  import { getPlatform } from "@june/ui/platform";
  import { onMount } from "svelte";

  interface Props {
    label: SendLabelInfo;
    onSend: () => void;
    onCancel: () => void;
  }
  let { label, onSend, onCancel }: Props = $props();

  let sendButton: HTMLButtonElement | undefined = $state();
  onMount(() => sendButton?.focus());

  const n = (count: number) => count.toLocaleString();
  const turns = (count: number | undefined) => (count === undefined ? "" : `${n(count)} ${count === 1 ? "turn" : "turns"}, `);
  const PART_NAMES: Record<string, string> = {
    draft: "Your message",
    history: "Earlier turns in this conversation",
    imported: "Imported turns (from an export, marked as imported)",
    note: "A note from your vault",
    attachment: "An attachment",
    system: "Instructions",
  };
</script>

<svelte:window onkeydown={(e) => e.key === "Escape" && onCancel()} />

<div class="backdrop" role="presentation" onclick={onCancel}></div>
<div class="sheet" role="dialog" aria-modal="true" aria-labelledby="label-title">
  <h2 id="label-title">{label.route.leavesDevice ? `About to send to ${label.route.host}` : "About to send, staying on this Mac"}</h2>
  <dl>
    <dt>Route</dt>
    <dd>{label.route.name}</dd>
    <dt>Model</dt>
    <dd><code>{label.model}</code></dd>
    <dt>Paid by</dt>
    <dd>{label.route.billing}</dd>
    <dt>Their data practice</dt>
    <dd>
      {label.route.dataNote === "unknown" ? "Not yet checked by June." : label.route.dataNote}
      {#if label.route.dataNoteSource}
        <button type="button" class="link" onclick={() => label.route.dataNoteSource && void getPlatform().openExternal(label.route.dataNoteSource.url)}>
          Source, checked {label.route.dataNoteSource.checked}
        </button>
      {/if}
    </dd>
  </dl>
  <h3>What is included</h3>
  <ul class="parts">
    {#each label.parts as part, i (i)}
      <li class:imported={part.kind === "imported"}>
        <span>{PART_NAMES[part.kind] ?? part.kind}</span>
        <span class="size">{turns(part.turns)}{n(part.chars)} characters</span>
      </li>
    {/each}
  </ul>
  <p class="total">{n(label.totalChars)} characters in all. A receipt is written before anything leaves.</p>
  <div class="buttons">
    <button type="button" onclick={onCancel}>Cancel</button>
    <button type="button" class="primary" bind:this={sendButton} onclick={onSend}>Send</button>
  </div>
</div>

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.35);
    z-index: 10;
  }
  .sheet {
    position: fixed;
    z-index: 11;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    width: min(32rem, calc(100vw - 2rem));
    max-height: calc(100vh - 2rem);
    overflow-y: auto;
    padding: var(--space-5);
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
  }
  h2 {
    margin: 0 0 var(--space-4);
    font-size: var(--size-md);
  }
  h3 {
    margin: var(--space-4) 0 var(--space-2);
    font-size: var(--size-sm);
    color: var(--color-fg-muted);
    font-weight: 600;
  }
  dl {
    display: grid;
    grid-template-columns: 9rem 1fr;
    gap: var(--space-2) var(--space-3);
    margin: 0;
    font-size: var(--size-sm);
  }
  dt {
    color: var(--color-fg-muted);
  }
  dd {
    margin: 0;
  }
  .parts {
    list-style: none;
    margin: 0;
    padding: 0;
    font-size: var(--size-sm);
  }
  .parts li {
    display: flex;
    justify-content: space-between;
    gap: var(--space-3);
    padding: var(--space-1) 0;
    border-bottom: 1px solid var(--color-border);
  }
  .parts li.imported span:first-child::before {
    content: "▲ ";
    color: var(--color-warn);
  }
  .size {
    color: var(--color-fg-muted);
    white-space: nowrap;
  }
  .total {
    margin: var(--space-3) 0 0;
    font-size: var(--size-xs);
    color: var(--color-fg-muted);
  }
  .buttons {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-2);
    margin-top: var(--space-4);
  }
  .link {
    background: none;
    border: none;
    padding: 0;
    color: var(--color-accent);
    cursor: pointer;
    font: inherit;
    text-decoration: underline;
  }
</style>
