<script lang="ts">
  import ChatBubble from "./ChatBubble.svelte";
  import type { ChatMessage } from "./types.js";

  interface Props {
    messages: ChatMessage[];
    streaming?: boolean;
  }

  const { messages, streaming = false }: Props = $props();
  const lastId = $derived(messages[messages.length - 1]?.id);

  let scrollEl: HTMLDivElement | undefined = $state();
  let pinned = $state(true);

  // Auto-scroll to the bottom only while the user is pinned to the tail.
  // If they scroll up to re-read history, we stop yanking the viewport.
  const stickyTrigger = $derived(
    messages.reduce((sum, m) => sum + m.content.length, 0) + messages.length,
  );

  function isNearBottom(el: HTMLDivElement): boolean {
    const slack = 48;
    return el.scrollHeight - el.clientHeight - el.scrollTop < slack;
  }

  function handleScroll() {
    if (!scrollEl) return;
    pinned = isNearBottom(scrollEl);
  }

  $effect(() => {
    void stickyTrigger;
    if (scrollEl && pinned) scrollEl.scrollTop = scrollEl.scrollHeight;
  });
</script>

<div class="list" bind:this={scrollEl} onscroll={handleScroll}>
  {#each messages as message (message.id)}
    <ChatBubble
      role={message.role}
      content={message.content}
      toolName={message.toolName}
      pending={streaming && message.id === lastId}
    />
  {/each}
  {#if !pinned && messages.length > 0}
    <button
      type="button"
      class="jump"
      onclick={() => {
        if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;
      }}
      aria-label="Scroll to bottom"
    >
      ↓ Jump to latest
    </button>
  {/if}
</div>

<style>
  .list {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    padding: var(--space-5) var(--space-4);
    scroll-behavior: smooth;
    position: relative;
  }

  .list::-webkit-scrollbar {
    width: 8px;
  }
  .list::-webkit-scrollbar-thumb {
    background: var(--color-border);
    border-radius: var(--radius-pill);
  }

  .jump {
    position: sticky;
    bottom: var(--space-3);
    align-self: center;
    padding: var(--space-2) var(--space-4);
    font-size: var(--size-xs);
    font-family: var(--font-sans);
    color: var(--color-fg-primary);
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border-strong);
    border-radius: var(--radius-pill);
    cursor: pointer;
    box-shadow: var(--shadow-md);
  }
  .jump:hover {
    background: var(--color-bg-sunken);
  }
</style>
