<script lang="ts">
  import ChatBubble from "./ChatBubble.svelte";
  import type { ChatMessage } from "./types.js";

  interface Props {
    messages: ChatMessage[];
  }

  const { messages }: Props = $props();

  let scrollEl: HTMLDivElement | undefined = $state();

  // Auto-scroll to the bottom whenever the message list changes. We
  // compute the total token count across all messages so that streams
  // (which mutate the last message's content) also trigger scroll.
  const stickyTrigger = $derived(
    messages.reduce((sum, m) => sum + m.content.length, 0) + messages.length,
  );

  $effect(() => {
    void stickyTrigger;
    if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;
  });
</script>

<div class="list" bind:this={scrollEl}>
  {#each messages as message (message.id)}
    <ChatBubble
      role={message.role}
      content={message.content}
      toolName={message.toolName}
    />
  {/each}
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
  }

  .list::-webkit-scrollbar {
    width: 8px;
  }
  .list::-webkit-scrollbar-thumb {
    background: var(--color-border);
    border-radius: var(--radius-pill);
  }
</style>
