<script lang="ts">
  import { Composer, MessageList, type ChatMessage } from "@june/ui";

  // Sending arrives in M1 with providers and the turn loop. Until then the
  // composer is visibly disabled rather than pretending to work.
  const messages: ChatMessage[] = [];
</script>

<main class="chat">
  <section class="messages" aria-label="Conversation">
    {#if messages.length === 0}
      <p class="empty">No conversation yet. Connecting a model arrives in the next milestone.</p>
    {:else}
      <MessageList {messages} />
    {/if}
  </section>
  <Composer disabled placeholder="Connect a model to start" onSubmit={() => {}} />
</main>

<style>
  .chat {
    display: grid;
    grid-template-rows: 1fr auto;
    min-height: 0;
    max-width: 48rem;
    width: 100%;
    margin: 0 auto;
    padding: 1rem 1.25rem;
  }
  .messages {
    overflow-y: auto;
  }
  .empty {
    opacity: 0.6;
    margin-top: 30vh;
    text-align: center;
  }
</style>
