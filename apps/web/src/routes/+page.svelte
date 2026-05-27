<script lang="ts">
  import { Composer, MessageList, OfflineNotice } from "@june/ui";
  import {
    chat,
    sendMessage,
    cancelStream,
    regenerateLast,
    voteRecall,
  } from "$lib/stores/chat.svelte.js";
  import { system, loadSystem } from "$lib/stores/system.svelte.js";
  import { onMount } from "svelte";
  import { client } from "$lib/api.js";
  import { profileName } from "$lib/stores/user.svelte.js";
  import { capture, submitCapture } from "$lib/stores/capture.svelte.js";

  let focusComposer: (() => void) | undefined = $state();
  let greeting = $state("");
  let captureText = $state("");

  async function runCapture() {
    if (await submitCapture(captureText)) captureText = "";
  }

  function onCaptureKey(e: KeyboardEvent) {
    if (e.key === "Enter") {
      e.preventDefault();
      void runCapture();
    }
  }

  onMount(async () => {
    try {
      const res = await client.getGreeting(profileName.value, profileName.value);
      greeting = res.greeting;
    } catch {
      // Brain down — the static fallback in the template covers this.
    }
  });

  const canRegenerate = $derived(
    !chat.streaming &&
      chat.messages.length > 0 &&
      chat.messages[chat.messages.length - 1]?.role === "assistant" &&
      chat.messages.some((m) => m.role === "user"),
  );

  // Show an elapsed-time hint once a stream has been running for a few
  // seconds with nothing to show. Gemma cold-starts from Ollama can take
  // 10-30s before the first token — without feedback the UI looks hung.
  let now = $state(Date.now());
  $effect(() => {
    if (!chat.streaming) return;
    const id = setInterval(() => (now = Date.now()), 500);
    return () => clearInterval(id);
  });
  const elapsedSec = $derived(
    chat.streaming && chat.streamStartedAt
      ? Math.floor((now - chat.streamStartedAt) / 1000)
      : 0,
  );
  const awaitingFirstToken = $derived(
    chat.streaming &&
      chat.messages[chat.messages.length - 1]?.role === "assistant" &&
      chat.messages[chat.messages.length - 1]?.content === "",
  );

  function handleGlobalKey(event: KeyboardEvent) {
    const isMod = event.metaKey || event.ctrlKey;
    const target = event.target as HTMLElement | null;
    const inEditable =
      target?.tagName === "INPUT" ||
      target?.tagName === "TEXTAREA" ||
      target?.isContentEditable;

    if (isMod && event.key.toLowerCase() === "k") {
      event.preventDefault();
      focusComposer?.();
    } else if (event.key === "Escape" && chat.streaming) {
      event.preventDefault();
      cancelStream();
    } else if (event.key === "/" && !inEditable && !isMod && !event.shiftKey) {
      event.preventDefault();
      focusComposer?.();
    }
  }
</script>

<svelte:head>
  <title>June — your personal AI</title>
</svelte:head>

<svelte:window onkeydown={handleGlobalKey} />

<main class="app" id="main-content">
  <section class="transcript" aria-label="Conversation">
    {#if chat.messages.length === 0 && !system.data && system.error}
      <div class="empty">
        <OfflineNotice
          kind="system"
          detail={system.error}
          onRetry={loadSystem}
          retrying={system.loading}
        />
      </div>
    {:else if chat.messages.length === 0}
      <div class="empty">
        <p>{greeting || "Hi, I'm June. I'll remember what matters so you don't have to."}</p>
        <div class="capture">
          <input
            type="text"
            bind:value={captureText}
            onkeydown={onCaptureKey}
            placeholder="Capture a thought — a task, a promise, how you feel…"
            aria-label="Quick capture"
          />
          <button type="button" onclick={runCapture} disabled={capture.loading || !captureText.trim()}>
            {capture.loading ? "…" : "Capture"}
          </button>
        </div>
        {#if capture.error}
          <p class="capture-error">Couldn't capture that — is the brain running?</p>
        {:else if capture.result}
          <div class="capture-result" aria-live="polite">
            <p class="kinds">
              Understood as: {capture.result.kinds.join(", ") || "note"}
            </p>
            {#if capture.result.message}
              <p class="support">{capture.result.message}</p>
            {/if}
            {#if capture.result.candidates.length}
              <ul>
                {#each capture.result.candidates as c (c.id)}
                  <li>{c.title}{c.requires_approval ? " · needs your approval" : " · saved"}</li>
                {/each}
              </ul>
            {/if}
          </div>
        {/if}
        <p class="muted">Or type below to chat.</p>
      </div>
    {:else}
      <MessageList
        messages={chat.messages}
        streaming={chat.streaming}
        onVote={voteRecall}
      />
      {#if awaitingFirstToken && elapsedSec >= 4}
        <p class="waiting" aria-live="polite">
          Still thinking… {elapsedSec}s
          {#if elapsedSec >= 15}
            · Gemma sometimes takes a moment to warm up.
          {/if}
        </p>
      {/if}
    {/if}
  </section>

  <footer class="compose">
    {#if canRegenerate}
      <div class="actions">
        <button type="button" class="regenerate" onclick={regenerateLast}>
          ↻ Regenerate
        </button>
      </div>
    {/if}
    <Composer
      streaming={chat.streaming}
      bind:focus={focusComposer}
      onSubmit={sendMessage}
      onCancel={cancelStream}
    />
  </footer>
</main>

<style>
  .app {
    display: flex;
    flex-direction: column;
    height: calc(100dvh - 60px);
    max-width: 860px;
    margin: 0 auto;
    padding: 0 var(--space-4);
  }

  .transcript {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }

  .empty {
    margin: auto;
    max-width: 40ch;
    text-align: center;
    color: var(--color-fg-muted);
  }
  .empty p {
    margin: var(--space-2) 0;
  }
  .empty .muted {
    color: var(--color-fg-subtle);
    font-size: var(--size-sm);
  }
  .capture {
    display: flex;
    gap: var(--space-2);
    margin: var(--space-4) 0;
  }
  .capture input {
    flex: 1;
    padding: var(--space-2) var(--space-3);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-bg-raised);
    color: var(--color-fg-primary);
    font: inherit;
  }
  .capture button {
    background: var(--color-accent);
    color: var(--color-bg);
    border: none;
    border-radius: var(--radius-md);
    padding: var(--space-2) var(--space-4);
    cursor: pointer;
  }
  .capture button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .capture-result {
    text-align: left;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    padding: var(--space-3) var(--space-4);
    margin: var(--space-2) 0;
  }
  .capture-result .kinds {
    font-weight: 500;
    color: var(--color-fg-primary);
  }
  .capture-result .support {
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
  }
  .capture-result ul {
    margin: var(--space-2) 0 0;
    padding-left: var(--space-4);
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
  }
  .capture-error {
    color: var(--color-accent);
    font-size: var(--size-sm);
  }

  .waiting {
    margin: var(--space-2) auto 0;
    font-size: var(--size-xs);
    font-family: var(--font-mono);
    color: var(--color-fg-subtle);
    text-align: center;
  }

  .compose {
    padding: var(--space-3) 0 var(--space-5);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .actions {
    display: flex;
    justify-content: center;
  }

  .regenerate {
    background: transparent;
    color: var(--color-fg-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-pill);
    padding: var(--space-1) var(--space-4);
    font-size: var(--size-xs);
    font-family: var(--font-sans);
    cursor: pointer;
    transition: color 120ms ease, border-color 120ms ease;
  }
  .regenerate:hover {
    color: var(--color-fg-primary);
    border-color: var(--color-border-strong);
  }
</style>
