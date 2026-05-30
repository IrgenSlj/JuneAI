<script lang="ts">
  import { Composer, MessageList, OfflineNotice, ActivityStream } from "@june/ui";
  import {
    chat,
    sendMessage,
    cancelStream,
    regenerateLast,
    toggleActivity,
    voteRecall,
    loadHistory,
  } from "$lib/stores/chat.svelte.js";
  import { system, loadSystem } from "$lib/stores/system.svelte.js";
  import { onMount } from "svelte";
  import { client } from "$lib/api.js";
  import { profileName } from "$lib/stores/user.svelte.js";

  let focusComposer: (() => void) | undefined = $state();
  let greeting = $state("");

  onMount(async () => {
    await loadHistory(profileName.value);
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

  // Most recent provenance step for the collapsed activity strip.
  const latestProvenanceStep = $derived(
    [...chat.activity].reverse().find((s) => s.kind === "provenance"),
  );
  const latestActivityStep = $derived(
    chat.activity.length > 0 ? chat.activity[chat.activity.length - 1] : null,
  );
  // Prefer provenance step in the collapsed strip so cloud boundary stays visible.
  const collapsedStep = $derived(latestProvenanceStep ?? latestActivityStep);

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
  <!-- CONVERSATION region (top) -->
  <section
    class="transcript"
    class:transcript-half={chat.activityOpen}
    aria-label="Conversation"
  >
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
        <p class="muted">Type below to chat.</p>
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

  <!-- COMPOSER BAND (center fulcrum) -->
  <div class="compose-band">
    {#if canRegenerate}
      <div class="actions">
        <button type="button" class="regenerate" onclick={regenerateLast}>
          &#8635; Regenerate
        </button>
      </div>
    {/if}
    <div class="composer-row">
      <button
        type="button"
        class="activity-toggle"
        class:open={chat.activityOpen}
        onclick={toggleActivity}
        aria-expanded={chat.activityOpen}
        aria-label={chat.activityOpen ? "Hide activity" : "Show activity"}
        title={chat.activityOpen ? "Hide activity" : "Show activity"}
      >
        <!-- Chevron SVG, rotates when open -->
        <svg
          class="chevron"
          width="14"
          height="14"
          viewBox="0 0 14 14"
          fill="none"
          aria-hidden="true"
          xmlns="http://www.w3.org/2000/svg"
        >
          <polyline
            points="3,5 7,9 11,5"
            stroke="currentColor"
            stroke-width="1.5"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
      </button>
      <div class="composer-wrap">
        <Composer
          streaming={chat.streaming}
          bind:focus={focusComposer}
          onSubmit={sendMessage}
          onCancel={cancelStream}
        />
      </div>
    </div>
  </div>

  <!-- ACTIVITY TERMINAL (bottom) -->
  <div
    class="activity-region"
    class:activity-open={chat.activityOpen}
    class:activity-collapsed={!chat.activityOpen}
  >
    {#if chat.activityOpen}
      <ActivityStream steps={chat.activity} open />
    {:else}
      <!-- Slim one-line strip: shows most recent (preferring provenance) step -->
      <div class="activity-strip" aria-label="Latest activity">
        {#if collapsedStep}
          {#if collapsedStep.kind === "provenance"}
            <span
              class="strip-dot"
              class:cloud={collapsedStep.cloud}
              class:local={!collapsedStep.cloud}
              aria-label={collapsedStep.cloud ? "cloud" : "local"}
            ></span>
          {/if}
          <span class="strip-label">{collapsedStep.label}</span>
        {/if}
      </div>
    {/if}
  </div>
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

  /* CONVERSATION */
  .transcript {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: auto;
  }

  .transcript.transcript-half {
    flex: 1 1 50%;
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
  .waiting {
    margin: var(--space-2) auto 0;
    font-size: var(--size-xs);
    font-family: var(--font-mono);
    color: var(--color-fg-subtle);
    text-align: center;
  }

  /* COMPOSER BAND */
  .compose-band {
    flex: 0 0 auto;
    padding: var(--space-3) 0 var(--space-3);
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

  .composer-row {
    display: flex;
    align-items: flex-end;
    gap: var(--space-2);
  }

  .composer-wrap {
    flex: 1;
    min-width: 0;
  }

  .activity-toggle {
    flex-shrink: 0;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    color: var(--color-fg-subtle);
    cursor: pointer;
    transition: color 120ms ease, border-color 120ms ease, background 120ms ease;
    padding: 0;
    margin-bottom: var(--space-2);
  }
  .activity-toggle:hover {
    color: var(--color-fg-muted);
    border-color: var(--color-border-strong);
    background: var(--color-bg-raised);
  }
  .activity-toggle.open {
    color: var(--color-accent);
    border-color: var(--color-accent);
    background: color-mix(in srgb, var(--color-accent) 8%, transparent);
  }

  .chevron {
    display: block;
    transition: transform 200ms ease;
  }
  .activity-toggle.open .chevron {
    transform: rotate(180deg);
  }

  /* ACTIVITY TERMINAL */
  .activity-region {
    min-height: 0;
  }

  .activity-region.activity-open {
    flex: 1 1 50%;
    overflow: hidden;
  }

  .activity-region.activity-collapsed {
    flex: 0 0 auto;
    padding-bottom: var(--space-3);
  }

  .activity-strip {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-1) var(--space-2);
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
    border-top: 1px solid var(--color-border);
    min-height: 24px;
    overflow: hidden;
  }

  .strip-dot {
    width: 6px;
    height: 6px;
    border-radius: var(--radius-pill);
    flex-shrink: 0;
  }

  .strip-dot.cloud {
    background: var(--color-accent);
  }

  .strip-dot.local {
    background: var(--color-success);
  }

  .strip-label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
