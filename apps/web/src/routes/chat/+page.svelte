<script lang="ts">
  import {
    Composer,
    MessageList,
    OfflineNotice,
    ActivityStream,
    type TaskView,
  } from "@june/ui";
  import {
    chat,
    sendMessage,
    cancelStream,
    regenerateLast,
    toggleActivity,
    voteRecall,
    loadHistory,
    switchToPrivateAndRetry,
    approveAndRetry,
  } from "$lib/stores/chat.svelte.js";
  import { system, loadSystem } from "$lib/stores/system.svelte.js";
  import { onMount } from "svelte";
  import { client } from "$lib/api.js";
  import { dueState } from "$lib/dates.js";
  import { profileName } from "$lib/stores/user.svelte.js";

  let focusComposer: (() => void) | undefined = $state();
  let greeting = $state("");
  let continuityTasks: TaskView[] = $state([]);
  let continuityError: string | null = $state(null);

  const ACTIVE_PROMISE_STATUSES = new Set(["planning", "running", "paused", "awaiting_user"]);
  const hasActivity = $derived(chat.activity.length > 0);
  // Show the platform-correct send-shortcut glyph in the composer hint.
  const modKeyLabel =
    typeof navigator !== "undefined" && /Mac|iP(hone|ad|od)/.test(navigator.platform)
      ? "⌘"
      : "Ctrl";

  onMount(() => {
    void loadHistory(profileName.value);
    void loadContinuity();
    void client
      .getGreeting(profileName.value, profileName.value)
      .then((res) => {
        greeting = res.greeting;
      })
      .catch(() => {
        // Brain down — the static fallback in the template covers this.
      });
  });

  async function loadContinuity(): Promise<void> {
    continuityError = null;
    try {
      const res = await client.getTasks(profileName.value, undefined, 20);
      continuityTasks = res.tasks ?? [];
    } catch (err) {
      continuityError = err instanceof Error ? err.message : String(err);
    }
  }

  function isLocalOnlyBlocked(task: TaskView): boolean {
    const blockedReason = String(task.blocked_reason ?? "").toLowerCase();
    if (blockedReason.includes("local-only")) return true;
    return (task.plan ?? []).some((step) => {
      const description = String(step.description ?? "").toLowerCase();
      const result = String(step.tool_result ?? "").toLowerCase();
      return (
        step.status === "pending" &&
        (description.includes("local-only") || result.includes("local-only"))
      );
    });
  }

  function promiseSummary(
    openCount: number,
    waitingCount: number,
    blockedCount: number,
    overdueCount: number,
    dueSoonCount: number,
  ): string {
    if (continuityError) return "I could not read open promises.";
    if (openCount === 0) return "No open promises are waiting right now.";

    const parts = [`${openCount} open ${openCount === 1 ? "promise" : "promises"}`];
    if (waitingCount > 0) parts.push(`${waitingCount} waiting on you`);
    if (blockedCount > 0) parts.push(`${blockedCount} blocked by local-only mode`);
    if (overdueCount > 0) parts.push(`${overdueCount} overdue`);
    else if (dueSoonCount > 0) parts.push(`${dueSoonCount} due soon`);
    return `I am holding ${parts.join(", ")}.`;
  }

  function dialLabel(dial: string): string {
    return (
      {
        local_only: "local-only",
        private_by_default: "private",
        cloud_first: "cloud-first",
      }[dial] ?? dial
    );
  }

  const openPromises = $derived(
    continuityTasks.filter((task) => ACTIVE_PROMISE_STATUSES.has(task.status)),
  );
  const waitingPromises = $derived(
    continuityTasks.filter((task) => task.status === "awaiting_user"),
  );
  const blockedPromises = $derived(waitingPromises.filter(isLocalOnlyBlocked));
  const openPromiseCount = $derived(openPromises.length);
  const waitingCount = $derived(waitingPromises.length);
  const blockedCount = $derived(blockedPromises.length);
  // Due state is derived here, at render time — no background timer (ADR 0016).
  const overdueCount = $derived(
    openPromises.filter((t) => dueState(t.due_at) === "overdue").length,
  );
  const dueSoonCount = $derived(
    openPromises.filter((t) => dueState(t.due_at) === "due_soon").length,
  );
  const recallStatus = $derived(system.data?.semantic_recall_status ?? "unknown");
  const recallDegraded = $derived(recallStatus === "degraded");
  const continuityLine = $derived(
    promiseSummary(openPromiseCount, waitingCount, blockedCount, overdueCount, dueSoonCount),
  );
  const privacyLabel = $derived(
    system.data ? dialLabel(system.data.privacy_dial) : system.error ? "offline" : "checking",
  );
  const runtimeLabel = $derived(
    system.data ? (system.data.mode === "local" ? "local runtime" : "cloud runtime") : "runtime",
  );

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
      <div class="home-state">
        <section class="continuity" aria-label="Continuity summary">
          <div class="continuity-line">
            <span
              class="continuity-dot"
              class:warn={waitingCount > 0 || overdueCount > 0 || recallDegraded || continuityError !== null}
              class:ok={waitingCount === 0 && overdueCount === 0 && !recallDegraded && continuityError === null}
              aria-hidden="true"
            ></span>
            <span>{continuityLine}</span>
          </div>
          <div class="continuity-grid">
            <a class="continuity-metric" href="/tasks">
              <span class="metric-main">{openPromiseCount}</span>
              <span class="metric-label">open promises</span>
            </a>
            <a class="continuity-metric" class:warn={waitingCount > 0} href="/tasks">
              <span class="metric-main">{waitingCount}</span>
              <span class="metric-label">waiting on you</span>
            </a>
            {#if overdueCount > 0 || dueSoonCount > 0}
              <a
                class="continuity-metric"
                class:warn={overdueCount > 0}
                href="/tasks"
              >
                <span class="metric-main">{overdueCount > 0 ? overdueCount : dueSoonCount}</span>
                <span class="metric-label">{overdueCount > 0 ? "overdue" : "due soon"}</span>
              </a>
            {/if}
            <a class="continuity-metric" href="/system">
              <span class="metric-main">{privacyLabel}</span>
              <span class="metric-label">{runtimeLabel}</span>
            </a>
            <a class="continuity-metric" class:warn={recallDegraded} href="/system">
              <span class="metric-main">{recallStatus}</span>
              <span class="metric-label">semantic recall</span>
            </a>
          </div>
          {#if continuityError}
            <button type="button" class="continuity-retry" onclick={loadContinuity}>
              Retry promises
            </button>
          {/if}
        </section>
        <div class="greeting">
          <p class="greeting-line">{greeting || "Hi, I'm June."}</p>
          <p class="greeting-sub">I'll remember what matters so you don't have to.</p>
        </div>
      </div>
    {:else}
      <MessageList
        messages={chat.messages}
        streaming={chat.streaming}
        onVote={voteRecall}
      />
      {#if awaitingFirstToken && elapsedSec >= 4}
        <p class="waiting" aria-live="polite">
          {system.data?.mode === "local" ? "Thinking locally" : "Thinking"}… {elapsedSec}s
          {#if elapsedSec >= 15 && system.data?.mode === "local"}
            · the on-device model sometimes takes a moment to warm up.
          {/if}
        </p>
      {/if}
    {/if}
  </section>

  <!-- COMPOSER BAND (center fulcrum) -->
  <div class="compose-band">
    {#if chat.blockedTool && !chat.streaming}
      <div class="blocked-notice" role="status">
        <span class="blocked-text">
          June stopped before using <strong>{chat.blockedTool.name}</strong> — it
          needs the internet, which <strong>Local-only</strong> mode blocks.
        </span>
        <button type="button" class="blocked-switch" onclick={switchToPrivateAndRetry}>
          Switch to Private-by-default &amp; retry
        </button>
      </div>
    {/if}
    {#if chat.pendingApproval && !chat.streaming}
      <div class="approval-notice" role="status">
        <span class="blocked-text">
          June wants to use <strong>{chat.pendingApproval.name}</strong>{#if chat.pendingApproval.reason}
            — {chat.pendingApproval.reason}{/if}. This needs your approval before it runs.
        </span>
        <button type="button" class="blocked-switch" onclick={approveAndRetry}>
          Approve &amp; retry
        </button>
      </div>
    {/if}
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
        <!-- Stacked-arrow glyph (rotates when open) + accent dot when activity awaits -->
        <svg
          class="glyph"
          width="16"
          height="16"
          viewBox="0 0 16 16"
          fill="none"
          aria-hidden="true"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M8 2.5v7M5 6.5l3 3 3-3"
            stroke="currentColor"
            stroke-width="1.5"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
          <path
            d="M3 13h10"
            stroke="currentColor"
            stroke-width="1.5"
            stroke-linecap="round"
          />
        </svg>
        {#if hasActivity && !chat.activityOpen}
          <span class="activity-dot" aria-hidden="true"></span>
        {/if}
      </button>
      <div class="composer-wrap">
        <Composer
          streaming={chat.streaming}
          placeholder={chat.streaming ? "June is replying…" : "Write to June…"}
          bind:focus={focusComposer}
          onSubmit={sendMessage}
          onCancel={cancelStream}
        />
        <div class="composer-hint">
          <kbd>{modKeyLabel} &#9166;</kbd>
          <span>to send</span>
          {#if chat.streaming}
            <span class="hint-sep">·</span>
            <span>Esc to stop</span>
          {/if}
          <span class="hint-sep">·</span>
          <a class="glass-link" href="/system/glass">Open Glass Box →</a>
        </div>
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

  .home-state {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    gap: var(--space-4);
    padding: var(--space-4) var(--space-2) var(--space-5);
  }

  .continuity {
    width: min(100%, 720px);
    margin: 0 auto;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: color-mix(in srgb, var(--color-bg-raised) 88%, transparent);
    padding: var(--space-3);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .continuity-line {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    color: var(--color-fg-secondary);
    font-size: var(--size-sm);
    line-height: var(--leading-normal);
  }
  .continuity-dot {
    width: 8px;
    height: 8px;
    border-radius: var(--radius-pill);
    flex: 0 0 auto;
  }
  .continuity-dot.ok {
    background: var(--color-success);
  }
  .continuity-dot.warn {
    background: var(--color-warn);
  }
  .continuity-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1px;
    overflow: hidden;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-border);
  }
  .continuity-metric {
    min-width: 0;
    background: var(--color-bg-base);
    color: var(--color-fg-primary);
    text-decoration: none;
    padding: var(--space-2) var(--space-3);
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
  .continuity-metric:hover {
    color: var(--color-accent);
  }
  .continuity-metric.warn .metric-main {
    color: var(--color-warn);
  }
  .metric-main {
    font-family: var(--font-mono);
    font-size: var(--size-sm);
    font-weight: 700;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .metric-label {
    color: var(--color-fg-subtle);
    font-size: var(--size-xs);
    line-height: var(--leading-tight);
  }
  .continuity-retry {
    align-self: flex-start;
    background: transparent;
    color: var(--color-fg-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-3);
    font: inherit;
    font-size: var(--size-xs);
    cursor: pointer;
  }
  .continuity-retry:hover {
    color: var(--color-accent);
    border-color: var(--color-accent);
  }

  .greeting {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 0 var(--space-3);
  }
  .greeting-line {
    font-family: var(--font-sans);
    font-size: 26px;
    line-height: 1.4;
    letter-spacing: 0;
    color: var(--color-fg-primary);
    margin: 0;
    max-width: 560px;
  }
  .greeting-sub {
    font-family: var(--font-sans);
    font-size: 18px;
    line-height: 1.55;
    color: var(--color-fg-muted);
    margin: var(--space-2) 0 0;
    max-width: 560px;
  }

  .empty {
    margin: auto;
    max-width: 40ch;
    text-align: center;
    color: var(--color-fg-muted);
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
    padding: var(--space-4) 0;
    border-top: 1px solid var(--color-border);
    border-bottom: 1px solid var(--color-border);
    background: var(--color-bg-base);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .blocked-notice {
    max-width: 820px;
    width: 100%;
    margin: 0 auto var(--space-2);
    padding: var(--space-3) var(--space-4);
    display: flex;
    align-items: center;
    gap: var(--space-3);
    flex-wrap: wrap;
    border: 1px solid var(--color-warn);
    border-radius: var(--radius-md);
    background: color-mix(in srgb, var(--color-warn) 10%, transparent);
    font-size: var(--size-sm);
    color: var(--color-fg-secondary);
  }
  .approval-notice {
    max-width: 820px;
    width: 100%;
    margin: 0 auto var(--space-2);
    padding: var(--space-3) var(--space-4);
    display: flex;
    align-items: center;
    gap: var(--space-3);
    flex-wrap: wrap;
    border: 1px solid var(--color-accent);
    border-radius: var(--radius-md);
    background: color-mix(in srgb, var(--color-accent) 10%, transparent);
    font-size: var(--size-sm);
    color: var(--color-fg-secondary);
  }
  .blocked-text {
    flex: 1;
    min-width: 12rem;
    line-height: var(--leading-normal);
  }
  .blocked-switch {
    flex-shrink: 0;
    background: var(--color-accent);
    color: var(--color-accent-ink);
    border: none;
    border-radius: var(--radius-md);
    padding: var(--space-2) var(--space-3);
    font-size: var(--size-sm);
    font-weight: 500;
    cursor: pointer;
  }
  .blocked-switch:hover {
    background: var(--color-accent-muted);
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
    align-items: flex-start;
    gap: var(--space-3);
    max-width: 820px;
    width: 100%;
    margin: 0 auto;
    padding: 0 var(--space-5);
  }

  .composer-wrap {
    flex: 1;
    min-width: 0;
  }

  .activity-toggle {
    position: relative;
    flex-shrink: 0;
    width: 50px;
    height: 50px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border-strong);
    border-radius: var(--radius-lg);
    box-shadow: 0 1px 0 var(--color-border), var(--shadow-md);
    color: var(--color-fg-muted);
    cursor: pointer;
    transition: color 120ms ease, border-color 120ms ease, background 120ms ease;
    padding: 0;
  }
  .activity-toggle:hover {
    color: var(--color-fg-secondary);
    border-color: var(--color-accent);
  }
  .activity-toggle.open {
    background: var(--color-bg-sunken);
    color: var(--color-fg-secondary);
  }

  .glyph {
    display: block;
    transition: transform var(--motion-base, 220ms) var(--ease, ease);
  }
  .activity-toggle.open .glyph {
    transform: rotate(180deg);
  }

  .activity-dot {
    position: absolute;
    top: 7px;
    right: 7px;
    width: 5px;
    height: 5px;
    border-radius: var(--radius-pill);
    background: var(--color-accent);
  }

  .composer-hint {
    margin-top: var(--space-2);
    padding-left: var(--space-1);
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--color-fg-subtle);
  }
  .composer-hint kbd {
    font-family: var(--font-mono);
    color: var(--color-fg-muted);
    border: 1px solid var(--color-border);
    border-radius: 4px;
    padding: 1px 6px;
  }
  .hint-sep {
    opacity: 0.5;
  }

  .glass-link {
    color: var(--color-fg-subtle);
    text-decoration: none;
    font-family: var(--font-mono);
    font-size: 11px;
  }
  .glass-link:hover {
    color: var(--color-fg-muted);
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

  @media (max-width: 640px) {
    .home-state {
      justify-content: center;
      padding: var(--space-4) 0 var(--space-4);
    }
    .continuity-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .continuity-metric {
      min-height: 64px;
    }
  }
</style>
