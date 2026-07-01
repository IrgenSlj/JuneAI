<script lang="ts">
  import { onMount } from "svelte";
  import { goto } from "$app/navigation";
  import {
    Mascot,
    OfflineNotice,
    EgressLine,
    ReasonChip,
    type HomeHoldings,
    type SurfacingDecisionView,
    type TaskView,
  } from "@june/ui";
  import { client } from "$lib/api.js";
  import { formatRelative, dueState } from "$lib/dates.js";
  import { profileName } from "$lib/stores/user.svelte.js";

  const ACTIVE = new Set(["planning", "running", "paused", "awaiting_user"]);
  const MOVING = new Set(["planning", "running", "paused"]);

  let holdings = $state<HomeHoldings | null>(null);
  let tasks = $state<TaskView[]>([]);
  let greeting = $state("");
  let loading = $state(true);
  let loadError = $state<string | null>(null);
  let drainedItems = $state<SurfacingDecisionView[]>([]);
  let drainPanelVisible = $state(false);
  let drainNothingWaiting = $state(false);

  async function load() {
    loading = true;
    loadError = null;
    try {
      const [hold, taskRes] = await Promise.all([
        client.getHoldings(profileName.value),
        client
          .getTasks(profileName.value, undefined, 20)
          .catch(() => ({ tasks: [], count: 0 })),
      ]);
      holdings = hold;
      tasks = (taskRes.tasks ?? []).filter((t) => ACTIVE.has(t.status));
    } catch (err) {
      loadError = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
    // Greeting is best-effort; the static fallback covers a brain that's down.
    client
      .getGreeting(profileName.value, profileName.value)
      .then((res) => (greeting = res.greeting))
      .catch(() => {});
  }

  onMount(load);

  const waiting = $derived(tasks.filter((t) => t.status === "awaiting_user"));
  const moving = $derived(tasks.filter((t) => MOVING.has(t.status)));
  const blockedLocal = $derived(
    waiting.filter((t) => (t.blocked_kind ?? "") === "local_only"),
  );
  const held: SurfacingDecisionView[] = $derived(holdings?.held_digest ?? []);
  const needsNow: SurfacingDecisionView[] = $derived(holdings?.needs_now ?? []);
  const egress = $derived(holdings?.egress_today ?? 0);
  const openCount = $derived(holdings?.open_promises ?? tasks.length);
  const clear = $derived(!loading && openCount === 0 && held.length === 0 && needsNow.length === 0);

  const summary = $derived.by(() => {
    if (clear) return "You're clear.";
    const parts: string[] = [];
    parts.push(`You're holding ${openCount} open thread${openCount === 1 ? "" : "s"}`);
    const tail: string[] = [];
    if (waiting.length > 0)
      tail.push(`${waiting.length} waiting on a moment from you`);
    if (moving.length > 0)
      tail.push(`${moving.length} moving on ${moving.length === 1 ? "its" : "their"} own`);
    let s = parts[0];
    if (tail.length) s += ` — ${tail.join(", ")}`;
    if (held.length > 0)
      s += `. June kept ${held.length} small thing${held.length === 1 ? "" : "s"} back for when you have a minute`;
    return s + ".";
  });

  function openSurfacingItem(d: SurfacingDecisionView): void {
    // For deadline kind candidate_id is "deadline:{task_id}"; all kinds route to /tasks for now.
    void d;
    goto("/tasks").catch(() => {});
  }

  async function dismissNeedsNow(d: SurfacingDecisionView): Promise<void> {
    if (holdings?.needs_now) {
      holdings = { ...holdings, needs_now: holdings.needs_now.filter((x) => x.id !== d.id) };
    }
    try {
      await client.sendSurfacingFeedback(d.id, "bad");
    } catch {
      // best-effort
    }
    load().catch(() => {});
  }

  async function dismissDrained(d: SurfacingDecisionView): Promise<void> {
    drainedItems = drainedItems.filter((x) => x.id !== d.id);
    try {
      await client.sendSurfacingFeedback(d.id, "bad");
    } catch {
      // best-effort
    }
  }

  async function catchMeUp(): Promise<void> {
    try {
      const res = await client.drainSurfacing();
      drainedItems = res.drained ?? [];
      drainPanelVisible = true;
      drainNothingWaiting = drainedItems.length === 0;
      load().catch(() => {});
    } catch {
      // best-effort; don't white-screen the page
    }
  }

  function humanizeKind(kind: string): string {
    const map: Record<string, string> = {
      deadline: "Deadline",
      contradiction: "Contradiction",
      promise_nudge: "Promise nudge",
    };
    return map[kind] ?? kind.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
  }

  function movingBody(t: TaskView): string {
    return t.next_action || "Carrying on in the background, the way you asked — no nagging.";
  }
  function waitingBody(t: TaskView): string {
    return t.blocked_reason || "Waiting on a moment from you before it can continue.";
  }
</script>

<svelte:head><title>June — what I'm holding</title></svelte:head>

<main class="home" id="main-content">
  <div class="wrap">
    {#if loadError && !holdings}
      <OfflineNotice kind="system" detail={loadError} onRetry={load} retrying={loading} />
    {/if}

    <!-- hero -->
    <section class="hero">
      <div class="hero-mark"><Mascot /></div>
      <div class="hero-text">
        <p class="hero-greeting">{greeting || "Good to see you."}</p>
        <h1 class="hero-summary" class:clear>{summary}</h1>
        <div class="hero-egress">
          <EgressLine calls={egress} onView={() => (location.href = "/system/receipts")} />
        </div>
      </div>
    </section>

    {#if clear}
      <section class="clear-state">
        <div class="clear-card">
          <p class="clear-lead">Nothing is waiting on you. Nothing's blocked. I'm not holding anything back.</p>
          <p class="clear-sub">
            When something genuinely needs you, it'll show up here — once, plainly. Until then I'll
            be quiet. That's working as intended, not a quiet failure.
          </p>
          <a class="clear-btn" href="/chat">Start something anyway</a>
        </div>
      </section>
    {:else}
      {#if needsNow.length > 0}
        <section class="needs-now-section">
          <div class="needs-now-head">
            <span class="needs-now-label">Needs you now</span>
            <span class="block-count">{needsNow.length} item{needsNow.length === 1 ? "" : "s"}</span>
          </div>
          <div class="needs-now-rows">
            {#each needsNow as d (d.id)}
              <div class="needs-now-row">
                <div class="needs-now-body">
                  <span class="needs-now-kind">{humanizeKind(d.kind)}</span>
                  <p class="needs-now-reason">{d.reason}</p>
                </div>
                <div class="needs-now-actions">
                  <button class="action-btn primary" onclick={() => openSurfacingItem(d)}>Open</button>
                  <button class="action-btn" onclick={() => dismissNeedsNow(d)}>Dismiss</button>
                </div>
              </div>
            {/each}
          </div>
        </section>
      {/if}

      <section class="grid">
        <!-- main column -->
        <div class="col-main">
          <div class="block">
            <div class="block-head">
              <span class="block-label">What you're holding</span>
              <span class="block-count">{waiting.length} waiting · {moving.length} moving</span>
            </div>
            {#if waiting.length === 0 && moving.length === 0}
              <p class="hint">No open promises right now.</p>
            {:else}
              <div class="rows">
                {#each waiting as t (t.id)}
                  <a class="holding-row" href="/tasks">
                    <div class="holding-top">
                      <span class="holding-title">{t.goal}</span>
                      <span class="holding-state"><span class="dot waiting" aria-hidden="true"></span>waiting on you</span>
                    </div>
                    <p class="holding-body">{waitingBody(t)}</p>
                    <span class="holding-meta">{formatRelative(t.updated_at)}</span>
                  </a>
                {/each}
                {#each moving as t (t.id)}
                  <a class="holding-row" href="/tasks">
                    <div class="holding-top">
                      <span class="holding-title">{t.goal}</span>
                      <span class="holding-state"><span class="dot moving" aria-hidden="true"></span>moving on its own</span>
                    </div>
                    <p class="holding-body">{movingBody(t)}</p>
                    {#if t.attempts > 1}
                      <span class="holding-attempt" aria-label="Attempt {t.attempts} of 5">attempt {t.attempts} of 5</span>
                    {/if}
                    <span class="holding-meta">{formatRelative(t.updated_at)}</span>
                  </a>
                {/each}
              </div>
            {/if}
          </div>

          {#if held.length > 0}
            <div class="block">
              <div class="block-head">
                <span class="block-label">June kept these back</span>
                <span class="block-count">{holdings?.held_count ?? held.length} held for later</span>
                <button class="catch-up-btn" onclick={catchMeUp}>Catch me up</button>
              </div>
              <div class="rows">
                {#each held as d (d.id)}
                  <div class="held-row">
                    <div class="held-main">
                      <span class="held-title">{humanizeKind(d.kind)}</span>
                      <span class="held-line">{d.reason}</span>
                    </div>
                    <ReasonChip text={d.reason} />
                  </div>
                {/each}
              </div>
              <a class="why-link" href="/system/silence">Why these were held, and what I didn't show you →</a>
            </div>
          {/if}

          {#if drainPanelVisible}
            <div class="block drain-panel">
              <div class="block-head">
                <span class="block-label">
                  {drainNothingWaiting ? "Nothing was waiting" : "Caught up — here's what June was holding"}
                </span>
              </div>
              {#if drainNothingWaiting}
                <p class="hint drain-nothing">Nothing was being held back. You're fully caught up.</p>
              {:else}
                <div class="rows">
                  {#each drainedItems as d (d.id)}
                    <div class="held-row">
                      <div class="held-main">
                        <span class="held-title">{humanizeKind(d.kind)}</span>
                        <span class="held-line">{d.reason}</span>
                      </div>
                      <button class="action-btn" onclick={() => dismissDrained(d)}>Dismiss</button>
                    </div>
                  {/each}
                </div>
              {/if}
            </div>
          {/if}
        </div>

        <!-- right rail -->
        <aside class="col-rail">
          {#if holdings?.next_deadline}
            <div class="rail-card edge-warn">
              <div class="rail-head">
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path d="M4 11.4V7.6a4 4 0 018 0v3.8l1.2 1.4H2.8L4 11.4z" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round" />
                  <path d="M6.6 13.8a1.5 1.5 0 002.8 0" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" />
                </svg>
                <span class="rail-label">Next deadline</span>
              </div>
              <div class="rail-title">{holdings.next_deadline.goal}</div>
              <div class="rail-sub">
                {#if dueState(holdings.next_deadline.due_at) === "overdue"}
                  Overdue. I'll surface it once — I'm not counting down.
                {:else}
                  Due {formatRelative(holdings.next_deadline.due_at)}. I'll remind you once, at the deadline.
                {/if}
              </div>
            </div>
          {/if}

          {#if blockedLocal.length > 0}
            <div class="rail-card">
              <div class="rail-head muted">
                <span class="dot subtle" aria-hidden="true"></span>
                <span class="rail-label">Blocked · local-only</span>
              </div>
              <div class="rail-title">{blockedLocal[0].goal}</div>
              <div class="rail-sub strong">
                {blockedLocal[0].blocked_reason ||
                  "A networked tool is switched off right now. I can't reach it without you moving the dial — so I've left it."}
              </div>
              <a class="rail-btn" href="/settings">Change the privacy dial — once</a>
              <p class="rail-note">Granted once, revocable anytime, always visible.</p>
            </div>
          {/if}

          <EgressLine calls={egress} as="card" onView={() => (location.href = "/system/receipts")} />

          {#if holdings}
            <a class="chain-line" href="/system/receipts">
              <span class="dot" class:ok={holdings.chain_verified === true} class:subtle={holdings.chain_verified !== true} aria-hidden="true"></span>
              {#if holdings.chain_verified === true}
                Records verified · chain intact
              {:else if holdings.chain_verified === false}
                Records need attention — verify
              {:else}
                Records not verified this session
              {/if}
            </a>
          {/if}
        </aside>
      </section>
    {/if}
  </div>
</main>

<style>
  .home {
    min-height: calc(100dvh - 56px);
    background: var(--color-bg-base);
  }
  .wrap {
    max-width: 1040px;
    margin: 0 auto;
    padding: var(--space-6) var(--space-5) var(--space-8);
    display: flex;
    flex-direction: column;
    gap: var(--space-5);
  }

  .hero {
    display: flex;
    gap: var(--space-5);
    align-items: flex-start;
  }
  .hero-mark {
    flex-shrink: 0;
    margin-top: 2px;
  }
  .hero-text {
    flex: 1;
    min-width: 0;
  }
  .hero-greeting {
    margin: 0 0 7px;
    font-family: var(--font-sans);
    font-size: 13px;
    color: var(--color-fg-muted);
  }
  .hero-summary {
    margin: 0;
    font-family: var(--font-sans);
    font-size: 21px;
    font-weight: 400;
    line-height: 1.4;
    letter-spacing: -0.01em;
    color: var(--color-fg-primary);
    max-width: 680px;
    text-wrap: pretty;
  }
  .hero-summary.clear {
    font-size: 27px;
  }
  .hero-egress {
    margin-top: 14px;
  }

  /* clear state */
  .clear-state {
    border-top: 1px solid var(--color-border);
    padding-top: var(--space-6);
  }
  .clear-card {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-xl);
    background: var(--color-bg-raised);
    padding: 40px 36px;
    text-align: center;
    max-width: 560px;
    margin: 0 auto;
  }
  .clear-lead {
    margin: 0;
    font-family: var(--font-sans);
    font-size: 16.5px;
    color: var(--color-fg-secondary);
    line-height: 1.6;
  }
  .clear-sub {
    margin: 12px 0 0;
    font-family: var(--font-sans);
    font-size: 13.5px;
    color: var(--color-fg-muted);
    line-height: 1.6;
  }
  .clear-btn {
    display: inline-block;
    margin-top: 22px;
    border: 1px solid var(--color-border-strong);
    background: transparent;
    color: var(--color-fg-secondary);
    text-decoration: none;
    font-family: var(--font-sans);
    font-size: 13.5px;
    padding: 9px 18px;
    border-radius: var(--radius-md);
  }
  .clear-btn:hover {
    border-color: var(--color-fg-muted);
  }

  /* holding grid */
  .grid {
    display: grid;
    grid-template-columns: 1fr 320px;
    gap: 28px;
    align-items: start;
  }
  @media (max-width: 860px) {
    .grid {
      grid-template-columns: 1fr;
    }
  }

  .col-main {
    display: flex;
    flex-direction: column;
    gap: 30px;
    min-width: 0;
  }
  .block-head {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 13px;
  }
  .block-label {
    font-family: var(--font-sans);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--color-fg-muted);
  }
  .block-count {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--color-fg-subtle);
  }
  .hint {
    color: var(--color-fg-subtle);
    font-size: var(--size-sm);
  }

  .rows {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }
  .holding-row {
    display: block;
    text-decoration: none;
    padding: 15px 18px;
    background: var(--color-bg-raised);
    border-bottom: 1px solid var(--color-border);
  }
  .holding-row:last-child {
    border-bottom: none;
  }
  .holding-row:hover {
    background: color-mix(in srgb, var(--color-fg-muted) 4%, var(--color-bg-raised));
  }
  .holding-top {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 16px;
  }
  .holding-title {
    font-family: var(--font-sans);
    font-size: 15.5px;
    color: var(--color-fg-primary);
    font-weight: 500;
    overflow-wrap: anywhere;
  }
  .holding-state {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    flex-shrink: 0;
    font-family: var(--font-sans);
    font-size: 12px;
    color: var(--color-fg-muted);
  }
  .holding-body {
    margin: 6px 0 0;
    font-family: var(--font-sans);
    font-size: 13.5px;
    color: var(--color-fg-secondary);
    line-height: 1.55;
    max-width: 560px;
  }
  .holding-meta {
    display: block;
    margin-top: 9px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--color-fg-subtle);
  }
  .holding-attempt {
    display: block;
    margin-top: 5px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--color-fg-subtle);
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: var(--radius-pill);
    flex-shrink: 0;
  }
  .dot.waiting {
    background: var(--color-accent);
  }
  .dot.moving,
  .dot.ok {
    background: var(--color-success);
  }
  .dot.subtle {
    background: var(--color-fg-subtle);
  }

  .held-row {
    display: flex;
    align-items: baseline;
    gap: 12px;
    justify-content: space-between;
    padding: 14px 18px;
    background: var(--color-bg-raised);
    border-bottom: 1px solid var(--color-border);
  }
  .held-row:last-child {
    border-bottom: none;
  }
  .held-title {
    font-family: var(--font-sans);
    font-size: 14.5px;
    color: var(--color-fg-primary);
  }
  .held-line {
    display: block;
    font-family: var(--font-sans);
    font-size: 13px;
    color: var(--color-fg-muted);
    line-height: 1.5;
    margin-top: 4px;
  }
  .why-link {
    display: inline-block;
    margin-top: 12px;
    font-family: var(--font-sans);
    font-size: 12.5px;
    color: var(--color-fg-muted);
    text-decoration: underline;
    text-underline-offset: 2px;
    text-decoration-color: var(--color-border);
  }
  .why-link:hover {
    color: var(--color-fg-secondary);
  }

  /* right rail */
  .col-rail {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .rail-card {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    background: var(--color-bg-raised);
    padding: 16px 18px;
  }
  .rail-card.edge-warn {
    border-left: 3px solid var(--color-warn);
  }
  .rail-head {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 9px;
    color: var(--color-warn);
  }
  .rail-head.muted {
    color: var(--color-fg-muted);
  }
  .rail-label {
    font-family: var(--font-sans);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--color-fg-muted);
  }
  .rail-title {
    font-family: var(--font-sans);
    font-size: 15px;
    color: var(--color-fg-primary);
    line-height: 1.4;
    overflow-wrap: anywhere;
  }
  .rail-sub {
    font-family: var(--font-sans);
    font-size: 12.5px;
    color: var(--color-fg-muted);
    line-height: 1.55;
    margin-top: 8px;
  }
  .rail-sub.strong {
    color: var(--color-fg-secondary);
  }
  .rail-btn {
    display: block;
    text-align: center;
    text-decoration: none;
    margin-top: 13px;
    border: 1px solid var(--color-accent);
    background: var(--color-accent-soft);
    color: var(--color-fg-primary);
    font-family: var(--font-sans);
    font-size: 13px;
    padding: 8px 14px;
    border-radius: 9px;
  }
  .rail-btn:hover {
    background: color-mix(in srgb, var(--color-accent) 20%, transparent);
  }
  .rail-note {
    margin: 9px 0 0;
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--color-fg-subtle);
    line-height: 1.5;
  }
  .chain-line {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: var(--font-sans);
    font-size: 12.5px;
    color: var(--color-fg-muted);
    text-decoration: none;
    padding: 0 2px;
  }
  .chain-line:hover {
    color: var(--color-fg-secondary);
  }

  @media (max-width: 640px) {
    .wrap {
      padding: var(--space-5) var(--space-4) var(--space-7);
    }
    .hero {
      gap: var(--space-4);
    }
  }

  /* needs-now banner */
  .needs-now-section {
    border: 1px solid var(--color-border);
    border-left: 3px solid var(--color-accent);
    border-radius: var(--radius-lg);
    background: var(--color-bg-raised);
    padding: 18px 20px;
  }
  .needs-now-head {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 14px;
  }
  .needs-now-label {
    font-family: var(--font-sans);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--color-accent);
  }
  .needs-now-rows {
    display: flex;
    flex-direction: column;
  }
  .needs-now-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    justify-content: space-between;
    padding: 13px 0;
    border-bottom: 1px solid var(--color-border);
  }
  .needs-now-row:first-child {
    padding-top: 0;
  }
  .needs-now-row:last-child {
    border-bottom: none;
    padding-bottom: 0;
  }
  .needs-now-body {
    flex: 1;
    min-width: 0;
  }
  .needs-now-kind {
    display: block;
    font-family: var(--font-sans);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--color-fg-muted);
    margin-bottom: 5px;
  }
  .needs-now-reason {
    margin: 0;
    font-family: var(--font-sans);
    font-size: 14.5px;
    color: var(--color-fg-primary);
    line-height: 1.5;
    overflow-wrap: anywhere;
  }
  .needs-now-actions {
    display: flex;
    gap: 8px;
    flex-shrink: 0;
    align-items: flex-start;
    padding-top: 2px;
  }

  /* shared action buttons (needs-now Open/Dismiss; drain Dismiss) */
  .action-btn {
    font-family: var(--font-sans);
    font-size: 12.5px;
    padding: 6px 12px;
    border-radius: var(--radius-md);
    cursor: pointer;
    border: 1px solid var(--color-border-strong);
    background: transparent;
    color: var(--color-fg-secondary);
    line-height: 1;
  }
  .action-btn:hover {
    background: color-mix(in srgb, var(--color-fg-muted) 6%, transparent);
  }
  .action-btn.primary {
    border-color: var(--color-accent);
    color: var(--color-fg-primary);
    background: var(--color-accent-soft);
  }
  .action-btn.primary:hover {
    background: color-mix(in srgb, var(--color-accent) 20%, transparent);
  }

  /* catch me up inline button in block-head */
  .catch-up-btn {
    margin-left: auto;
    font-family: var(--font-sans);
    font-size: 12px;
    padding: 4px 10px;
    border-radius: var(--radius-md);
    cursor: pointer;
    border: 1px solid var(--color-border-strong);
    background: transparent;
    color: var(--color-fg-secondary);
    line-height: 1;
  }
  .catch-up-btn:hover {
    border-color: var(--color-fg-muted);
    color: var(--color-fg-primary);
  }

  /* drain panel */
  .drain-panel {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    background: var(--color-bg-raised);
    padding: 16px 18px;
  }
  .drain-nothing {
    margin: 0;
    color: var(--color-fg-muted);
  }
</style>
