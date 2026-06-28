<script lang="ts">
  import { onMount } from "svelte";
  import { page } from "$app/stores";
  import "@june/design/tokens.css";
  import "./app.css";
  import { system, loadSystem } from "$lib/stores/system.svelte.js";
  import { theme, toggleTheme } from "$lib/stores/theme.svelte.js";
  import { chat } from "$lib/stores/chat.svelte.js";
  import { Mascot } from "@june/ui";

  const { children } = $props();

  const pathname = $derived($page.url.pathname);
  // /setup is the onboarding flow — the top nav points at routes that
  // need a configured API, so hide it there and let the wizard own the
  // full viewport until the user finishes.
  const showHeader = $derived(!pathname.startsWith("/setup"));

  // The privacy dial is the mode that gates networked tools; show it in the header.
  function dialLabel(dial: string): string {
    return (
      {
        local_only: "local-only",
        private_by_default: "private by default",
        cloud_first: "cloud-first",
      }[dial] ?? dial
    );
  }

  // The dial governs egress (networked tools / cloud), which is a SEPARATE axis
  // from where the model runs. Spell out the egress consequence so "private by
  // default" never reads as contradicting an on-device "(local)" model label.
  function dialTooltip(dial: string): string {
    return (
      {
        local_only:
          "Privacy dial: local-only — networked tools are blocked; nothing leaves your machine. Change in Settings.",
        private_by_default:
          "Privacy dial: private by default — answers on-device, but networked tools (e.g. web search) are allowed when a request needs them. Change in Settings.",
        cloud_first:
          "Privacy dial: cloud-first — prefers the cloud model; networked tools allowed. Change in Settings.",
      }[dial] ?? "Privacy dial — gates networked tools. Change in Settings."
    );
  }

  onMount(async () => {
    void loadSystem();

    try {
      const { registerSW } = await import("virtual:pwa-register");
      registerSW({ immediate: true });
    } catch {
      // SW module only exists in prod builds; ignore in dev.
    }
  });
</script>

<a class="skip-link" href="#main-content">Skip to main content</a>

{#if showHeader}
  <header class="site-header">
    <div class="site-header-inner">
      <div class="left">
        <a class="brand" href="/" aria-label="June — chat">
          <Mascot busy={chat.streaming} />
        </a>
        <nav class="nav-links" aria-label="Primary">
          <a href="/tasks" class:active={pathname.startsWith("/tasks")}>Tasks</a>
          <a href="/memory" class:active={pathname.startsWith("/memory")}>Memory</a>
          <a href="/skills" class:active={pathname.startsWith("/skills")}>Skills</a>
          <a href="/system" class:active={pathname.startsWith("/system")}>Trust</a>
        </nav>
      </div>

      <div class="right">
        {#if system.data}
          {@const s = system.data}
          <span
            class="runtime"
            title="Endpoint: {s.base_url || 'none'}"
          >
            <span class="dot" data-mode={s.mode}></span>
            <span
              class="runtime-text"
              title={s.mode === "local"
                ? "Model — runs on your device"
                : "Model — runs in the cloud"}>{s.label} · {s.model}</span>
            <span class="runtime-sep" aria-hidden="true">·</span>
            <span class="privacy" data-dial={s.privacy_dial} title={dialTooltip(s.privacy_dial)}>privacy: {dialLabel(s.privacy_dial)}</span>
            {#if s.provider === "gemma"}
              {#if s.ollama_reachable && s.ollama_has_model}
                <span class="runtime-note">· ready</span>
              {:else}
                <a class="warn-link" href="/help/ollama">
                  · {s.ollama_reachable ? "model missing" : "Ollama offline"}
                </a>
              {/if}
            {:else if !s.api_key_present}
              <a class="warn-link" href="/settings">· key missing</a>
            {/if}
            {#if s.version && s.version !== "unknown"}
              <span class="runtime-build" title="Build (git SHA)">· build {s.version}</span>
            {/if}
          </span>
        {:else if system.error}
          <span class="runtime offline">
            <span class="dot" data-mode="api"></span>
            <span class="runtime-text">API unreachable</span>
          </span>
        {/if}

        <button
          type="button"
          class="icon-btn"
          onclick={toggleTheme}
          aria-label={theme.value === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          title={theme.value === "dark" ? "Switch to light mode" : "Switch to dark mode"}
        >
          {#if theme.value === "light"}
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <circle cx="12" cy="12" r="4" />
              <line x1="12" y1="2" x2="12" y2="4" />
              <line x1="12" y1="20" x2="12" y2="22" />
              <line x1="4.93" y1="4.93" x2="6.34" y2="6.34" />
              <line x1="17.66" y1="17.66" x2="19.07" y2="19.07" />
              <line x1="2" y1="12" x2="4" y2="12" />
              <line x1="20" y1="12" x2="22" y2="12" />
              <line x1="4.93" y1="19.07" x2="6.34" y2="17.66" />
              <line x1="17.66" y1="6.34" x2="19.07" y2="4.93" />
            </svg>
          {:else}
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
            </svg>
          {/if}
        </button>

        <a
          href="/settings"
          class="icon-btn"
          class:active={pathname.startsWith("/settings")}
          aria-label="Settings"
          title="Settings"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        </a>
      </div>
    </div>
  </header>
{/if}

{@render children()}

<style>
  .site-header {
    position: sticky;
    top: 0;
    z-index: 10;
    background: color-mix(in srgb, var(--color-bg-base) 88%, transparent);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--color-border);
  }

  .site-header-inner {
    max-width: 980px;
    margin: 0 auto;
    min-height: 56px;
    padding: 0 var(--space-4);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4);
  }

  .left {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    min-width: 0;
  }

  .brand {
    display: inline-flex;
    align-items: center;
    text-decoration: none;
    opacity: 1;
    transition: opacity 150ms ease;
  }
  .brand:hover {
    opacity: 0.75;
  }

  .nav-links {
    display: inline-flex;
    gap: var(--space-3);
  }
  .nav-links a {
    font-size: var(--size-sm);
    color: var(--color-fg-muted);
    text-decoration: none;
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius-sm);
  }
  .nav-links a:hover {
    color: var(--color-accent);
  }
  .nav-links a.active {
    color: var(--color-fg-primary);
    background: var(--color-bg-raised);
  }

  .right {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }

  .runtime {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--color-fg-muted);
    max-width: 44ch;
  }
  .runtime-text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--color-fg-secondary);
  }
  .runtime-sep {
    color: var(--color-fg-subtle);
    opacity: 0.6;
  }
  .privacy {
    color: var(--color-success);
    text-transform: lowercase;
    white-space: nowrap;
  }
  .privacy[data-dial="cloud_first"] {
    color: var(--color-warn);
  }
  .runtime-note {
    color: var(--color-fg-subtle);
  }
  .runtime-build {
    color: var(--color-fg-subtle);
    opacity: 0.7;
    font-size: 10.5px;
  }
  .runtime.offline {
    color: var(--color-danger);
  }
  .warn-link {
    color: var(--color-accent);
    text-decoration: none;
    margin-left: var(--space-1);
  }
  .warn-link:hover {
    text-decoration: underline;
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: var(--radius-pill);
    background: var(--color-success);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--color-success) 20%, transparent);
    flex-shrink: 0;
  }
  .dot[data-mode="cloud"] {
    background: var(--color-warn);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--color-warn) 20%, transparent);
  }
  .dot[data-mode="api"] {
    background: var(--color-accent);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--color-accent) 20%, transparent);
  }

  .icon-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    border-radius: var(--radius-md);
    background: transparent;
    color: var(--color-fg-muted);
    border: 1px solid transparent;
    cursor: pointer;
    text-decoration: none;
    transition: color 120ms ease, background 120ms ease, border-color 120ms ease;
  }
  .icon-btn:hover {
    color: var(--color-fg-primary);
    background: var(--color-bg-raised);
  }
  .icon-btn.active {
    color: var(--color-fg-primary);
    background: var(--color-bg-raised);
    border-color: var(--color-border);
  }

  /* At narrow widths drop the runtime text to keep the header from wrapping. */
  @media (max-width: 640px) {
    .runtime-text,
    .runtime-sep,
    .privacy,
    .runtime-note {
      display: none;
    }
    .warn-link {
      font-size: var(--size-xs);
    }
  }
</style>
