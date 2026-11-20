<script lang="ts">
  // Renders inline Markdown nodes as plain elements. No {@html} anywhere.
  import { getPlatform } from "../platform/index.js";
  import type { Inline } from "./parse.js";
  import Self from "./Inline.svelte";

  const { nodes }: { nodes: Inline[] } = $props();

  // In the Mac app a link must open in the user's browser, never inside June's
  // window (which holds the token). The href was already checked by safeHref.
  function open(event: MouseEvent, href: string): void {
    const platform = getPlatform();
    if (platform.runtime !== "tauri") return;
    event.preventDefault();
    void platform.openExternal(href);
  }
</script>

{#each nodes as node, index (index)}
  {#if node.type === "text"}{node.value}{:else if node.type === "break"}<br />{:else if node.type === "code"}<code>{node.value}</code>{:else if node.type === "strong"}<strong><Self nodes={node.children} /></strong>{:else if node.type === "em"}<em><Self nodes={node.children} /></em>{:else if node.type === "link"}<a href={node.href} target="_blank" rel="noopener noreferrer" onclick={(event) => open(event, node.href)}><Self nodes={node.children} /></a>{/if}
{/each}
