<script lang="ts">
  // Renders inline Markdown nodes as plain elements. No {@html} anywhere.
  import type { Inline } from "./parse.js";
  import Self from "./Inline.svelte";

  const { nodes }: { nodes: Inline[] } = $props();
</script>

{#each nodes as node, index (index)}
  {#if node.type === "text"}{node.value}{:else if node.type === "break"}<br />{:else if node.type === "code"}<code>{node.value}</code>{:else if node.type === "strong"}<strong><Self nodes={node.children} /></strong>{:else if node.type === "em"}<em><Self nodes={node.children} /></em>{:else if node.type === "link"}<a href={node.href} target="_blank" rel="noopener noreferrer"><Self nodes={node.children} /></a>{/if}
{/each}
