<script lang="ts">
  import type { Block } from "./parse.js";
  import Inline from "./Inline.svelte";
  import Self from "./Blocks.svelte";

  const { blocks }: { blocks: Block[] } = $props();
</script>

{#each blocks as block, index (index)}
  {#if block.type === "paragraph"}
    <p><Inline nodes={block.children} /></p>
  {:else if block.type === "heading"}
    <svelte:element this={`h${Math.min(block.level + 2, 6)}`}><Inline nodes={block.children} /></svelte:element>
  {:else if block.type === "code"}
    <pre data-lang={block.lang || undefined}><code>{block.value}</code></pre>
  {:else if block.type === "quote"}
    <blockquote><Self blocks={block.children} /></blockquote>
  {:else if block.type === "list"}
    {#if block.ordered}
      <ol start={block.start}>
        {#each block.items as item, i (i)}<li><Self blocks={item} /></li>{/each}
      </ol>
    {:else}
      <ul>
        {#each block.items as item, i (i)}<li><Self blocks={item} /></li>{/each}
      </ul>
    {/if}
  {:else if block.type === "table"}
    <div class="table">
      <table>
        <thead>
          <tr>{#each block.head as cell, i (i)}<th><Inline nodes={cell} /></th>{/each}</tr>
        </thead>
        <tbody>
          {#each block.rows as row, r (r)}
            <tr>{#each row as cell, i (i)}<td><Inline nodes={cell} /></td>{/each}</tr>
          {/each}
        </tbody>
      </table>
    </div>
  {:else if block.type === "rule"}
    <hr />
  {/if}
{/each}
