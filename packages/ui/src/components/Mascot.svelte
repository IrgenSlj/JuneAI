<script lang="ts">
  interface Props {
    busy?: boolean;
    size?: number;
  }

  const { busy = false, size = 28 }: Props = $props();

  // 8 rays evenly spaced at 45° increments
  const rays = Array.from({ length: 8 }, (_, i) => {
    const angle = (i * 45 * Math.PI) / 180;
    const innerR = 7;
    const outerR = 11;
    const cx = 16;
    const cy = 16;
    return {
      x1: cx + innerR * Math.cos(angle),
      y1: cy + innerR * Math.sin(angle),
      x2: cx + outerR * Math.cos(angle),
      y2: cy + outerR * Math.sin(angle),
    };
  });
</script>

<svg
  aria-hidden="true"
  width={size}
  height={size}
  viewBox="0 0 32 32"
  fill="none"
  xmlns="http://www.w3.org/2000/svg"
  class="mascot"
  class:busy
>
  <!-- Corona / rays group -->
  <g class="corona">
    {#each rays as ray}
      <line
        x1={ray.x1}
        y1={ray.y1}
        x2={ray.x2}
        y2={ray.y2}
        stroke="color-mix(in srgb, var(--color-accent) 55%, transparent)"
        stroke-width="1.8"
        stroke-linecap="round"
      />
    {/each}
  </g>
  <!-- Central disc -->
  <circle
    class="disc"
    cx="16"
    cy="16"
    r="5"
    fill="var(--color-accent)"
  />
</svg>

<style>
  .mascot {
    display: block;
    flex-shrink: 0;
    /* transition between idle/busy states */
    transition: opacity 350ms ease;
  }

  /* ── Idle: gentle corona breathing ── */
  .corona {
    transform-origin: 16px 16px;
    animation: corona-breathe 5s ease-in-out infinite;
  }

  /* ── Busy: corona rotates + pulse ── */
  .busy .corona {
    animation: corona-spin 8s linear infinite;
  }

  .busy .disc {
    animation: disc-pulse 2s ease-in-out infinite;
  }

  /* ── Keyframes ── */
  @keyframes corona-breathe {
    0%, 100% { opacity: 0.55; }
    50%       { opacity: 1;    }
  }

  @keyframes corona-spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
  }

  @keyframes disc-pulse {
    0%, 100% { opacity: 1;    }
    50%       { opacity: 0.7; }
  }

  /* ── Reduced-motion: disable everything ── */
  @media (prefers-reduced-motion: reduce) {
    .corona,
    .disc {
      animation: none !important;
    }
  }
</style>
