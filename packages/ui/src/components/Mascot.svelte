<script lang="ts">
  interface Props {
    busy?: boolean;
    size?: number;
  }

  const { busy = false, size = 30 }: Props = $props();

  const ri = 10.5;
  const ro = 17;
  const cx = 24;
  const cy = 24;

  const raysA: { x1: number; y1: number; x2: number; y2: number }[] = [];
  const raysB: { x1: number; y1: number; x2: number; y2: number }[] = [];

  for (let i = 0; i < 12; i++) {
    const angle = (i * 30 * Math.PI) / 180;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const ray = {
      x1: cx + cos * ri,
      y1: cy + sin * ri,
      x2: cx + cos * ro,
      y2: cy + sin * ro,
    };
    if (i % 2 === 0) {
      raysA.push(ray);
    } else {
      raysB.push(ray);
    }
  }
</script>

<svg
  role="img"
  aria-label="June"
  width={size}
  height={size}
  viewBox="0 0 48 48"
  fill="none"
  xmlns="http://www.w3.org/2000/svg"
  class="june-mascot"
  data-state={busy ? 'busy' : 'idle'}
>
  <g class="rays">
    <g class="rays-a">
      {#each raysA as ray}
        <line
          x1={ray.x1}
          y1={ray.y1}
          x2={ray.x2}
          y2={ray.y2}
          stroke="var(--color-accent)"
          stroke-width="3.2"
          stroke-linecap="round"
        />
      {/each}
    </g>
    <g class="rays-b">
      {#each raysB as ray}
        <line
          x1={ray.x1}
          y1={ray.y1}
          x2={ray.x2}
          y2={ray.y2}
          stroke="var(--color-accent)"
          stroke-width="3.2"
          stroke-linecap="round"
        />
      {/each}
    </g>
  </g>
  <circle class="disc" cx="24" cy="24" r="7.4" fill="var(--color-accent)" />
</svg>

<style>
  .june-mascot { display: block; flex-shrink: 0; }
  .june-mascot :global(.rays),
  .june-mascot :global(.rays-a),
  .june-mascot :global(.rays-b),
  .june-mascot :global(.disc) { transform-box: fill-box; transform-origin: center; }
  .june-mascot :global(.disc) { transition: transform var(--motion-slow, 420ms) var(--ease, cubic-bezier(.4,0,.2,1)); }

  /* idle: big calm sun, slowly drifting rays */
  .june-mascot[data-state="idle"] :global(.disc) { transform: scale(1.18); }
  .june-mascot[data-state="idle"] :global(.rays-a) { animation: juneRayIdle 6200ms ease-in-out infinite; }
  .june-mascot[data-state="idle"] :global(.rays-b) { animation: juneRayIdle 6200ms ease-in-out infinite; animation-delay: -3100ms; }

  /* busy: the sun turns slowly while the rays counter-pulse */
  .june-mascot[data-state="busy"] :global(.disc) { transform: scale(1); }
  .june-mascot[data-state="busy"] :global(.rays) { animation: juneSpin 22s linear infinite; }
  .june-mascot[data-state="busy"] :global(.rays-a) { animation: juneRayBusy 3400ms ease-in-out infinite; }
  .june-mascot[data-state="busy"] :global(.rays-b) { animation: juneRayBusy 3400ms ease-in-out infinite; animation-delay: -1700ms; }

  @keyframes juneSpin { to { transform: rotate(360deg); } }
  @keyframes juneRayIdle { 0%,100% { transform: scale(.62); } 50% { transform: scale(.82); } }
  @keyframes juneRayBusy { 0%,100% { transform: scale(.92); } 50% { transform: scale(1.16); } }
  @media (prefers-reduced-motion: reduce) { .june-mascot :global(*) { animation: none !important; } }
</style>
