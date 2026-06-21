// Mascot — the abstract "June sun / solstice" mark. Replaces the wordmark.
// A mark, not a character. Three variants on the warm-light theme:
//   1 · solid disc + corona of short rays
//   2 · thin-ring sun
//   3 · first-light orb rising over a horizon
// Doubles as the global busy indicator:
//   idle = slow breathing glow; busy = rays rotate + corona pulses; settle ~420ms.
// Inline SVG + CSS keyframes so the motion is visible.

(function injectMascotCSS() {
  if (document.getElementById('june-mascot-css')) return;
  const s = document.createElement('style');
  s.id = 'june-mascot-css';
  s.textContent = `
    .june-mascot { display:block; }
    .june-mascot .rays,
    .june-mascot .rays-a,
    .june-mascot .rays-b,
    .june-mascot .disc {
      transform-box: fill-box;
      transform-origin: center;
    }
    .june-mascot .disc { transition: transform 420ms cubic-bezier(.4,0,.2,1); }

    /* idle: a big calm sun with short, slowly drifting rays */
    .june-mascot[data-state="idle"] .disc { transform: scale(1.18); }
    .june-mascot[data-state="idle"] .rays-a { animation: juneRayIdle 6200ms ease-in-out infinite; }
    .june-mascot[data-state="idle"] .rays-b { animation: juneRayIdle 6200ms ease-in-out infinite; animation-delay:-3100ms; }

    /* busy: the sun turns slowly while the rays counter-pulse */
    .june-mascot[data-state="busy"] .disc   { transform: scale(1); }
    .june-mascot[data-state="busy"] .rays   { animation: juneSpin 22s linear infinite; }
    .june-mascot[data-state="busy"] .rays-a { animation: juneRayBusy 3400ms ease-in-out infinite; }
    .june-mascot[data-state="busy"] .rays-b { animation: juneRayBusy 3400ms ease-in-out infinite; animation-delay:-1700ms; }

    @keyframes juneSpin { to { transform:rotate(360deg); } }
    @keyframes juneRayIdle { 0%,100% { transform:scale(.62); } 50% { transform:scale(.82); } }
    @keyframes juneRayBusy { 0%,100% { transform:scale(.92); } 50% { transform:scale(1.16); } }
    @media (prefers-reduced-motion: reduce) {
      .june-mascot * { animation: none !important; }
    }
  `;
  document.head.appendChild(s);
})();

function Mascot({ variant = 1, state = 'idle', size = 28, accent = '#8A5A3B' }) {
  const uid = `${size}-${String(accent).replace(/[^a-z0-9]/gi, '')}`;
  // Variant 1 rays — 12 thick, blunt, round-capped spokes (flat single fill),
  // split into two interleaved sets so they can counter-pulse.
  const ri = 10.5, ro = 17;
  const raysA = [], raysB = [];
  for (let i = 0; i < 12; i++) {
    const a = (i * 30) * Math.PI / 180;
    const ux = Math.cos(a), uy = Math.sin(a);
    const el = (
      <line key={i}
        x1={24 + ux * ri} y1={24 + uy * ri}
        x2={24 + ux * ro} y2={24 + uy * ro}
        stroke={accent} strokeWidth="3.2" strokeLinecap="round" />
    );
    (i % 2 === 0 ? raysA : raysB).push(el);
  }
  const rays8 = [];
  for (let i = 0; i < 8; i++) {
    const a = (i * 45) * Math.PI / 180;
    const r1 = 13, r2 = 16.5;
    rays8.push(
      <line key={i}
        x1={24 + Math.cos(a) * r1} y1={24 + Math.sin(a) * r1}
        x2={24 + Math.cos(a) * r2} y2={24 + Math.sin(a) * r2}
        stroke={accent} strokeWidth="1.6" strokeLinecap="round" />
    );
  }

  return (
    <svg className="june-mascot" data-state={state}
      width={size} height={size} viewBox="0 0 48 48"
      role="img" aria-label="June">
      {variant === 1 && (
        <>
          <g className="rays"><g className="rays-a">{raysA}</g><g className="rays-b">{raysB}</g></g>
          <circle className="disc" cx="24" cy="24" r="7.4" fill={accent} />
        </>
      )}
      {variant === 2 && (
        <>
          <circle className="corona" cx="24" cy="24" r="17" fill={accent} opacity="0.11" />
          <g className="rays">{rays8}</g>
          <circle className="disc" cx="24" cy="24" r="9" fill="none" stroke={accent} strokeWidth="2.2" />
          <circle cx="24" cy="24" r="2.2" fill={accent} />
        </>
      )}
      {variant === 3 && (
        <>
          <defs>
            <clipPath id={`june-horizon-${size}`}>
              <rect x="0" y="0" width="48" height="29.5" />
            </clipPath>
          </defs>
          <circle className="glow" cx="24" cy="29.5" r="15" fill={accent} opacity="0.12" />
          <g clipPath={`url(#june-horizon-${size})`}>
            <circle className="disc" cx="24" cy="29.5" r="8.5" fill={accent} />
            <g className="rays" style={{ transformOrigin: '24px 29.5px' }}>
              {[-50, -25, 0, 25, 50].map((deg, i) => {
                const a = (deg - 90) * Math.PI / 180;
                return (
                  <line key={i} className="ray"
                    x1={24 + Math.cos(a) * 11} y1={29.5 + Math.sin(a) * 11}
                    x2={24 + Math.cos(a) * 15} y2={29.5 + Math.sin(a) * 15}
                    stroke={accent} strokeWidth="1.4" strokeLinecap="round" />
                );
              })}
            </g>
          </g>
          <line x1="6" y1="29.5" x2="42" y2="29.5" stroke={accent} strokeWidth="1.5" strokeLinecap="round" />
        </>
      )}
    </svg>
  );
}

Object.assign(window, { Mascot });
