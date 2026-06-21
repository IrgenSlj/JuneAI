// FirstRun — June is installed, not subscribed to. No account, no signup, no login.
// One calm screen: confirm the local model is ready, then "Hi, I'm June."

const READY = [
  { label: 'Found your local model', detail: 'gemma4:e2b · via Ollama', done: true },
  { label: 'Created your memory store on this device', detail: '~/June · empty, and yours', done: true },
  { label: 'Probing what it’s good at', detail: 'summarizing, recall, structure — all sharp', done: true },
  { label: 'No account needed', detail: 'nothing to sign up for — you’re already in', done: true },
];

function ReadyRow({ variant, mode, row }) {
  const p = palette(variant, mode);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '11px 0' }}>
      <span style={{
        flexShrink: 0, width: 20, height: 20, borderRadius: 999,
        border: `1.5px solid ${row.done ? p.ok : p.lineStrong}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: row.done ? p.ok : 'transparent',
      }}>
        {row.done && (
          <svg width="11" height="11" viewBox="0 0 12 12" fill="none">
            <path d="M2.5 6.2L5 8.6l4.5-5" stroke={p.bg} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        )}
      </span>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flex: 1, minWidth: 0 }}>
        <span style={{ fontFamily: TYPE[variant].family, fontSize: 14.5, color: p.ink }}>{row.label}</span>
        <span style={{ flex: 1 }} />
        <span style={{ fontFamily: TYPE[variant].mono, fontSize: 11.5, color: p.muted2, whiteSpace: 'nowrap' }}>{row.detail}</span>
      </div>
    </div>
  );
}

function FirstRun({ variant, mode, mascotVariant = 1 }) {
  const p = palette(variant, mode);
  return (
    <div style={{
      height: '100%', background: p.bg, overflow: 'hidden',
      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      padding: '0 32px',
    }}>
      <div style={{ width: '100%', maxWidth: 480 }}>
        {/* mascot + greeting */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', marginBottom: 40 }}>
          <Mascot variant={mascotVariant} state="idle" size={68} accent={p.accent} />
          <div style={{
            fontFamily: TYPE[variant].family, fontSize: 28, color: p.ink,
            letterSpacing: -0.01, marginTop: 22,
          }}>Hi, I’m June.</div>
          <div style={{
            fontFamily: TYPE[variant].family, fontSize: 17, color: p.muted,
            lineHeight: 1.55, marginTop: 8, maxWidth: 380,
          }}>I’ll remember what matters so you don’t have to. Everything I learn stays on this machine unless you ask otherwise.</div>
        </div>

        {/* readiness */}
        <div style={{
          border: `1px solid ${p.line}`, borderRadius: RADII.lg, background: p.surface,
          padding: '10px 22px 14px',
        }}>
          {READY.map((r, i) => <ReadyRow key={i} variant={variant} mode={mode} row={r} />)}
        </div>

        {/* begin */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 28 }}>
          <button style={{
            appearance: 'none', cursor: 'pointer', border: 'none',
            background: p.accent, color: p.accentInk,
            fontFamily: TYPE[variant].family, fontSize: 15, fontWeight: 500,
            padding: '13px 34px', borderRadius: 12,
            boxShadow: `0 8px 24px ${mode === 'dark' ? 'rgba(0,0,0,0.3)' : 'rgba(232,150,90,0.28)'}`,
          }}>Say hello</button>
          <div style={{
            fontFamily: TYPE[variant].family, fontSize: 12.5, color: p.muted2, marginTop: 16,
          }}>You can change how far I reach any time, in Settings.</div>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { FirstRun });
