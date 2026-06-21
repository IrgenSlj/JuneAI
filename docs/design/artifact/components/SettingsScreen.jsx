// SettingsScreen — lighter surface. The hero is THE PRIVACY DIAL.
// "Efficiency and privacy are one axis", so it's a spectrum, not a checklist:
//   Mode 1 local-only (default) · Mode 2 encrypted backup · Mode 3 Google skills.
// Local is the calm, common case on the left; reach is the visible exception right.

const MODES = [
  {
    n: 1, key: 'local', name: 'Local only', tag: 'default',
    blurb: 'Nothing leaves this machine. Chat and memory live on your disk; Gemini is unreachable. This is the calm, common case — most of what June does, she does right here.',
    points: [
      'Conversations and recall never touch the network',
      'The egress log stays empty by design',
      'Cloud isn’t off-limits — it’s simply not connected yet',
    ],
  },
  {
    n: 2, key: 'backup', name: 'Encrypted backup', tag: 'opt-in',
    blurb: 'Your whole data directory is encrypted on this device before it’s uploaded. The provider only ever holds an opaque blob — they can’t read it, and neither can we.',
    points: [
      'Client-side encryption with vetted libraries only',
      'Key lives in your OS keychain; passphrase only when moving machines',
      'Restore is copy-the-folder-back; nothing is interpreted in the cloud',
    ],
  },
  {
    n: 3, key: 'google', name: 'Google skills', tag: 'per-service',
    blurb: 'Connect Gmail, Calendar, Drive, or Maps one service at a time. Each is granted once and revocable anytime — and always visible while it’s active. Reads are wired before writes.',
    points: [
      'OAuth per service — grant exactly what you mean to',
      'Always shown in the header while a service is live',
      'Revoke any one without touching the others',
    ],
  },
];

function Dial({ variant, mode, value, onChange }) {
  const p = palette(variant, mode);
  const idx = MODES.findIndex(m => m.key === value);
  const pct = (idx / (MODES.length - 1)) * 100;
  return (
    <div style={{ padding: '12px 8px 4px' }}>
      <div style={{ position: 'relative', height: 56 }}>
        {/* track */}
        <div style={{
          position: 'absolute', left: 12, right: 12, top: 26, height: 5, borderRadius: 5,
          background: `linear-gradient(90deg, ${p.ok}44, ${p.accent}55, ${p.warn}55)`,
        }} />
        {/* filled portion to thumb */}
        <div style={{
          position: 'absolute', left: 12, top: 26, height: 5, borderRadius: 5,
          width: `calc(${pct}% * (100% - 24px) / 100%)`,
          background: 'transparent',
        }} />
        {/* nodes */}
        {MODES.map((m, i) => {
          const on = m.key === value;
          const left = `calc(12px + ${(i / (MODES.length - 1)) * 100}% - ${(i / (MODES.length - 1)) * 24}px)`;
          return (
            <button key={m.key} onClick={() => onChange(m.key)} title={m.name} style={{
              appearance: 'none', cursor: 'pointer', border: 'none', background: 'transparent',
              position: 'absolute', top: 14, left, transform: 'translateX(-50%)', padding: 0,
            }}>
              <span style={{
                display: 'block', width: on ? 28 : 16, height: on ? 28 : 16, borderRadius: 999,
                background: on ? p.accent : p.surface,
                border: `2px solid ${on ? p.accent : p.lineStrong}`,
                boxShadow: on ? `0 2px 8px ${mode === 'dark' ? 'rgba(0,0,0,0.4)' : 'rgba(20,16,10,0.18)'}` : 'none',
                transition: `all ${MOTION.base} ${MOTION.ease}`,
                position: 'relative',
              }}>
                {on && <span style={{
                  position: 'absolute', inset: 0, margin: 'auto', width: 8, height: 8,
                  borderRadius: 8, background: p.accentInk,
                }} />}
              </span>
            </button>
          );
        })}
      </div>
      {/* labels */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, padding: '0 2px' }}>
        {MODES.map(m => {
          const on = m.key === value;
          return (
            <button key={m.key} onClick={() => onChange(m.key)} style={{
              appearance: 'none', cursor: 'pointer', border: 'none', background: 'transparent',
              textAlign: 'center', flex: 1, padding: 0,
            }}>
              <div style={{
                fontFamily: TYPE[variant].family, fontSize: 13, color: on ? p.ink : p.muted,
                fontWeight: on ? 500 : 400,
              }}>{m.name}</div>
              <div style={{ fontFamily: TYPE[variant].mono, fontSize: 10.5, color: p.muted2, marginTop: 2 }}>
                mode {m.n} · {m.tag}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function QuietRow({ variant, mode, title, desc, control }) {
  const p = palette(variant, mode);
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 24,
      padding: '16px 0', borderBottom: `1px solid ${p.line}`,
    }}>
      <div>
        <div style={{ fontFamily: TYPE[variant].family, fontSize: 14.5, color: p.ink }}>{title}</div>
        <div style={{ fontFamily: TYPE[variant].family, fontSize: 13, color: p.muted, marginTop: 3, lineHeight: 1.5, maxWidth: 460 }}>{desc}</div>
      </div>
      <div style={{ flexShrink: 0 }}>{control}</div>
    </div>
  );
}

function SettingsScreen({ variant, mode, mascotVariant = 1, onToggleMode, onNavigate, onHome }) {
  const p = palette(variant, mode);
  const [pmode, setPmode] = React.useState('local');
  const [escalate, setEscalate] = React.useState(false);
  const active = MODES.find(m => m.key === pmode);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: p.bg, overflow: 'hidden' }}>
      <ProductHeader
        variant={variant} mode={mode} active="settings"
        mascotVariant={mascotVariant} route="local"
        onToggleMode={onToggleMode} onNavigate={onNavigate} onHome={onHome} />

      <div style={{ flex: 1, overflow: 'auto' }}>
        <div style={{ maxWidth: 760, margin: '0 auto', padding: '44px 32px 80px' }}>
          <div style={{ marginBottom: 30 }}>
            <div style={{
              fontFamily: TYPE[variant].family, fontSize: 28, color: p.ink,
              letterSpacing: -0.01, fontWeight: 400, marginBottom: 8,
            }}>Settings</div>
            <div style={{ fontFamily: TYPE[variant].family, fontSize: 14, color: p.muted, lineHeight: 1.6, maxWidth: 540 }}>
              One dial matters more than the rest: how far June is allowed to reach.
              You hold it, and you can move it back any time.
            </div>
          </div>

          {/* The privacy dial */}
          <div style={{
            border: `1px solid ${p.line}`, borderRadius: RADII.lg, background: p.surface,
            padding: '24px 26px 26px',
          }}>
            <div style={{
              fontFamily: TYPE[variant].family, fontSize: 11, fontWeight: 500,
              letterSpacing: 0.14, textTransform: 'uppercase', color: p.muted, marginBottom: 8,
            }}>Privacy dial</div>
            <Dial variant={variant} mode={mode} value={pmode} onChange={setPmode} />

            <div style={{
              marginTop: 22, paddingTop: 20, borderTop: `1px solid ${p.line}`,
            }}>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
                <div style={{ fontFamily: TYPE[variant].family, fontSize: 17, color: p.ink, fontWeight: 500 }}>{active.name}</div>
                <div style={{
                  fontFamily: TYPE[variant].mono, fontSize: 11, color: p.muted2,
                  border: `1px solid ${p.line}`, borderRadius: 4, padding: '1px 6px',
                }}>mode {active.n}</div>
              </div>
              <div style={{ fontFamily: TYPE[variant].family, fontSize: 14, color: p.ink2, lineHeight: 1.6, marginTop: 8 }}>{active.blurb}</div>
              <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 8 }}>
                {active.points.map((pt, i) => (
                  <div key={i} style={{ display: 'flex', gap: 10, alignItems: 'baseline' }}>
                    <span style={{ flexShrink: 0, marginTop: 1 }}>
                      <StatusDot color={pmode === 'local' ? p.ok : p.accent} size={5} />
                    </span>
                    <span style={{ fontFamily: TYPE[variant].family, fontSize: 13.5, color: p.muted, lineHeight: 1.5 }}>{pt}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* A few quiet settings */}
          <div style={{ marginTop: 40 }}>
            <QuietRow
              variant={variant} mode={mode}
              title="Reach Gemini when a task is clearly beyond local"
              desc="Even then, June shows you the call before it’s sent and after it returns. Off means she’ll tell you a task needs the cloud and wait."
              control={
                <button onClick={() => setEscalate(e => !e)} style={{
                  appearance: 'none', border: 'none', background: 'transparent', cursor: 'pointer', padding: 0,
                }}>
                  <Toggle variant={variant} mode={mode} on={escalate} />
                </button>
              } />

            <QuietRow
              variant={variant} mode={mode}
              title="Appearance"
              desc="June is built for both. Follows your system by default."
              control={
                <button onClick={onToggleMode} style={{
                  appearance: 'none', cursor: 'pointer', border: `1px solid ${p.lineStrong}`,
                  background: 'transparent', borderRadius: 9, padding: '8px 14px',
                  fontFamily: TYPE[variant].family, fontSize: 13, color: p.ink2,
                }}>{mode === 'dark' ? 'Dark' : 'Light'}</button>
              } />
          </div>

          <div style={{
            marginTop: 36, padding: '16px 18px', borderRadius: RADII.md,
            border: `1px dashed ${p.lineStrong}`, color: p.muted,
            fontFamily: TYPE[variant].family, fontSize: 13, lineHeight: 1.6,
          }}>
            <span style={{ color: p.ink2 }}>No account. No sign-in.</span>{' '}
            June is installed, not subscribed to. Everything she is lives in one folder you own —
            move machines by copying it.
          </div>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { SettingsScreen });
