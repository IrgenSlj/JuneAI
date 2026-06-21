// TokensView — the named design tokens, exported to packages/design/src/tokens.ts.
// Color · type · radii · and the motion scale (durations + easing).

function TokSwatch({ c, name }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 9, minWidth: 150 }}>
      <div style={{ width: 26, height: 26, borderRadius: 6, background: c, border: '1px solid rgba(0,0,0,0.10)' }} />
      <div>
        <div style={{ fontFamily: '"Helvetica Neue",Helvetica,Arial,sans-serif', fontSize: 12, color: '#22190F' }}>{name}</div>
        <div style={{ fontFamily: 'ui-monospace,Menlo,monospace', fontSize: 10.5, color: '#A59787' }}>{c}</div>
      </div>
    </div>
  );
}

function TokSection({ title, children }) {
  return (
    <div style={{ marginBottom: 34 }}>
      <div style={{
        fontFamily: '"Helvetica Neue",Helvetica,Arial,sans-serif',
        fontSize: 11, fontWeight: 500, letterSpacing: 0.14, textTransform: 'uppercase',
        color: '#75675A', marginBottom: 14,
      }}>{title}</div>
      {children}
    </div>
  );
}

function TokensView({ variant }) {
  const keys = ['bg', 'surface', 'surface2', 'ink', 'ink2', 'muted', 'accent', 'ok', 'warn', 'err'];
  const rows = [
    { label: `${variant} · light`, p: window.TOKENS[variant].light },
    { label: `${variant} · dark`,  p: window.TOKENS[variant].dark  },
  ];
  const type = [
    { n: 'Display', s: 26, w: 400 },
    { n: 'Title',   s: 22, w: 500 },
    { n: 'Body',    s: 15.5, w: 400 },
    { n: 'Bubble',  s: 15.5, w: 400 },
    { n: 'Label',   s: 11, w: 500 },
    { n: 'Mono / activity', s: 11.5, w: 400, mono: true },
  ];
  const motion = [
    ['fast', MOTION.fast, 'hover, toggle thumb'],
    ['base', MOTION.base, 'terminal collapse / expand, tabs'],
    ['enter', MOTION.enter, 'activity line fade + slide in'],
    ['slow', MOTION.slow, 'mascot settle-to-rest'],
    ['breath', MOTION.breath, 'mascot idle breathing'],
    ['spin', MOTION.spin, 'mascot busy ray rotation'],
    ['pulse', MOTION.pulse, 'streaming last-line pulse'],
  ];

  return (
    <div style={{ padding: '8px 4px 40px', maxWidth: 760 }}>
      <TokSection title="Color">
        {rows.map(r => (
          <div key={r.label} style={{ marginBottom: 16 }}>
            <div style={{ fontFamily: '"Helvetica Neue",Helvetica,Arial,sans-serif', fontSize: 12, color: '#3A2E20', marginBottom: 10 }}>{r.label}</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
              {keys.map(k => <TokSwatch key={k} c={r.p[k]} name={k} />)}
            </div>
          </div>
        ))}
      </TokSection>

      <TokSection title="Type — Helvetica Neue, one mono">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {type.map(t => (
            <div key={t.n} style={{ display: 'flex', alignItems: 'baseline', gap: 16 }}>
              <span style={{ width: 130, fontFamily: 'ui-monospace,Menlo,monospace', fontSize: 11, color: '#A59787' }}>{t.n} · {t.s}</span>
              <span style={{
                fontFamily: t.mono ? 'ui-monospace,Menlo,monospace' : '"Helvetica Neue",Helvetica,Arial,sans-serif',
                fontSize: t.s, fontWeight: t.w, color: '#22190F',
              }}>The quiet sun remembers</span>
            </div>
          ))}
        </div>
      </TokSection>

      <TokSection title="Radii">
        <div style={{ display: 'flex', gap: 18 }}>
          {Object.entries(RADII).map(([k, v]) => (
            <div key={k} style={{ textAlign: 'center' }}>
              <div style={{ width: 54, height: 40, background: '#EDE6D8', borderRadius: v, border: '1px solid rgba(60,40,20,0.12)' }} />
              <div style={{ fontFamily: 'ui-monospace,Menlo,monospace', fontSize: 10.5, color: '#A59787', marginTop: 6 }}>{k} · {v}</div>
            </div>
          ))}
        </div>
      </TokSection>

      <TokSection title="Motion scale — durations + easing">
        <div style={{ border: '1px solid rgba(60,40,20,0.12)', borderRadius: 10, overflow: 'hidden' }}>
          {motion.map(([k, v, d], i) => (
            <div key={k} style={{
              display: 'grid', gridTemplateColumns: '120px 90px 1fr',
              padding: '9px 14px', alignItems: 'center',
              background: i % 2 ? '#FBF7EE' : '#F4EFE6',
              fontFamily: 'ui-monospace,Menlo,monospace', fontSize: 11.5,
            }}>
              <span style={{ color: '#22190F' }}>{k}</span>
              <span style={{ color: '#7A4A2A' }}>{v}</span>
              <span style={{ color: '#75675A', fontFamily: '"Helvetica Neue",Helvetica,Arial,sans-serif' }}>{d}</span>
            </div>
          ))}
          <div style={{
            padding: '9px 14px', background: '#F4EFE6',
            fontFamily: 'ui-monospace,Menlo,monospace', fontSize: 11.5,
            borderTop: '1px solid rgba(60,40,20,0.10)',
          }}>
            <span style={{ color: '#22190F' }}>ease</span>
            <span style={{ color: '#7A4A2A', marginLeft: 24 }}>{MOTION.ease}</span>
          </div>
        </div>
      </TokSection>
    </div>
  );
}

Object.assign(window, { TokensView });
