// MascotBoard — the June sun. One mark, shown thoroughly: a large idle/busy
// hero, then the size + light/dark ramp down to favicon.

function MascotChip({ variant, paletteMode, state, size, label }) {
  const p = palette(variant, paletteMode);
  const dark = paletteMode === 'dark';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 9 }}>
      <div style={{
        width: 92, height: 92, borderRadius: 18,
        background: p.bg, border: `1px solid ${p.line}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <Mascot variant={1} state={state} size={size} accent={p.accent} />
      </div>
      <span style={{ fontFamily: TYPE[variant].mono, fontSize: 10.5, color: dark ? '#9A8C78' : '#9A948A' }}>{label}</span>
    </div>
  );
}

function HeroPane({ variant, paletteMode, state, label }) {
  const p = palette(variant, paletteMode);
  const dark = paletteMode === 'dark';
  return (
    <div style={{
      flex: 1, borderRadius: 20, background: p.bg,
      border: `1px solid ${p.line}`,
      padding: '38px 24px 22px',
      display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 22,
    }}>
      <Mascot variant={1} state={state} size={104} accent={p.accent} />
      <div style={{ textAlign: 'center' }}>
        <div style={{
          fontFamily: TYPE[variant].family, fontSize: 13, color: p.ink2, fontWeight: 500,
          letterSpacing: 0.02, textTransform: 'capitalize',
        }}>{label}</div>
        <div style={{ fontFamily: TYPE[variant].mono, fontSize: 11, color: dark ? '#9A8C78' : '#9A948A', marginTop: 3 }}>
          {state === 'idle' ? 'calm · 6.2s' : 'rotating · 22s'}
        </div>
      </div>
    </div>
  );
}

function MascotBoard({ variant }) {
  return (
    <div style={{ padding: '8px 4px 40px' }}>
      <p style={{
        fontFamily: '"Helvetica Neue",Helvetica,Arial,sans-serif',
        fontSize: 14, lineHeight: 1.6, color: '#75675A', maxWidth: 620, margin: '0 0 30px',
      }}>
        June is the month of the solstice — the longest day, peak light. The mark is abstract:
        a warm disc and twelve soft rays. Non-gendered, scaling from header to favicon.
        It doubles as the global busy indicator — <b style={{ color: '#22190F', fontWeight: 500 }}>idle</b> is
        a big calm sun with short, slowly drifting rays; <b style={{ color: '#22190F', fontWeight: 500 }}>busy</b> turns
        slowly while the rays counter-pulse. Everything below is live.
      </p>

      {/* hero: idle vs busy, light + dark */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 30 }}>
        <HeroPane variant={variant} paletteMode="light" state="idle" label="idle" />
        <HeroPane variant={variant} paletteMode="light" state="busy" label="busy" />
        <HeroPane variant={variant} paletteMode="dark"  state="idle" label="idle · dark" />
        <HeroPane variant={variant} paletteMode="dark"  state="busy" label="busy · dark" />
      </div>

      {/* size + mode ramp */}
      <div style={{
        border: '1px solid rgba(60,40,20,0.12)', borderRadius: 16, background: '#FBF7EE',
        padding: '26px 28px',
      }}>
        <div style={{
          fontFamily: '"Helvetica Neue",Helvetica,Arial,sans-serif',
          fontSize: 11, fontWeight: 500, letterSpacing: 0.14, textTransform: 'uppercase',
          color: '#75675A', marginBottom: 20,
        }}>Scale &amp; placement</div>
        <div style={{ display: 'flex', gap: 26, flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <MascotChip variant={variant} paletteMode="light" state="busy" size={48} label="nav · 48" />
          <MascotChip variant={variant} paletteMode="light" state="busy" size={30} label="header · 30" />
          <MascotChip variant={variant} paletteMode="light" state="idle" size={16} label="favicon · 16" />
          <div style={{ width: 1, alignSelf: 'stretch', background: 'rgba(60,40,20,0.1)' }} />
          <MascotChip variant={variant} paletteMode="dark"  state="busy" size={30} label="header · dark" />
          <MascotChip variant={variant} paletteMode="dark"  state="idle" size={16} label="favicon · dark" />
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { MascotBoard });
