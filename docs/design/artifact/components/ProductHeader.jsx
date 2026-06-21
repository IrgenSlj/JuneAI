// ProductHeader — slim top bar shared across every surface.
// [mascot] · Tasks Memory Skills System · · · runtime·privacy  ☼/☾  ⚙

function SunMoon({ mode, color }) {
  return mode === 'dark' ? (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
      <path d="M13.5 9.5A5.5 5.5 0 016.5 2.5 5.5 5.5 0 1013.5 9.5z" stroke={color} strokeWidth="1.3" strokeLinejoin="round"/>
    </svg>
  ) : (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8" r="3.2" stroke={color} strokeWidth="1.3"/>
      {[0,45,90,135,180,225,270,315].map((d,i)=>{
        const a=d*Math.PI/180;
        return <line key={i} x1={8+Math.cos(a)*5.2} y1={8+Math.sin(a)*5.2} x2={8+Math.cos(a)*6.6} y2={8+Math.sin(a)*6.6} stroke={color} strokeWidth="1.3" strokeLinecap="round"/>;
      })}
    </svg>
  );
}

function Gear({ color }) {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8" r="2.2" stroke={color} strokeWidth="1.3"/>
      <path d="M8 1.5v2M8 12.5v2M14.5 8h-2M3.5 8h-2M12.6 3.4l-1.4 1.4M4.8 11.2l-1.4 1.4M12.6 12.6l-1.4-1.4M4.8 4.8L3.4 3.4"
        stroke={color} strokeWidth="1.3" strokeLinecap="round"/>
    </svg>
  );
}

function RuntimeStatus({ variant, mode, route = 'local' }) {
  const p = palette(variant, mode);
  const cloud = route === 'cloud';
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 9,
      fontFamily: TYPE[variant].mono, fontSize: 11.5, letterSpacing: 0,
      color: p.muted, whiteSpace: 'nowrap',
    }}>
      <StatusDot color={cloud ? p.warn : p.ok} size={6} />
      <span style={{ color: p.ink2 }}>{cloud ? 'Gemini · cloud' : 'Gemma · local'}</span>
      <span style={{ color: p.muted2 }}>·</span>
      <span style={{ color: cloud ? p.warn : p.ok }}>{cloud ? 'cloud-opt-in' : 'local-only'}</span>
    </div>
  );
}

function HeaderIconBtn({ variant, mode, onClick, title, children }) {
  const p = palette(variant, mode);
  return (
    <button onClick={onClick} title={title} style={{
      appearance: 'none', cursor: 'pointer',
      width: 30, height: 30, borderRadius: 8,
      border: `1px solid ${p.line}`, background: 'transparent',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      transition: `background ${MOTION.fast} ${MOTION.ease}`,
    }}
      onMouseEnter={e => e.currentTarget.style.background = p.surface2}
      onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
      {children}
    </button>
  );
}

function NavStrip({ variant, mode, active, onNavigate }) {
  const p = palette(variant, mode);
  const items = ['Tasks', 'Memory', 'Skills', 'System'];
  return (
    <nav style={{ display: 'flex', gap: 2 }}>
      {items.map(it => {
        const on = it.toLowerCase() === active;
        const live = true;
        return (
          <button key={it}
            onClick={() => live && onNavigate && onNavigate(it.toLowerCase())}
            style={{
              appearance: 'none', border: 'none', cursor: live ? 'pointer' : 'default',
              background: on ? p.surface2 : 'transparent',
              color: on ? p.ink : (live ? p.muted : p.muted2),
              fontFamily: TYPE[variant].family, fontSize: 13,
              padding: '5px 11px', borderRadius: 8, letterSpacing: 0,
              transition: `color ${MOTION.fast} ${MOTION.ease}`,
            }}>
            {it}
          </button>
        );
      })}
    </nav>
  );
}

function ProductHeader({
  variant, mode, active = 'chat', busy = false,
  mascotVariant = 1, route = 'local',
  onToggleMode, onNavigate, onHome,
}) {
  const p = palette(variant, mode);
  return (
    <header style={{
      display: 'flex', alignItems: 'center', gap: 18,
      padding: '0 22px', height: 56, flexShrink: 0,
      borderBottom: `1px solid ${p.line}`,
      background: p.bg,
    }}>
      <button onClick={onHome} title="June — home" style={{
        appearance: 'none', border: 'none', background: 'transparent',
        cursor: 'pointer', padding: 0, display: 'flex', alignItems: 'center',
      }}>
        <Mascot variant={mascotVariant} state={busy ? 'busy' : 'idle'} size={30} accent={p.accent} />
      </button>

      <NavStrip variant={variant} mode={mode} active={active} onNavigate={onNavigate} />

      <div style={{ flex: 1 }} />

      <RuntimeStatus variant={variant} mode={mode} route={route} />
      <div style={{ width: 1, height: 18, background: p.line, margin: '0 4px' }} />
      <HeaderIconBtn variant={variant} mode={mode} onClick={onToggleMode} title="Toggle light / dark">
        <SunMoon mode={mode} color={p.ink2} />
      </HeaderIconBtn>
      <HeaderIconBtn variant={variant} mode={mode} title="Settings" onClick={() => onNavigate && onNavigate('settings')}>
        <Gear color={p.ink2} />
      </HeaderIconBtn>
    </header>
  );
}

Object.assign(window, { ProductHeader, RuntimeStatus, NavStrip, SunMoon, Gear });
