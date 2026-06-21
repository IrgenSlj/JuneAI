// DesktopFrame — minimal window chrome around a 1280x820 screen.

function DesktopFrame({ variant, mode, children, label }) {
  const p = palette(variant, mode);
  const chrome = mode === 'dark' ? '#0A0906' : '#E8E4DC';
  return (
    <div style={{
      width: 1280, height: 820,
      borderRadius: 14, overflow: 'hidden',
      boxShadow: '0 30px 60px rgba(20,16,10,0.18), 0 2px 6px rgba(20,16,10,0.08)',
      background: chrome,
      border: `1px solid ${mode === 'dark' ? 'rgba(255,255,255,0.08)' : 'rgba(20,16,10,0.10)'}`,
      display: 'flex', flexDirection: 'column',
    }}>
      {/* traffic lights + label */}
      <div style={{
        height: 32, padding: '0 14px',
        display: 'flex', alignItems: 'center', gap: 16,
        background: chrome,
        borderBottom: `1px solid ${p.line}`,
      }}>
        <div style={{ display: 'flex', gap: 7 }}>
          {['#E0725B', '#E0B93B', '#6FB357'].map((c, i) => (
            <div key={i} style={{ width: 11, height: 11, borderRadius: 11, background: c, opacity: 0.85 }} />
          ))}
        </div>
        <div style={{
          flex: 1, textAlign: 'center',
          fontFamily: TYPE[variant].family, fontSize: 12,
          color: p.muted, letterSpacing: 0.02,
        }}>{label || 'june'}</div>
        <div style={{ width: 45 }} />
      </div>
      <div style={{ flex: 1, overflow: 'hidden', background: p.bg }}>
        {children}
      </div>
    </div>
  );
}

Object.assign(window, { DesktopFrame });
