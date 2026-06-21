// Shared primitives — Wordmark, StatusDot, ModelStatus

function Wordmark({ variant = 'tight', mode = 'light', size = 20 }) {
  const p = palette(variant, mode);
  const t = TYPE[variant];
  return (
    <span style={{
      fontFamily: t.wordmark,
      fontWeight: t.wordmarkWeight,
      fontSize: size,
      letterSpacing: t.wordmarkLetter,
      color: p.ink,
      fontStyle: variant === 'warm' ? 'italic' : 'normal',
      lineHeight: 1,
    }}>june</span>
  );
}

function StatusDot({ color = '#4E6B4A', size = 6 }) {
  return (
    <span style={{
      display: 'inline-block',
      width: size, height: size, borderRadius: size,
      background: color, flexShrink: 0,
      boxShadow: `0 0 0 3px ${color}22`,
    }} />
  );
}

// Thin divider
function Hair({ color, vertical = false, style = {} }) {
  return (
    <div style={{
      ...(vertical
        ? { width: 1, alignSelf: 'stretch' }
        : { height: 1, width: '100%' }),
      background: color,
      ...style,
    }} />
  );
}

// Model + reachability status pill — compact, right-aligned
function ModelStatus({ variant, mode, model = 'local', online = true, privacy = 'local-only' }) {
  const p = palette(variant, mode);
  const name = model === 'local' ? 'Gemma · local' : 'Gemini · cloud';
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10,
      fontFamily: TYPE[variant].family,
      fontSize: 12, color: p.muted, letterSpacing: 0.01,
    }}>
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
        <StatusDot color={online ? p.ok : p.muted2} size={6} />
        <span style={{ color: p.ink2 }}>{name}</span>
      </span>
      <span style={{ color: p.muted2 }}>·</span>
      <span style={{
        textTransform: 'lowercase',
        color: privacy === 'local-only' ? p.ok : p.warn,
      }}>{privacy}</span>
    </div>
  );
}

// Small quiet button
function QuietButton({ variant, mode, children, onClick, active = false, style = {} }) {
  const p = palette(variant, mode);
  return (
    <button onClick={onClick} style={{
      appearance: 'none', border: 'none', cursor: 'pointer',
      padding: '6px 10px', borderRadius: 8,
      fontFamily: TYPE[variant].family, fontSize: 13,
      color: active ? p.ink : p.muted,
      background: active ? p.surface2 : 'transparent',
      letterSpacing: 0,
      ...style,
    }}>{children}</button>
  );
}

Object.assign(window, { Wordmark, StatusDot, Hair, ModelStatus, QuietButton });
