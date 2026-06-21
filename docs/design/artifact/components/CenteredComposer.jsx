// CenteredComposer — the fulcrum of the screen. Vertically centered band.
// [‹ activity toggle] · [ text input  ⏎ ] · [stop while streaming]
// The toggle on the LEFT collapses/expands the activity terminal below.

function ActivityToggle({ variant, mode, expanded, onClick, hasActivity, size = 50 }) {
  const p = palette(variant, mode);
  return (
    <button onClick={onClick} title={expanded ? 'Hide activity' : 'Show activity'} style={{
      appearance: 'none', cursor: 'pointer', flexShrink: 0,
      width: size, height: size, borderRadius: RADII.lg,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      position: 'relative',
      border: `1px solid ${p.lineStrong}`,
      background: expanded ? p.surface2 : p.surface,
      color: expanded ? p.ink2 : p.muted,
      boxShadow: `0 1px 0 ${p.line}, 0 8px 24px ${mode === 'dark' ? 'rgba(0,0,0,0.25)' : 'rgba(20,16,10,0.05)'}`,
      transition: `background ${MOTION.fast} ${MOTION.ease}, color ${MOTION.fast} ${MOTION.ease}`,
    }}>
      {/* stacked-lines glyph = the activity log; chevron shows open/close direction */}
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{
        transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
        transition: `transform ${MOTION.base} ${MOTION.ease}`,
      }}>
        <path d="M8 2.5v7M5 6.5l3 3 3-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M3 13h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
      {hasActivity && !expanded && (
        <span style={{
          position: 'absolute', top: 8, right: 8,
          width: 5, height: 5, borderRadius: 5, background: p.accent,
        }} />
      )}
    </button>
  );
}

function CenteredComposer({
  variant, mode, streaming = false,
  expanded, onToggleActivity, hasActivity,
}) {
  const p = palette(variant, mode);
  return (
    <div style={{
      flexShrink: 0,
      borderTop: `1px solid ${p.line}`,
      borderBottom: `1px solid ${p.line}`,
      background: p.bg,
      padding: '16px 0',
    }}>
      <div style={{
        maxWidth: 820, margin: '0 auto', padding: '0 32px',
        display: 'flex', alignItems: 'flex-start', gap: 12,
      }}>
        <ActivityToggle
          variant={variant} mode={mode}
          expanded={expanded} onClick={onToggleActivity} hasActivity={hasActivity} />

        <div style={{ flex: 1 }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 10,
            background: p.surface,
            border: `1px solid ${p.lineStrong}`,
            borderRadius: RADII.lg,
            padding: '0 8px 0 16px',
            minHeight: 50,
            boxShadow: `0 1px 0 ${p.line}, 0 8px 24px ${mode === 'dark' ? 'rgba(0,0,0,0.25)' : 'rgba(20,16,10,0.05)'}`,
          }}>
            <div style={{
              flex: 1,
              fontFamily: TYPE[variant].family, fontSize: 15.5,
              color: p.muted, lineHeight: 1.4, padding: '14px 0',
            }}>
              {streaming ? 'June is replying…' : 'Write to June…'}
            </div>

            {streaming && (
              <button style={{
                appearance: 'none', cursor: 'pointer',
                border: `1px solid ${p.lineStrong}`, background: 'transparent',
                color: p.ink2, fontFamily: TYPE[variant].family, fontSize: 12.5,
                padding: '7px 13px', borderRadius: 9,
              }}>Stop</button>
            )}

            <button title="Send" style={{
              appearance: 'none', cursor: 'pointer', border: 'none',
              background: p.accent, color: p.accentInk,
              width: 38, height: 38, borderRadius: 10,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
                <path d="M8 13V3.5M4.5 7L8 3.5 11.5 7" stroke={p.accentInk} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </div>

          <div style={{
            marginTop: 8, paddingLeft: 4,
            display: 'flex', alignItems: 'center', gap: 8,
            fontFamily: TYPE[variant].mono, fontSize: 11, color: p.muted2,
          }}>
            <span style={{
              border: `1px solid ${p.line}`, padding: '1px 6px', borderRadius: 4, color: p.muted,
            }}>⌘ ⏎</span>
            <span>to send</span>
            {streaming && <><span style={{ color: p.muted2 }}>·</span><span>Esc to stop</span></>}
          </div>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { CenteredComposer, ActivityToggle });
