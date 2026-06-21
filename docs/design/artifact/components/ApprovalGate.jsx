// ApprovalGate — the "defers, not verifies" inversion as a UI moment.
// An INLINE card in the conversation register (never a modal). It fires when June
// wants a consequential / exfiltration-shaped action: any local/network write or
// execute, or a network read whose args are tainted by a prior tool result.
// A calm checkpoint — but the TAINTED case is visually unmistakable.

function GateGlyph({ tainted, color }) {
  return tainted ? (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <path d="M8 1.6l6.2 11.2H1.8L8 1.6z" stroke={color} strokeWidth="1.3" strokeLinejoin="round" />
      <path d="M8 6.2v3.2M8 11.2v.6" stroke={color} strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  ) : (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <rect x="3.2" y="7" width="9.6" height="7" rx="1.6" stroke={color} strokeWidth="1.3" />
      <path d="M5.4 7V5.2a2.6 2.6 0 015.2 0V7" stroke={color} strokeWidth="1.3" />
    </svg>
  );
}

function GateButton({ variant, mode, children, kind, onClick }) {
  const p = palette(variant, mode);
  const styles = {
    approve: { bg: p.accent, fg: p.accentInk, bd: p.accent },
    deny:    { bg: 'transparent', fg: p.ink2, bd: p.lineStrong },
    always:  { bg: 'transparent', fg: p.muted, bd: p.line },
  }[kind];
  return (
    <button onClick={onClick} style={{
      appearance: 'none', cursor: 'pointer',
      fontFamily: TYPE[variant].family, fontSize: 13.5,
      padding: '9px 16px', borderRadius: 10,
      border: `1px solid ${styles.bd}`, background: styles.bg, color: styles.fg,
      fontWeight: kind === 'approve' ? 500 : 400,
    }}>{children}</button>
  );
}

function ApprovalGate({ variant, mode, gate }) {
  const p = palette(variant, mode);
  const c = chrome(variant, mode);
  const [choice, setChoice] = React.useState(null); // null | 'approve' | 'deny'
  const tainted = gate.tainted;
  const edge = tainted ? p.err : p.accent;
  // tainted network writes NEVER offer "always allow"
  const allowAlways = !tainted;

  return (
    <div style={{ display: 'flex', justifyContent: 'flex-start', margin: '18px 0 6px' }}>
      <div style={{ maxWidth: '88%', width: 560 }}>
        <div style={{
          fontFamily: TYPE[variant].family, fontSize: 10.5, letterSpacing: 0.08,
          textTransform: 'uppercase', color: tainted ? p.err : p.muted2,
          marginBottom: 6, paddingLeft: 4,
        }}>{tainted ? 'June paused — read this' : 'June is asking first'}</div>

        <div style={{
          border: `1px solid ${tainted ? p.err : p.lineStrong}`,
          borderLeft: `3px solid ${edge}`,
          borderRadius: RADII.lg,
          background: p.surface,
          overflow: 'hidden',
          boxShadow: tainted ? `0 6px 24px ${mode === 'dark' ? 'rgba(0,0,0,0.4)' : 'rgba(138,59,59,0.14)'}` : `0 1px 0 ${p.line}`,
        }}>
          {/* header */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 10,
            padding: '14px 18px 0',
          }}>
            <span style={{ color: edge, display: 'flex' }}><GateGlyph tainted={tainted} color={edge} /></span>
            <span style={{ fontFamily: TYPE[variant].family, fontSize: 15.5, color: p.ink, fontWeight: 500 }}>{gate.title}</span>
          </div>

          {/* why */}
          <div style={{
            fontFamily: TYPE[variant].family, fontSize: 14, color: p.ink2,
            lineHeight: 1.6, padding: '8px 18px 0',
          }}>{gate.why}</div>

          {/* exact payload / target */}
          <div style={{
            margin: '14px 18px 0', borderRadius: RADII.md,
            background: c.termBg, border: `1px solid ${c.termLine}`,
            padding: '12px 14px',
            fontFamily: TYPE[variant].mono, fontSize: 12, lineHeight: 1.7, color: c.termInkStrong,
          }}>
            <div><span style={{ color: c.termDim }}>action&nbsp;&nbsp;</span>{gate.action}</div>
            <div><span style={{ color: c.termDim }}>target&nbsp;&nbsp;</span>{gate.target}</div>
            <div style={{ color: c.termInk, whiteSpace: 'pre-wrap', marginTop: 4 }}>
              <span style={{ color: c.termDim }}>payload&nbsp;</span>{gate.payload}
            </div>
          </div>

          {/* taint flag — the highest-attention case */}
          {tainted && (
            <div style={{
              margin: '12px 18px 0', borderRadius: RADII.md,
              background: p.accentSoft, border: `1px solid ${p.err}55`,
              padding: '11px 14px', display: 'flex', gap: 10, alignItems: 'flex-start',
            }}>
              <span style={{ color: p.err, flexShrink: 0, marginTop: 1 }}><GateGlyph tainted color={p.err} /></span>
              <div style={{ fontFamily: TYPE[variant].family, fontSize: 13, color: p.ink2, lineHeight: 1.55 }}>
                <span style={{ color: p.err, fontWeight: 500 }}>This carries content from a web page I read earlier this turn.</span>{' '}
                {gate.taintNote} Sending it outward is exactly the shape data exfiltration takes — so I won’t offer to remember this answer.
              </div>
            </div>
          )}

          {/* buttons / settled state */}
          <div style={{ padding: '16px 18px 18px' }}>
            {!choice ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                <GateButton variant={variant} mode={mode} kind="approve" onClick={() => setChoice('approve')}>Approve</GateButton>
                <GateButton variant={variant} mode={mode} kind="deny" onClick={() => setChoice('deny')}>Deny</GateButton>
                {allowAlways && (
                  <GateButton variant={variant} mode={mode} kind="always" onClick={() => setChoice('approve')}>Always allow this in this conversation</GateButton>
                )}
              </div>
            ) : (
              <div style={{
                display: 'flex', alignItems: 'center', gap: 9,
                fontFamily: TYPE[variant].mono, fontSize: 12,
                color: choice === 'approve' ? p.ok : p.muted,
              }}>
                <StatusDot color={choice === 'approve' ? p.ok : p.muted2} size={6} />
                {choice === 'approve'
                  ? 'Approved — going ahead. Recorded in activity below.'
                  : 'Denied — I’ll leave it. Recorded in activity below.'}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { ApprovalGate });
