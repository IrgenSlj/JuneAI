// MobileStage — the two registers on a phone. Native-feeling, not a shrunk desktop.
// Composer sits low (thumb reach); the activity terminal is a pull-up sheet that
// the toggle on the composer's left expands over the lower half.

function MobileStage({ variant, mode, scenario = 'active', expanded = false, mascotVariant = 1 }) {
  const p = palette(variant, mode);
  const c = chrome(variant, mode);
  const data = (window.TURNS && window.TURNS[scenario]) || {};
  const thread = data.thread || [];
  const turn = data.turn || { idle: true, steps: [] };
  const mono = TYPE[variant].mono;

  return (
    <div style={{
      height: '100%', width: '100%',
      display: 'flex', flexDirection: 'column',
      background: p.bg, fontFamily: TYPE[variant].family, overflow: 'hidden',
    }}>
      {/* header */}
      <div style={{
        paddingTop: 56, paddingBottom: 12, paddingLeft: 18, paddingRight: 18,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        borderBottom: `1px solid ${p.line}`, flexShrink: 0,
      }}>
        <Mascot variant={mascotVariant} state={data.busy ? 'busy' : 'idle'} size={28} accent={p.accent} />
        <div style={{ display: 'flex', alignItems: 'center', gap: 7, fontFamily: mono, fontSize: 10.5, color: p.muted }}>
          <StatusDot color={data.route === 'cloud' ? p.warn : p.ok} size={5} />
          <span style={{ color: p.ink2 }}>{data.route === 'cloud' ? 'Gemini · cloud' : 'Gemma · local'}</span>
        </div>
      </div>

      {/* conversation */}
      <div style={{
        flex: 1, minHeight: 0, overflow: 'auto',
        display: 'flex', flexDirection: 'column', justifyContent: 'flex-end',
        padding: '0 16px',
      }}>
        {data.greeting ? (
          <div style={{ textAlign: 'center', padding: '0 8px 24px' }}>
            <div style={{ fontSize: 21, color: p.ink, lineHeight: 1.35 }}>Hi, I’m June.</div>
            <div style={{ fontSize: 15, color: p.muted, marginTop: 6, lineHeight: 1.5 }}>
              I’ll remember what matters so you don’t have to.
            </div>
          </div>
        ) : (
          <div style={{ paddingBottom: 12 }}>
            {thread.map((m, i) => (
              <Bubble key={i} variant={variant} mode={mode}
                role={m.role} text={m.text} streamTail={m.streamTail} time={m.time} />
            ))}
          </div>
        )}
      </div>

      {/* composer band */}
      <div style={{
        flexShrink: 0, padding: '10px 14px',
        borderTop: `1px solid ${p.line}`, background: p.bg,
        display: 'flex', alignItems: 'center', gap: 8,
      }}>
        <div style={{
          width: 38, height: 38, borderRadius: 10, flexShrink: 0,
          border: `1px solid ${p.line}`,
          background: expanded ? p.surface2 : 'transparent',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: expanded ? p.ink2 : p.muted,
        }}>
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{
            transform: expanded ? 'rotate(-90deg)' : 'rotate(90deg)',
          }}>
            <path d="M4 2l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
        <div style={{
          flex: 1, display: 'flex', alignItems: 'center', gap: 8,
          background: p.surface, border: `1px solid ${p.lineStrong}`,
          borderRadius: 999, padding: '7px 7px 7px 15px',
        }}>
          <div style={{ flex: 1, fontSize: 14, color: p.muted }}>
            {data.streaming ? 'June is replying…' : 'Write to June…'}
          </div>
          <div style={{
            width: 30, height: 30, borderRadius: 999, flexShrink: 0,
            background: p.accent, display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
              <path d="M8 13V3.5M4.5 7L8 3.5 11.5 7" stroke={p.accentInk} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
        </div>
      </div>

      {/* activity terminal — slim strip or pull-up sheet */}
      <div style={{
        flexShrink: 0, background: c.termBg,
        borderTop: `1px solid ${c.termLine}`,
        height: expanded ? 240 : 40,
        transition: `height ${MOTION.base} ${MOTION.ease}`,
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
      }}>
        <div style={{
          height: 40, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 10,
          padding: '0 18px', fontFamily: mono, fontSize: 11,
        }}>
          <span style={{ color: c.termDim, textTransform: 'uppercase', fontSize: 9.5, letterSpacing: 0.1 }}>activity</span>
          {!turn.idle && <span style={{ width: 5, height: 5, borderRadius: 5, background: p.accent }} />}
          <span style={{ color: turn.idle ? c.termDim : c.termInk, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {turn.idle ? 'idle' : (turn.steps[turn.steps.length - 1]?.body || 'running')}
          </span>
        </div>
        {expanded && (
          <div className="june-term-scroll" style={{ flex: 1, overflow: 'auto', padding: '2px 18px 16px' }}>
            {(turn.steps || []).map((s, i) => (
              <TermLine key={turn.id + '-' + i} variant={variant} mode={mode} step={s} idx={i} />
            ))}
          </div>
        )}
        {/* home indicator clearance */}
        {!expanded && <div style={{ height: 0 }} />}
      </div>
    </div>
  );
}

Object.assign(window, { MobileStage });
