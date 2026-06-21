// ActivityTerminal — the BACKGROUND register. Everything June DID to reply.
// Subdued, monospace, clearly beneath the conversation. Collapsed = a slim
// one-line strip (latest step); expanded = a scrollable real-time log.
// The cloud-boundary / provenance line is the visual ANCHOR OF TRUST.

(function injectTermCSS() {
  if (document.getElementById('june-term-css')) return;
  const s = document.createElement('style');
  s.id = 'june-term-css';
  s.textContent = `
    @keyframes juneLineIn {
      from { opacity:0; transform:translateY(4px); }
      to   { opacity:1; transform:translateY(0); }
    }
    .june-term-line { opacity:1; animation: juneLineIn 320ms cubic-bezier(0,0,.2,1) forwards; }
    .june-term-scroll::-webkit-scrollbar { width:8px; }
    .june-term-scroll::-webkit-scrollbar-thumb { background:rgba(128,128,128,.25); border-radius:8px; }
  `;
  document.head.appendChild(s);
})();

function BoundaryGlyph({ cloud, color }) {
  return cloud ? (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
      <path d="M4.5 11.5a2.5 2.5 0 010-5 3.3 3.3 0 016.4-1 2.6 2.6 0 011.6 4.6" stroke={color} strokeWidth="1.2" strokeLinejoin="round"/>
      <path d="M8 13V8.2M6 9.8L8 7.8l2 2" stroke={color} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ) : (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
      <path d="M8 1.8l5 1.9v3.4c0 3-2.1 5.4-5 6.1-2.9-.7-5-3.1-5-6.1V3.7L8 1.8z" stroke={color} strokeWidth="1.2" strokeLinejoin="round"/>
      <path d="M5.8 8.1L7.3 9.6l3-3.4" stroke={color} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

function TermLine({ variant, mode, step, idx }) {
  const p = palette(variant, mode);
  const c = chrome(variant, mode);
  const mono = TYPE[variant].mono;
  const delay = `${Math.min(idx, 8) * 70}ms`;

  // The provenance / cloud-boundary line — the anchor of trust.
  if (step.kind === 'boundary') {
    const accent = step.cloud ? p.warn : p.ok;
    return (
      <div className="june-term-line" style={{
        animationDelay: delay,
        display: 'flex', alignItems: 'center', gap: 10,
        margin: '7px 0', padding: '8px 11px',
        borderLeft: `2px solid ${accent}`,
        background: step.cloud ? p.accentSoft : (mode === 'dark' ? 'rgba(138,168,132,0.08)' : 'rgba(78,107,74,0.07)'),
        borderRadius: 5,
        fontFamily: mono, fontSize: 11.5, letterSpacing: 0,
      }}>
        <span style={{ display: 'flex', alignItems: 'center', color: accent }}>
          <BoundaryGlyph cloud={step.cloud} color={accent} />
        </span>
        <span style={{ color: accent, fontWeight: 500, textTransform: 'uppercase', letterSpacing: 0.04 }}>
          {step.cloud ? 'cloud' : 'local'}
        </span>
        <span style={{ color: c.termInkStrong }}>{step.model}</span>
        <span style={{ color: c.termDim }}>·</span>
        <span style={{ color: step.cloud ? p.warn : p.ok }}>{step.sent}</span>
        <span style={{ flex: 1 }} />
        <span style={{ color: c.termDim }}>{step.timing}</span>
      </div>
    );
  }

  // Action gate — the loop paused for the user (defers, not verifies).
  if (step.kind === 'gate') {
    const accent = step.tainted ? p.err : p.warn;
    return (
      <div className="june-term-line" style={{
        animationDelay: delay,
        display: 'flex', alignItems: 'center', gap: 10,
        margin: '7px 0', padding: '8px 11px',
        borderLeft: `2px solid ${accent}`,
        background: step.tainted ? p.accentSoft : (mode === 'dark' ? 'rgba(200,162,96,0.08)' : 'rgba(138,106,47,0.07)'),
        borderRadius: 5, fontFamily: mono, fontSize: 11.5, letterSpacing: 0,
      }}>
        <span style={{ color: accent, fontWeight: 500, textTransform: 'uppercase', letterSpacing: 0.04 }}>gate</span>
        <span style={{ color: c.termInkStrong }}>{step.body}</span>
        {step.tainted && <span style={{ color: p.err }}>· tainted</span>}
        <span style={{ flex: 1 }} />
        <span style={{ color: c.termDim }}>awaiting you</span>
      </div>
    );
  }

  // Optional future "reasoning" slot — clearly a placeholder, never fabricated.
  if (step.kind === 'reasoning') {
    return (
      <div className="june-term-line" style={{
        animationDelay: delay,
        display: 'grid', gridTemplateColumns: '72px 64px 1fr', columnGap: 14,
        padding: '3px 0', alignItems: 'baseline',
        fontFamily: mono, fontSize: 11.5, color: c.termDim,
      }}>
        <span>{step.t}</span>
        <span style={{ color: c.termDim, fontStyle: 'italic' }}>reasoning</span>
        <span style={{
          fontStyle: 'italic', color: c.termDim,
          borderBottom: `1px dashed ${c.termLine}`, paddingBottom: 1, justifySelf: 'start',
        }}>not exposed in this build</span>
      </div>
    );
  }

  // Standard log line: time · kind · body
  const kindColor = {
    recall: p.accent,
    route:  c.termInkStrong,
    tool:   c.termInkStrong,
    result: c.termDim,
    done:   c.termDim,
  }[step.kind] || c.termInk;

  return (
    <div className="june-term-line" style={{
      animationDelay: delay,
      display: 'grid', gridTemplateColumns: '72px 64px 1fr', columnGap: 14,
      padding: '3px 0', alignItems: 'baseline',
      fontFamily: mono, fontSize: 11.5, color: c.termInk, letterSpacing: 0,
    }}>
      <span style={{ color: c.termDim }}>{step.t}</span>
      <span style={{ color: kindColor }}>
        {step.kind === 'result' ? '' : step.kind}
      </span>
      <span style={{ color: step.kind === 'result' ? c.termInk : c.termInk }}>
        {step.kind === 'result' && <span style={{ color: c.termDim }}>→ </span>}
        {step.body}
      </span>
    </div>
  );
}

function ActivityTerminal({ variant, mode, expanded, turn }) {
  const p = palette(variant, mode);
  const c = chrome(variant, mode);
  const mono = TYPE[variant].mono;
  const steps = turn.steps || [];
  const latest = steps[steps.length - 1];

  const latestSummary = (() => {
    if (turn.idle || !latest) return 'idle — nothing running';
    if (latest.kind === 'boundary') return `${latest.cloud ? 'cloud' : 'local'} · ${latest.model} · ${latest.sent}`;
    if (latest.kind === 'gate') return `action gate · ${latest.body} · awaiting you`;
    if (latest.kind === 'done') return latest.body;
    return `${latest.kind}${latest.body ? ' · ' + latest.body : ''}`;
  })();

  return (
    <div style={{
      flexShrink: 0,
      borderTop: `1px solid ${c.termLine}`,
      background: c.termBg,
      display: 'flex', flexDirection: 'column',
      height: expanded ? undefined : 40,
      flex: expanded ? 1 : 'none',
      minHeight: expanded ? 140 : 40,
      transition: `min-height ${MOTION.base} ${MOTION.ease}`,
      overflow: 'hidden',
    }}>
      {/* strip header — always visible */}
      <div style={{
        height: 40, flexShrink: 0,
        display: 'flex', alignItems: 'center', gap: 12,
        padding: '0 22px',
        fontFamily: mono, fontSize: 11.5,
      }}>
        <span style={{
          color: c.termDim, textTransform: 'uppercase', letterSpacing: 0.1, fontSize: 10,
        }}>activity</span>
        {!turn.idle && (
          <span style={{
            width: 5, height: 5, borderRadius: 5,
            background: turn.live ? p.accent : c.termDim,
            animation: turn.live ? 'juneCaret 1100ms ease-in-out infinite' : 'none',
          }} />
        )}
        <span style={{
          color: turn.idle ? c.termDim : c.termInk,
          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
        }}>{latestSummary}</span>
        <span style={{ flex: 1 }} />
        {!turn.idle && (
          <span style={{ color: c.termDim }}>{steps.length} steps</span>
        )}
      </div>

      {/* expanded log */}
      {expanded && (
        <div className="june-term-scroll" style={{
          flex: 1, overflow: 'auto',
          padding: '4px 22px 18px',
        }}>
          {turn.idle ? (
            <div style={{
              fontFamily: mono, fontSize: 11.5, color: c.termDim,
              padding: '8px 0',
            }}>
              Waiting. When you send a message, June’s steps stream here — what she
              recalled, where she routed, every tool she touched, and exactly what
              (if anything) left your device.
            </div>
          ) : (
            steps.map((s, i) => (
              <TermLine key={turn.id + '-' + i} variant={variant} mode={mode} step={s} idx={i} />
            ))
          )}
        </div>
      )}
    </div>
  );
}

Object.assign(window, { ActivityTerminal, TermLine });
