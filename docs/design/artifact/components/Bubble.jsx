// Bubble + Conversation — the FOREGROUND register.
// Only actual exchanged messages. June left, user right. Quiet + typographic,
// not iMessage candy. A streaming reply pulses its last line.

(function injectBubbleCSS() {
  if (document.getElementById('june-bubble-css')) return;
  const s = document.createElement('style');
  s.id = 'june-bubble-css';
  s.textContent = `
    @keyframes juneLinePulse { 0%,100%{opacity:.45} 50%{opacity:1} }
    @keyframes juneCaret     { 0%,100%{opacity:.15} 50%{opacity:1} }
    .june-stream-tail { animation: juneLinePulse 1100ms cubic-bezier(.4,0,.2,1) infinite; }
    .june-caret {
      display:inline-block; width:7px; height:1.05em; margin-left:3px;
      transform:translateY(2px);
      animation: juneCaret 1100ms cubic-bezier(.4,0,.2,1) infinite;
    }
  `;
  document.head.appendChild(s);
})();

function Bubble({ variant, mode, role, text, streamTail, time }) {
  const p = palette(variant, mode);
  const c = chrome(variant, mode);
  const isJune = role === 'june';

  return (
    <div style={{
      display: 'flex',
      justifyContent: isJune ? 'flex-start' : 'flex-end',
      margin: '14px 0',
    }}>
      <div style={{ maxWidth: '76%', minWidth: 0 }}>
        <div style={{
          fontFamily: TYPE[variant].family,
          fontSize: 10.5, letterSpacing: 0.08, textTransform: 'uppercase',
          color: p.muted2, marginBottom: 6,
          textAlign: isJune ? 'left' : 'right',
          paddingLeft: isJune ? 4 : 0, paddingRight: isJune ? 0 : 4,
        }}>
          {isJune ? 'june' : 'you'}{time ? <span style={{ color: p.muted2 }}> · {time}</span> : null}
        </div>

        <div style={{
          background: isJune ? c.juneBubble : c.userBubble,
          border: `1px solid ${isJune ? c.juneBubbleLine : c.userBubbleLine}`,
          borderRadius: RADII.bubble,
          borderBottomLeftRadius: isJune ? 5 : RADII.bubble,
          borderBottomRightRadius: isJune ? RADII.bubble : 5,
          padding: '13px 17px',
          fontFamily: TYPE[variant].family,
          fontSize: 15.5, lineHeight: 1.62,
          color: isJune ? p.ink : p.ink2,
          letterSpacing: variant === 'warm' ? 0.002 : 0,
          boxShadow: isJune ? `0 1px 0 ${p.line}` : 'none',
        }}>
          {text}
          {streamTail !== undefined && (
            <>
              {text ? ' ' : ''}
              <span className="june-stream-tail">{streamTail}</span>
              <span className="june-caret" style={{ background: p.accent }} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// Conversation register — bottom-anchored so the newest reply grows just
// above the composer.
function Conversation({ variant, mode, thread, greeting, approval }) {
  const p = palette(variant, mode);

  if (greeting) {
    return (
      <div style={{
        flex: 1, minHeight: 0,
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'flex-end',
        padding: '0 32px 22px',
      }}>
        <div style={{ maxWidth: 560, textAlign: 'center', paddingBottom: 8 }}>
          <div style={{
            fontFamily: TYPE[variant].family, fontWeight: 400,
            fontSize: 26, lineHeight: 1.4, color: p.ink,
            letterSpacing: variant === 'warm' ? -0.005 : -0.01,
          }}>
            Hi, I’m June.
          </div>
          <div style={{
            fontFamily: TYPE[variant].family,
            fontSize: 18, lineHeight: 1.55, color: p.muted,
            marginTop: 8,
          }}>
            I’ll remember what matters so you don’t have to.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{
      flex: 1, minHeight: 0, overflow: 'auto',
      display: 'flex', flexDirection: 'column', justifyContent: 'flex-end',
    }}>
      <div style={{ maxWidth: 760, width: '100%', margin: '0 auto', padding: '24px 32px 14px' }}>
        {thread.map((m, i) => (
          <Bubble key={i} variant={variant} mode={mode}
            role={m.role} text={m.text} streamTail={m.streamTail} time={m.time} />
        ))}
        {approval && <ApprovalGate variant={variant} mode={mode} gate={approval} />}
      </div>
    </div>
  );
}

Object.assign(window, { Bubble, Conversation });
