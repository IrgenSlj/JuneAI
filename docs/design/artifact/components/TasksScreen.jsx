// TasksScreen — "Promises, not TODOs".
// Standing intentions the USER made: observable, resumable, never a checkbox.
// A promise is continuing / waiting on you / surfaced / let go — never done.
// Hard deadlines become ONE OS notification, surfaced as "I'll remind you once".

const PROMISE_STATES = {
  continuing:  { label: 'continuing',   dot: 'ok',     note: 'moving on its own' },
  waiting:     { label: 'waiting on you',dot: 'accent', note: 'needs a moment from you' },
  surfaced:    { label: 'surfaced',     dot: 'warn',   note: 'brought up just now' },
  letgo:       { label: 'let go',       dot: 'muted2', note: 'set aside — reversible' },
};

const PROMISES = [
  {
    id: 'essays',
    title: 'Finish the book of essays',
    state: 'continuing',
    context: 'The draft for Maren. You asked me to keep this in view without nagging you about it.',
    last: 'You added about 600 words on Tuesday evening.',
    deadline: { text: 'Draft due to Maren — Sunday the 8th', remind: true },
    open: true,
    timeline: [
      { t: '2 Apr', body: 'You said you wanted this held as a standing intention, not a task.' },
      { t: '18 Apr', body: 'Moved your 2pm so the morning stayed clear for writing.' },
      { t: 'Tue', body: 'You opened the draft and added ~600 words.' },
    ],
  },
  {
    id: 'ella',
    title: 'Call Ella back',
    state: 'waiting',
    context: 'Your sister left a voicemail Sunday. You said you’d call when you had a quiet evening — not from the car.',
    last: 'No quiet evening yet this week.',
  },
  {
    id: 'priya',
    title: 'Answer Priya about the Oviform offer',
    state: 'waiting',
    context: 'She asked for a decision, not a maybe. You wanted to sit with it over the weekend first.',
    last: 'You read it twice and closed it.',
    deadline: { text: 'She needs an answer Friday', remind: true },
  },
  {
    id: 'climb',
    title: 'Lock the climbing weekend with Leo',
    state: 'surfaced',
    context: 'You wanted dates settled before the cabins fill. I found three weekends that clear both calendars.',
    last: 'Surfaced this morning because two of the three are now half-booked.',
  },
  {
    id: 'bread',
    title: 'Learn to make Ella’s bread',
    state: 'letgo',
    context: 'You haven’t come back to this since February. I’ve set it aside so it isn’t cluttering the list.',
    last: 'Say the word and I’ll bring it back exactly as it was.',
  },
];

function PromiseState({ variant, mode, state }) {
  const p = palette(variant, mode);
  const s = PROMISE_STATES[state];
  const color = { ok: p.ok, accent: p.accent, warn: p.warn, muted2: p.muted2 }[s.dot];
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 8,
      fontFamily: TYPE[variant].family, fontSize: 12.5, color: p.muted,
    }}>
      <StatusDot color={color} size={6} />
      <span style={{ color: p.ink2 }}>{s.label}</span>
    </span>
  );
}

function RemindLine({ variant, mode, text }) {
  const p = palette(variant, mode);
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10,
      marginTop: 16, padding: '11px 14px',
      borderRadius: RADII.md, background: p.surface2,
      border: `1px solid ${p.line}`,
    }}>
      <svg width="15" height="15" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}>
        <circle cx="8" cy="8.6" r="5.4" stroke={p.warn} strokeWidth="1.3" />
        <path d="M8 5.8V8.6l1.8 1.2M8 1.6V2.8" stroke={p.warn} strokeWidth="1.3" strokeLinecap="round" />
      </svg>
      <div style={{ fontFamily: TYPE[variant].family, fontSize: 13, lineHeight: 1.5, color: p.ink2 }}>
        <span style={{ color: p.ink }}>{text}.</span>{' '}
        I’ll remind you once, at the deadline — not before, and I’m not watching the clock in between.
      </div>
    </div>
  );
}

function PromiseAction({ variant, mode, children, primary }) {
  const p = palette(variant, mode);
  return (
    <button style={{
      appearance: 'none', cursor: 'pointer',
      fontFamily: TYPE[variant].family, fontSize: 13,
      padding: '8px 14px', borderRadius: 9,
      border: `1px solid ${primary ? p.accent : p.lineStrong}`,
      background: primary ? p.accent : 'transparent',
      color: primary ? p.accentInk : p.ink2,
    }}>{children}</button>
  );
}

function PromiseCard({ variant, mode, promise, open, onToggle }) {
  const p = palette(variant, mode);
  const dimmed = promise.state === 'letgo';
  return (
    <div style={{
      border: `1px solid ${p.line}`, borderRadius: RADII.lg,
      background: p.surface, overflow: 'hidden',
      opacity: dimmed ? 0.74 : 1,
    }}>
      <button onClick={onToggle} style={{
        appearance: 'none', border: 'none', background: 'transparent',
        cursor: 'pointer', width: '100%', textAlign: 'left',
        padding: '20px 22px', display: 'block',
      }}>
        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 16 }}>
          <div style={{
            fontFamily: TYPE[variant].family, fontSize: 16.5, color: p.ink,
            fontWeight: 500, letterSpacing: 0,
            textDecoration: dimmed ? 'none' : 'none',
          }}>{promise.title}</div>
          <PromiseState variant={variant} mode={mode} state={promise.state} />
        </div>
        <div style={{
          fontFamily: TYPE[variant].family, fontSize: 14, lineHeight: 1.55,
          color: p.ink2, marginTop: 8, maxWidth: 620,
        }}>{promise.context}</div>
        <div style={{
          fontFamily: TYPE[variant].mono, fontSize: 11.5, color: p.muted2,
          marginTop: 12,
        }}>{promise.last}</div>
      </button>

      {open && (
        <div style={{ borderTop: `1px solid ${p.line}`, background: p.bg, padding: '18px 22px 20px' }}>
          {promise.timeline && (
            <div style={{ marginBottom: promise.deadline ? 4 : 0 }}>
              <div style={{
                fontFamily: TYPE[variant].family, fontSize: 11, fontWeight: 500,
                letterSpacing: 0.14, textTransform: 'uppercase', color: p.muted, marginBottom: 12,
              }}>How this has moved</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
                {promise.timeline.map((e, i) => (
                  <div key={i} style={{
                    display: 'grid', gridTemplateColumns: '64px 1fr', columnGap: 16,
                    padding: '7px 0', borderBottom: i < promise.timeline.length - 1 ? `1px solid ${p.line}` : 'none',
                  }}>
                    <span style={{ fontFamily: TYPE[variant].mono, fontSize: 11.5, color: p.muted2 }}>{e.t}</span>
                    <span style={{ fontFamily: TYPE[variant].family, fontSize: 13.5, lineHeight: 1.5, color: p.ink2 }}>{e.body}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {promise.deadline && <RemindLine variant={variant} mode={mode} text={promise.deadline.text} />}
          <div style={{ display: 'flex', gap: 10, marginTop: 18 }}>
            {promise.state === 'letgo' ? (
              <PromiseAction variant={variant} mode={mode} primary>Bring it back</PromiseAction>
            ) : (
              <>
                <PromiseAction variant={variant} mode={mode}>I’ve got this from here</PromiseAction>
                <PromiseAction variant={variant} mode={mode}>Let it go</PromiseAction>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function TasksScreen({ variant, mode, mascotVariant = 1, onToggleMode, onNavigate, onHome }) {
  const p = palette(variant, mode);
  const [openId, setOpenId] = React.useState(PROMISES.find(x => x.open)?.id || null);
  const live = PROMISES.filter(x => x.state !== 'letgo').length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: p.bg, overflow: 'hidden' }}>
      <ProductHeader
        variant={variant} mode={mode} active="tasks"
        mascotVariant={mascotVariant} route="local"
        onToggleMode={onToggleMode} onNavigate={onNavigate} onHome={onHome} />

      <div style={{ flex: 1, overflow: 'auto' }}>
        <div style={{ maxWidth: 860, margin: '0 auto', padding: '44px 32px 80px' }}>
          <div style={{
            display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 28,
          }}>
            <div>
              <div style={{
                fontFamily: TYPE[variant].family, fontSize: 28, color: p.ink,
                letterSpacing: -0.01, fontWeight: 400, marginBottom: 8,
              }}>Promises</div>
              <div style={{
                fontFamily: TYPE[variant].family, fontSize: 14, color: p.muted,
                lineHeight: 1.6, maxWidth: 560,
              }}>
                Standing intentions you’ve made — things I keep in view and quietly carry
                forward. They don’t get checked off; they continue, wait, or get let go.
                Nothing here is trying to keep you busy.
              </div>
            </div>
            <div style={{
              fontFamily: TYPE[variant].mono, fontSize: 12, color: p.muted, whiteSpace: 'nowrap',
            }}>
              <span style={{ color: p.ink }}>{live}</span> in view
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {PROMISES.map(pr => (
              <PromiseCard
                key={pr.id} variant={variant} mode={mode} promise={pr}
                open={openId === pr.id}
                onToggle={() => setOpenId(id => id === pr.id ? null : pr.id)} />
            ))}
          </div>

          <div style={{
            marginTop: 40, fontFamily: TYPE[variant].family, fontSize: 13,
            color: p.muted, textAlign: 'center', lineHeight: 1.6,
          }}>
            No streaks, no counts, no nudges to come back. A promise is yours to keep,
            change, or release.
          </div>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { TasksScreen });
