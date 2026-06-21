// SkillsScreen — list of skill cards.

const SKILLS = [
  {
    name: 'Calendar',
    desc: 'Watches your calendar, writes drafts of replies, protects focus time.',
    status: 'running',
    enabled: true,
    tools: [
      { n: 'calendar.peek',     d: 'read upcoming events'},
      { n: 'calendar.propose',  d: 'draft a time, never send'},
      { n: 'calendar.reschedule', d: 'move an event you own'},
      { n: 'focus.protect',     d: 'decline during a block'},
    ],
  },
  {
    name: 'Health',
    desc: 'Reads Apple Health and your gym log. Never volunteers stats.',
    status: 'running',
    enabled: true,
    tools: [
      { n: 'health.summary', d: 'last 7 days, only when asked' },
      { n: 'gym.log',        d: 'lift + cardio entries' },
    ],
  },
  {
    name: 'Research',
    desc: 'Reads papers and long articles. Answers from what it actually read.',
    status: 'running',
    enabled: true,
    tools: [
      { n: 'web.fetch',  d: 'one URL at a time, cached locally' },
      { n: 'paper.read', d: 'PDFs dropped into ~/June/Papers' },
      { n: 'notes.cite', d: 'links back to the source line' },
    ],
  },
  {
    name: 'Files',
    desc: 'A quiet index of your Documents. It will not write without permission.',
    status: 'crashed',
    enabled: true,
    tools: [
      { n: 'files.find',  d: 'fuzzy search your filesystem' },
      { n: 'files.read',  d: 'open a text or markdown file' },
    ],
    error: "Index rebuild failed after OS upgrade. Last healthy: 14 Apr, 21:04.",
  },
  {
    name: 'Daily',
    desc: 'A short morning brief. Three things, no more. You can turn any of them off.',
    status: 'stopped',
    enabled: false,
    tools: [
      { n: 'daily.compose', d: 'one message at 8:00 local' },
      { n: 'daily.tune',    d: 'learn what you skim past' },
    ],
  },
];

function StatusBadge({ variant, mode, status }) {
  const p = palette(variant, mode);
  const map = {
    running:  { c: p.ok,     l: 'running' },
    stopped:  { c: p.muted2, l: 'stopped' },
    crashed:  { c: p.err,    l: 'needs attention' },
    disabled: { c: p.muted2, l: 'disabled' },
  }[status];
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 7,
      fontFamily: TYPE[variant].family, fontSize: 12,
      color: p.muted, letterSpacing: 0.01,
    }}>
      <StatusDot color={map.c} size={6} />
      {map.l}
    </span>
  );
}

function Toggle({ variant, mode, on }) {
  const p = palette(variant, mode);
  return (
    <div style={{
      width: 34, height: 20, borderRadius: 999,
      background: on ? p.accent : p.surface2,
      border: `1px solid ${on ? p.accent : p.lineStrong}`,
      position: 'relative', transition: 'all 120ms',
      flexShrink: 0,
    }}>
      <div style={{
        position: 'absolute', top: 1, left: on ? 15 : 1,
        width: 16, height: 16, borderRadius: 16,
        background: on ? p.accentInk : p.surface,
        boxShadow: '0 1px 2px rgba(0,0,0,0.12)',
        transition: 'left 120ms',
      }} />
    </div>
  );
}

function SkillCard({ variant, mode, skill, expanded, onToggleExpand }) {
  const p = palette(variant, mode);
  const dimmed = skill.status === 'stopped' || skill.status === 'disabled';
  return (
    <div style={{
      border: `1px solid ${p.line}`,
      borderRadius: RADII.lg,
      background: p.surface,
      overflow: 'hidden',
      opacity: dimmed ? 0.78 : 1,
    }}>
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr auto auto',
        columnGap: 20, rowGap: 6,
        padding: '20px 22px',
        alignItems: 'center',
      }}>
        <div style={{
          fontFamily: TYPE[variant].family, fontSize: 16,
          color: p.ink, fontWeight: 500, letterSpacing: 0,
        }}>{skill.name}</div>
        <StatusBadge variant={variant} mode={mode} status={skill.status} />
        <Toggle variant={variant} mode={mode} on={skill.enabled} />

        <div style={{
          gridColumn: '1 / 4',
          fontFamily: TYPE[variant].family, fontSize: 14,
          color: p.ink2, lineHeight: 1.55,
          marginTop: -2,
        }}>{skill.desc}</div>

        {skill.error && (
          <div style={{
            gridColumn: '1 / 4', marginTop: 10,
            padding: '10px 12px',
            borderLeft: `2px solid ${p.err}`,
            background: p.surface2,
            borderRadius: 4,
            fontFamily: TYPE[variant].family, fontSize: 13,
            color: p.ink2, lineHeight: 1.5,
          }}>
            <span style={{ color: p.err }}>Error.</span> {skill.error}{' '}
            <span style={{ color: p.accent, cursor: 'pointer' }}>Restart</span>
            <span style={{ color: p.muted2 }}>{' · '}</span>
            <span style={{ color: p.muted, cursor: 'pointer' }}>View log</span>
          </div>
        )}

        <div style={{
          gridColumn: '1 / 4', marginTop: 10,
          display: 'flex', alignItems: 'center', gap: 10,
        }}>
          <button onClick={onToggleExpand} style={{
            appearance: 'none', border: 'none', background: 'transparent',
            padding: 0, cursor: 'pointer',
            fontFamily: TYPE[variant].mono, fontSize: 11,
            color: p.muted, letterSpacing: 0.02,
          }}>
            {expanded ? '− hide' : '+ show'} {skill.tools.length} tools
          </button>
        </div>
      </div>

      {expanded && (
        <div style={{
          borderTop: `1px solid ${p.line}`,
          background: p.bg,
          padding: '14px 22px 18px',
        }}>
          {skill.tools.map((t, i) => (
            <div key={i} style={{
              display: 'grid',
              gridTemplateColumns: '220px 1fr',
              columnGap: 18,
              padding: '6px 0',
              fontFamily: TYPE[variant].mono, fontSize: 12,
              color: p.muted, letterSpacing: 0,
            }}>
              <span style={{ color: p.ink2 }}>{t.n}</span>
              <span>{t.d}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SkillsScreen({ variant, mode, mascotVariant = 1, onToggleMode, onNavigate, onHome }) {
  const p = palette(variant, mode);
  const [expanded, setExpanded] = React.useState({ 0: true });
  const enabledCount = SKILLS.filter(s => s.enabled).length;

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      height: '100%', background: p.bg,
      overflow: 'hidden',
    }}>
      <ProductHeader
        variant={variant} mode={mode} active="skills"
        mascotVariant={mascotVariant} route="local"
        onToggleMode={onToggleMode} onNavigate={onNavigate} onHome={onHome} />

      <div style={{ flex: 1, overflow: 'auto' }}>
        <div style={{ maxWidth: 860, margin: '0 auto', padding: '44px 32px 80px' }}>
          <div style={{
            display: 'flex', alignItems: 'baseline', justifyContent: 'space-between',
            marginBottom: 28,
          }}>
            <div>
              <div style={{
                fontFamily: TYPE[variant].family,
                fontSize: 28, color: p.ink, letterSpacing: -0.01,
                fontWeight: 400, marginBottom: 8,
              }}>Skills</div>
              <div style={{
                fontFamily: TYPE[variant].family,
                fontSize: 14, color: p.muted, lineHeight: 1.6, maxWidth: 520,
              }}>
                Small background workers June runs on your behalf. Each one is local,
                sandboxed, and describes exactly what it can do.
              </div>
            </div>
            <div style={{
              fontFamily: TYPE[variant].mono, fontSize: 12,
              color: p.muted, whiteSpace: 'nowrap',
            }}>
              <span style={{ color: p.ink }}>{enabledCount}</span> of {SKILLS.length} enabled
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {SKILLS.map((s, i) => (
              <SkillCard
                key={i} variant={variant} mode={mode} skill={s}
                expanded={!!expanded[i]}
                onToggleExpand={() => setExpanded(e => ({ ...e, [i]: !e[i] }))}
              />
            ))}
          </div>

          <div style={{
            marginTop: 48,
            fontFamily: TYPE[variant].family, fontSize: 13,
            color: p.muted, textAlign: 'center',
          }}>
            Skills you install live in{' '}
            <span style={{
              fontFamily: TYPE[variant].mono, color: p.ink2,
              background: p.surface2, padding: '1px 6px', borderRadius: 4,
            }}>~/June/Skills</span>.{' '}
            Drop a folder in to add one.
          </div>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { SkillsScreen, Toggle, StatusBadge });
