// SystemScreen — "the glass box". Two audiences, one calm on-demand page.
// 1 plain-language verdict (from the capability profile)
// 2 numbers for the technical
// 3 the egress log — the visible record of every time data left the device.

const CAPABILITY = [
  { op: 'Summarizing',        verdict: 'good', note: 'tight, faithful summaries' },
  { op: 'Structured output',  verdict: 'good', note: 'clean JSON, reliable fields' },
  { op: 'Relevance scoring',  verdict: 'good', note: 'memory recall is on point' },
  { op: 'Long-context recall',verdict: 'weak', note: 'fades past ~12k tokens — June compacts early to compensate' },
];

const NUMBERS = [
  { k: 'Throughput',     v: '41', unit: 'tok/sec', tone: 'ok' },
  { k: 'Context fill',   v: '38', unit: '%',       tone: 'ok' },
  { k: 'Memory pressure',v: 'low',unit: '',        tone: 'ok' },
  { k: 'Ollama',         v: 'reachable', unit: '', tone: 'ok' },
  { k: 'Gemini',         v: 'opt-in', unit: '',    tone: 'warn' },
  { k: 'Model',          v: 'gemma4:e2b', unit: '',tone: 'ink' },
];

// The trust ledger. Local turns never leave a row that "went" anywhere.
const EGRESS = [
  { t: '09:12:36', dest: 'Gemini API', kind: 'cloud', payload: 'Draft reply to Maren — 1,240 tok, encrypted in transit', reason: 'You asked for careful long-form drafting', turn: '#0461' },
  { t: '08:41:02', dest: 'maps.skill', kind: 'service', payload: 'Geocode “Mission Cliffs” — name only, no calendar', reason: 'Travel time for your 6pm', turn: '#0459' },
  { t: 'Yesterday 19:20', dest: 'Gemini API', kind: 'cloud', payload: 'Summarize a 40-page PDF you dropped in — 8,900 tok', reason: 'You asked for the gist', turn: '#0448' },
  { t: 'Yesterday 08:00', dest: 'calendar.skill', kind: 'service', payload: 'Read today’s events — stayed on device', reason: 'Morning brief', turn: '#0441' },
];

function SysSection({ variant, mode, label, count, caption }) {
  const p = palette(variant, mode);
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 14, marginBottom: 18 }}>
      <div style={{
        fontFamily: TYPE[variant].family, fontSize: 11, fontWeight: 500, letterSpacing: 0.14,
        textTransform: 'uppercase', color: p.muted,
      }}>{label}</div>
      <div style={{ fontFamily: TYPE[variant].mono, fontSize: 11, color: p.muted2 }}>{count}</div>
      <div style={{ flex: 1 }} />
      {caption && <div style={{ fontFamily: TYPE[variant].family, fontSize: 12, color: p.muted2 }}>{caption}</div>}
    </div>
  );
}

function CapBadge({ variant, mode, verdict }) {
  const p = palette(variant, mode);
  const map = { good: p.ok, weak: p.warn, poor: p.err };
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 7,
      fontFamily: TYPE[variant].family, fontSize: 12.5, color: p.muted,
    }}>
      <StatusDot color={map[verdict]} size={6} />
      {verdict}
    </span>
  );
}

function StatTile({ variant, mode, k, v, unit, tone }) {
  const p = palette(variant, mode);
  const col = { ok: p.ok, warn: p.warn, ink: p.ink2 }[tone] || p.ink;
  return (
    <div style={{
      border: `1px solid ${p.line}`, borderRadius: RADII.md,
      background: p.surface, padding: '15px 16px',
    }}>
      <div style={{ fontFamily: TYPE[variant].mono, fontSize: 11, color: p.muted2, letterSpacing: 0.02 }}>{k}</div>
      <div style={{ marginTop: 7, display: 'flex', alignItems: 'baseline', gap: 5 }}>
        <span style={{ fontFamily: TYPE[variant].family, fontSize: 22, color: col, letterSpacing: -0.01 }}>{v}</span>
        {unit && <span style={{ fontFamily: TYPE[variant].mono, fontSize: 11.5, color: p.muted2 }}>{unit}</span>}
      </div>
    </div>
  );
}

function EgressRow({ variant, mode, row, last }) {
  const p = palette(variant, mode);
  const c = chrome(variant, mode);
  const cloud = row.kind === 'cloud';
  const accent = cloud ? p.warn : p.muted;
  return (
    <div style={{
      display: 'grid', gridTemplateColumns: '108px 130px 1fr 64px',
      columnGap: 16, padding: '13px 16px', alignItems: 'baseline',
      borderBottom: last ? 'none' : `1px solid ${c.termLine}`,
      borderLeft: `2px solid ${cloud ? p.warn : 'transparent'}`,
      background: cloud ? p.accentSoft : 'transparent',
    }}>
      <span style={{ fontFamily: TYPE[variant].mono, fontSize: 11.5, color: c.termDim }}>{row.t}</span>
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7, fontFamily: TYPE[variant].mono, fontSize: 11.5, color: accent }}>
        <StatusDot color={cloud ? p.warn : p.muted2} size={5} />
        {row.dest}
      </span>
      <span style={{ fontFamily: TYPE[variant].family, fontSize: 13, lineHeight: 1.5, color: c.termInkStrong }}>
        {row.payload}
        <span style={{ display: 'block', fontFamily: TYPE[variant].mono, fontSize: 11, color: c.termDim, marginTop: 3 }}>{row.reason}</span>
      </span>
      <span style={{ fontFamily: TYPE[variant].mono, fontSize: 11, color: c.termDim, textAlign: 'right' }}>{row.turn}</span>
    </div>
  );
}

function SystemScreen({ variant, mode, mascotVariant = 1, onToggleMode, onNavigate, onHome }) {
  const p = palette(variant, mode);
  const c = chrome(variant, mode);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: p.bg, overflow: 'hidden' }}>
      <ProductHeader
        variant={variant} mode={mode} active="system"
        mascotVariant={mascotVariant} route="local"
        onToggleMode={onToggleMode} onNavigate={onNavigate} onHome={onHome} />

      <div style={{ flex: 1, overflow: 'auto' }}>
        <div style={{ maxWidth: 860, margin: '0 auto', padding: '44px 32px 80px' }}>
          <div style={{ marginBottom: 28 }}>
            <div style={{
              fontFamily: TYPE[variant].family, fontSize: 28, color: p.ink,
              letterSpacing: -0.01, fontWeight: 400, marginBottom: 8,
            }}>System</div>
            <div style={{
              fontFamily: TYPE[variant].family, fontSize: 14, color: p.muted,
              lineHeight: 1.6, maxWidth: 560,
            }}>
              How your local brain is doing, in plain words and in numbers — and the full
              record of every time anything left this device. Nothing here updates unless
              you’re looking.
            </div>
          </div>

          {/* Plain-language verdict */}
          <div style={{
            border: `1px solid ${p.line}`, borderRadius: RADII.lg, background: p.surface,
            padding: '24px 26px', display: 'flex', gap: 20, alignItems: 'flex-start',
          }}>
            <div style={{ flexShrink: 0, marginTop: 2 }}>
              <Mascot variant={mascotVariant} state="idle" size={40} accent={p.accent} />
            </div>
            <div>
              <div style={{
                fontFamily: TYPE[variant].family, fontSize: 19, color: p.ink,
                lineHeight: 1.45, letterSpacing: -0.005,
              }}>June’s local brain is running well today.</div>
              <div style={{
                fontFamily: TYPE[variant].family, fontSize: 14, color: p.ink2,
                lineHeight: 1.6, marginTop: 8, maxWidth: 560,
              }}>
                She’s sharp at summarizing, recall, and structured work. Long stretches of
                context tire her out — so she compacts early rather than lose the thread.
                Nothing needs your attention.
              </div>
            </div>
          </div>

          {/* Capability profile */}
          <div style={{ marginTop: 44 }}>
            <SysSection variant={variant} mode={mode} label="Capability profile" count="probed on launch" caption="What the local model is good at right now." />
            <div style={{ border: `1px solid ${p.line}`, borderRadius: RADII.md, overflow: 'hidden' }}>
              {CAPABILITY.map((cap, i) => (
                <div key={i} style={{
                  display: 'grid', gridTemplateColumns: '180px 120px 1fr', columnGap: 18,
                  padding: '14px 16px', alignItems: 'baseline',
                  background: p.surface, borderBottom: i < CAPABILITY.length - 1 ? `1px solid ${p.line}` : 'none',
                }}>
                  <span style={{ fontFamily: TYPE[variant].family, fontSize: 14, color: p.ink }}>{cap.op}</span>
                  <CapBadge variant={variant} mode={mode} verdict={cap.verdict} />
                  <span style={{ fontFamily: TYPE[variant].family, fontSize: 13, color: p.muted, lineHeight: 1.5 }}>{cap.note}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Numbers */}
          <div style={{ marginTop: 44 }}>
            <SysSection variant={variant} mode={mode} label="Runtime" count="live" caption="For when you want the numbers." />
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              {NUMBERS.map((n, i) => <StatTile key={i} variant={variant} mode={mode} {...n} />)}
            </div>
          </div>

          {/* The egress log — trust ledger */}
          <div style={{ marginTop: 44 }}>
            <SysSection variant={variant} mode={mode} label="Egress log" count="last 24h" caption="Every byte that left the device, and why." />
            <div style={{
              border: `1px solid ${c.termLine}`, borderRadius: RADII.md,
              background: c.termBg, overflow: 'hidden',
            }}>
              <div style={{
                display: 'grid', gridTemplateColumns: '108px 130px 1fr 64px', columnGap: 16,
                padding: '10px 16px', borderBottom: `1px solid ${c.termLine}`,
                fontFamily: TYPE[variant].mono, fontSize: 10, letterSpacing: 0.1,
                textTransform: 'uppercase', color: c.termDim,
              }}>
                <span>when</span><span>where</span><span>what · why</span><span style={{ textAlign: 'right' }}>turn</span>
              </div>
              {EGRESS.map((row, i) => (
                <EgressRow key={i} variant={variant} mode={mode} row={row} last={i === EGRESS.length - 1} />
              ))}
            </div>
            <div style={{
              marginTop: 14, fontFamily: TYPE[variant].family, fontSize: 13,
              color: p.muted, lineHeight: 1.6,
            }}>
              <span style={{ color: p.ok }}>In local-only mode this log stays empty.</span>{' '}
              Two rows today reached Gemini — both because you asked, both shown before
              they were sent. This record is provable in code, not just promised here.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { SystemScreen });
