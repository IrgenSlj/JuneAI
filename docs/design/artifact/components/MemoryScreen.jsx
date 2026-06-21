// MemoryScreen — feels like browsing a well-kept notebook.
// Facts (structured rows), Semantic memories (prose cards), People & places (chips).

const FACTS = [
  { k: 'Home',              v: 'Noe Valley, San Francisco',        src: 'inferred from location · 11 Apr',  confidence: 'high' },
  { k: 'Sister',            v: "Ella — lives in Oakland, works at a bookshop", src: 'you mentioned · 2 Apr' },
  { k: 'Coffee',            v: 'Oat flat white, no sugar',         src: 'you said · 14 Mar' },
  { k: 'Gym',               v: 'Mission Cliffs, mostly Tue/Thu',   src: 'calendar · ongoing' },
  { k: 'Food I avoid',      v: 'Shellfish — not an allergy, just never liked it', src: 'you said · 28 Feb' },
  { k: 'Working on',        v: "A book of essays. Draft due to Maren by 1 June.", src: 'you said · 3 Apr' },
];

const SEMANTIC = [
  {
    title: 'Thinks out loud before committing',
    body: "You tend to talk through decisions rather than decide silently. When you say \"I think I'll…\" it's usually a trial balloon, not a conclusion. I'll hold space for the rewrite.",
    learned: 'from 6 conversations · last refined 15 Apr',
  },
  {
    title: 'Protective of mornings',
    body: "You don't like meetings before 10 and you don't like being asked questions before coffee. If someone tries to book 9am I'll push back unless you've said yes in writing.",
    learned: 'pattern · observed across 4 weeks',
  },
  {
    title: 'Prefers concrete over generic',
    body: "Specific times, named places, real people. You mute suggestions that start with \"you could try…\" — so I won't phrase things that way.",
    learned: 'you said · 30 Mar',
  },
];

const PEOPLE = [
  { name: 'Ella',     rel: 'sister',             tint: 'close' },
  { name: 'Priya',    rel: 'manager at Oviform', tint: 'work' },
  { name: 'Arman',    rel: 'co-author',          tint: 'work' },
  { name: 'Maren',    rel: 'editor',             tint: 'work' },
  { name: 'Mom',      rel: 'family',             tint: 'close' },
  { name: 'Leo',      rel: 'climbing partner',   tint: 'friend' },
  { name: 'Noe Valley',    rel: 'neighbourhood',      tint: 'place' },
  { name: "Ottolenghi SF", rel: 'restaurant · frequent', tint: 'place' },
  { name: 'Mission Cliffs', rel: 'gym',              tint: 'place' },
];

function MemorySearch({ variant, mode }) {
  const p = palette(variant, mode);
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 12,
      border: `1px solid ${p.line}`,
      borderRadius: RADII.md,
      padding: '12px 14px',
      background: p.surface,
    }}>
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <circle cx="6" cy="6" r="4.5" stroke={p.muted2} strokeWidth="1.25" />
        <path d="M9.5 9.5L13 13" stroke={p.muted2} strokeWidth="1.25" strokeLinecap="round"/>
      </svg>
      <div style={{
        flex: 1, fontFamily: TYPE[variant].family, fontSize: 14, color: p.muted,
      }}>Search your memory — facts, moments, people…</div>
      <span style={{
        fontFamily: TYPE[variant].mono, fontSize: 11, color: p.muted2,
        border: `1px solid ${p.line}`, padding: '1px 5px', borderRadius: 4,
      }}>⌘ K</span>
    </div>
  );
}

function SectionHeader({ variant, mode, label, count, caption }) {
  const p = palette(variant, mode);
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 14, marginBottom: 18 }}>
      <div style={{
        fontFamily: TYPE[variant].family,
        fontSize: 11, fontWeight: 500, letterSpacing: 0.14,
        textTransform: 'uppercase', color: p.muted,
      }}>{label}</div>
      <div style={{
        fontFamily: TYPE[variant].mono, fontSize: 11, color: p.muted2,
      }}>{count}</div>
      <div style={{ flex: 1 }} />
      {caption && <div style={{
        fontFamily: TYPE[variant].family, fontSize: 12, color: p.muted2,
      }}>{caption}</div>}
    </div>
  );
}

function FactRow({ variant, mode, k, v, src }) {
  const p = palette(variant, mode);
  return (
    <div
      className="june-fact-row"
      style={{
        display: 'grid',
        gridTemplateColumns: '160px 1fr auto',
        columnGap: 24,
        padding: '14px 2px',
        borderBottom: `1px solid ${p.line}`,
        alignItems: 'baseline',
      }}>
      <div style={{
        fontFamily: TYPE[variant].family, fontSize: 13,
        color: p.muted, letterSpacing: 0.01,
      }}>{k}</div>
      <div style={{
        fontFamily: TYPE[variant].family, fontSize: 15, color: p.ink, lineHeight: 1.5,
      }}>{v}</div>
      <div style={{
        fontFamily: TYPE[variant].mono, fontSize: 11,
        color: p.muted2, letterSpacing: 0,
      }}>
        <span className="june-src">{src}</span>
      </div>
    </div>
  );
}

function SemanticCard({ variant, mode, title, body, learned }) {
  const p = palette(variant, mode);
  return (
    <div style={{
      background: p.surface,
      border: `1px solid ${p.line}`,
      borderRadius: RADII.lg,
      padding: '20px 22px',
    }}>
      <div style={{
        fontFamily: TYPE[variant].family,
        fontSize: 15, fontWeight: 500, color: p.ink,
        marginBottom: 6, letterSpacing: 0,
      }}>{title}</div>
      <div style={{
        fontFamily: TYPE[variant].family,
        fontSize: 14, lineHeight: 1.6, color: p.ink2,
      }}>{body}</div>
      <div style={{
        marginTop: 14,
        fontFamily: TYPE[variant].mono, fontSize: 11,
        color: p.muted2, letterSpacing: 0,
      }}>{learned}</div>
    </div>
  );
}

function PersonChip({ variant, mode, name, rel, tint }) {
  const p = palette(variant, mode);
  const dotColor = {
    close:  p.accent,
    work:   p.ink2,
    friend: p.ok,
    place:  p.muted2,
  }[tint] || p.muted;
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 10,
      padding: '8px 12px',
      border: `1px solid ${p.line}`,
      borderRadius: 999,
      background: p.surface,
    }}>
      <StatusDot color={dotColor} size={6} />
      <span style={{
        fontFamily: TYPE[variant].family, fontSize: 13, color: p.ink,
      }}>{name}</span>
      <span style={{
        fontFamily: TYPE[variant].family, fontSize: 12, color: p.muted,
      }}>{rel}</span>
    </div>
  );
}

function MemoryScreen({ variant, mode, mascotVariant = 1, onToggleMode, onNavigate, onHome }) {
  const p = palette(variant, mode);
  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      height: '100%', background: p.bg,
      overflow: 'hidden',
    }}>
      <ProductHeader
        variant={variant} mode={mode} active="memory"
        mascotVariant={mascotVariant} route="local"
        onToggleMode={onToggleMode} onNavigate={onNavigate} onHome={onHome} />

      <div style={{ flex: 1, overflow: 'auto' }}>
        <div style={{ maxWidth: 860, margin: '0 auto', padding: '44px 32px 80px' }}>
          {/* Title + intro */}
          <div style={{ marginBottom: 28 }}>
            <div style={{
              fontFamily: TYPE[variant].family,
              fontSize: 28, color: p.ink, letterSpacing: -0.01,
              fontWeight: 400, marginBottom: 8,
            }}>Memory</div>
            <div style={{
              fontFamily: TYPE[variant].family,
              fontSize: 14, color: p.muted, lineHeight: 1.6, maxWidth: 560,
            }}>
              Everything June remembers about you lives here, on your device. You can edit or
              delete anything. Nothing is sent anywhere unless you ask.
            </div>
          </div>

          <MemorySearch variant={variant} mode={mode} />

          {/* Facts */}
          <div style={{ marginTop: 48 }}>
            <SectionHeader
              variant={variant} mode={mode}
              label="Facts" count={`${FACTS.length} entries`}
              caption="Structured. Edit inline."
            />
            <div>
              {FACTS.map((f, i) => (
                <FactRow key={i} variant={variant} mode={mode} k={f.k} v={f.v} src={f.src} />
              ))}
            </div>
          </div>

          {/* Semantic */}
          <div style={{ marginTop: 56 }}>
            <SectionHeader
              variant={variant} mode={mode}
              label="Semantic memories" count={`${SEMANTIC.length} notes`}
              caption="What June has inferred about how you work."
            />
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              {SEMANTIC.map((s, i) => (
                <SemanticCard key={i} variant={variant} mode={mode} {...s} />
              ))}
            </div>
          </div>

          {/* People & places */}
          <div style={{ marginTop: 56 }}>
            <SectionHeader
              variant={variant} mode={mode}
              label="People & places" count={`${PEOPLE.length} known`}
              caption="Relationships and anchors."
            />
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {PEOPLE.map((pp, i) => (
                <PersonChip key={i} variant={variant} mode={mode} {...pp} />
              ))}
            </div>
          </div>

          {/* Quiet empty-state demonstration */}
          <div style={{
            marginTop: 64,
            padding: '22px 24px',
            border: `1px dashed ${p.lineStrong}`,
            borderRadius: RADII.md,
            color: p.muted,
            fontFamily: TYPE[variant].family, fontSize: 13, lineHeight: 1.6,
          }}>
            <span style={{ color: p.ink2 }}>Nothing about your health yet.</span>{' '}
            Grant the Health skill access and June will quietly learn your patterns — no dashboards, no streaks.
          </div>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { MemoryScreen });
