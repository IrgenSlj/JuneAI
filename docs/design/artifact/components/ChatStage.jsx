// ChatStage — the two-register screen, assembled.
// Header · Conversation (top) · centered Composer · Activity terminal (bottom).
// scenario: 'greeting' | 'active' (local turn) | 'cloud' (Gemini escalation)

const TURNS = {
  greeting: {
    route: 'local', busy: false, streaming: false, greeting: true,
    thread: [],
    turn: { id: 'idle', idle: true, live: false, steps: [] },
  },

  active: {
    route: 'local', busy: true, streaming: true, greeting: false,
    thread: [
      { role: 'june', time: '8:09', text: 'Morning. I moved your 2pm with Priya to 3:30 — she asked for the extra half hour, and it clears your lunch.' },
      { role: 'user', text: 'thanks. what’s on my plate today?' },
      {
        role: 'june',
        text: 'Four things, but only two really need you. The essays draft for Maren is the real one — it’s due Sunday and you haven’t opened it since Tuesday. The 3:30 with Priya is the other.',
        streamTail: 'The rest I can handle quietly — unless you',
      },
    ],
    turn: {
      id: 'local', idle: false, live: true,
      steps: [
        { t: '12:04:01', kind: 'recall', body: '3 memories · salience 0.81' },
        { t: '12:04:01', kind: 'route',  body: 'standard → local-fast' },
        { t: '12:04:02', kind: 'tool',   body: 'list_tasks {}' },
        { t: '12:04:02', kind: 'result', body: '4 tasks · 1 overdue' },
        { t: '12:04:03', kind: 'reasoning' },
        { t: '12:04:05', kind: 'boundary', cloud: false, model: 'gemma4:e2b', sent: '0 sent to cloud', timing: '1.2s · 320 tok' },
        { t: '12:04:05', kind: 'done',   body: 'done · 1.2s · 320 tok' },
      ],
    },
  },

  approval: {
    route: 'local', busy: true, streaming: false, greeting: false,
    thread: [
      { role: 'user', text: 'read the article Priya linked and send the three takeaways to the partners list' },
      {
        role: 'june',
        text: 'I read it and pulled three takeaways. Before anything leaves the machine, look at exactly what would go where — the text I’d send is lifted from a page I don’t control, not from anything you told me.',
      },
    ],
    approval: {
      tainted: true,
      title: 'Send the takeaways to an external address',
      why: 'You asked me to forward the key points from the article Priya linked. This goes to a list outside your contacts.',
      action: 'net.write · mail.send',
      target: 'partners@oviform-extern.com',
      payload: 'Three takeaways drafted from the linked article (above). ~140 words.',
      taintNote: 'It’s lifted from a page on the open web, not from your memory.',
    },
    turn: {
      id: 'approval', idle: false, live: true,
      steps: [
        { t: '14:22:09', kind: 'recall', body: '2 memories · salience 0.74' },
        { t: '14:22:09', kind: 'route',  body: 'standard → local-fast' },
        { t: '14:22:10', kind: 'tool',   body: 'web.fetch { url: "oviform.io/blog/state-of" }' },
        { t: '14:22:11', kind: 'result', body: '1 page · 2,300 words · marked untrusted' },
        { t: '14:22:12', kind: 'tool',   body: 'mail.compose { to: "partners@…" }' },
        { t: '14:22:12', kind: 'gate',   body: 'net.write from untrusted read', tainted: true },
      ],
    },
  },

  cloud: {
    route: 'cloud', busy: true, streaming: true, greeting: false,
    thread: [
      { role: 'user', text: 'draft a careful reply to Maren — I need another week on the essays' },
      {
        role: 'june',
        text: 'Drafted. I kept it short and didn’t over-apologise — you’ve only moved this once. It leads with the new date, the 8th, gives one honest reason, and offers her the first three essays now so the wait costs her nothing.',
        streamTail: 'Want me to send it as-is, or read it back first',
      },
    ],
    turn: {
      id: 'cloud', idle: false, live: true,
      steps: [
        { t: '09:12:30', kind: 'recall', body: '5 memories · salience 0.93' },
        { t: '09:12:30', kind: 'route',  body: 'complex → escalated to cloud · long-form drafting' },
        { t: '09:12:31', kind: 'tool',   body: 'read_thread { with: "Maren" }' },
        { t: '09:12:31', kind: 'result', body: '12 messages · last reply 9 days ago' },
        { t: '09:12:32', kind: 'reasoning' },
        { t: '09:12:36', kind: 'boundary', cloud: true, model: 'gemini-2.x', sent: '1,240 tokens ↑ · encrypted', timing: '3.4s · 890 tok' },
        { t: '09:12:36', kind: 'done',   body: 'done · 3.4s · 890 tok' },
      ],
    },
  },
};

function ChatStage({
  variant, mode, scenario = 'active', mascotVariant = 1,
  defaultExpanded = false,
  onToggleMode, onNavigate,
}) {
  const p = palette(variant, mode);
  const data = TURNS[scenario] || TURNS.active;
  const [expanded, setExpanded] = React.useState(defaultExpanded);

  // keep in sync if the orchestrator flips the default (e.g. switching tabs)
  React.useEffect(() => { setExpanded(defaultExpanded); }, [defaultExpanded, scenario]);

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      height: '100%', background: p.bg, overflow: 'hidden',
    }}>
      <ProductHeader
        variant={variant} mode={mode} active="chat"
        busy={data.busy} mascotVariant={mascotVariant} route={data.route}
        onToggleMode={onToggleMode} onNavigate={onNavigate} />

      <Conversation
        variant={variant} mode={mode}
        thread={data.thread} greeting={data.greeting} approval={data.approval} />

      <CenteredComposer
        variant={variant} mode={mode} streaming={data.streaming}
        expanded={expanded} hasActivity={!data.turn.idle}
        onToggleActivity={() => setExpanded(e => !e)} />

      <ActivityTerminal
        variant={variant} mode={mode}
        expanded={expanded} turn={data.turn} />
    </div>
  );
}

Object.assign(window, { ChatStage, TURNS });
