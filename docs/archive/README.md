# Archive — superseded planning docs

This is a **logical** archive index. The superseded plans below are **not
physically moved** out of their original paths, because they are referenced by
append-only ADRs and by earlier docs; physically moving them would either break
those links or force edits to append-only ADRs (an invariant violation, see
`docs/decisions/README.md`). Instead each carries a loud **SUPERSEDED** banner
at the top of the file, and this page is the index of record.

For the current state of the project, see [`../CURRENT.md`](../CURRENT.md). For
the active plan, see [`../../docs/product/v0.3-development-plan.md`](../product/v0.3-development-plan.md).

## Active plan

The current lead plan is [`v0.3-development-plan.md`](../product/v0.3-development-plan.md) (created 2026-07-24).

## Superseded plans (history only — do not follow for new work)

| Doc | Superseded by | Why kept in place |
|---|---|---|
| [`../../JUNE_V02_BRIEF.md`](../../JUNE_V02_BRIEF.md) | `v0.3-development-plan.md` | Contains detailed v0.2 workstream specs still referenced by ADRs 0021-0024 and RECONCILIATION.md |
| [`../product/v0.2-execution-plan.md`](../product/v0.2-execution-plan.md) | `v0.3-development-plan.md` | Execution sequencing for v0.2; historical reference |
| [`../product/development-plan.md`](../product/development-plan.md) | `JUNE_V02_BRIEF.md` then `v0.3-development-plan.md` | Linked by CLAUDE.md, README, ROADMAP, and ADRs 0014-0017 |
| [`../product/rebuild-plan.md`](../product/rebuild-plan.md) | `JUNE_V02_BRIEF.md` (carried its open items) | Linked by ADRs 0018-0021, vision.md, README |
| [`../product/ship-to-revenue.md`](../product/ship-to-revenue.md) | `JUNE_V02_BRIEF.md` section 11 | Linked by other superseded/historical docs |
| [`../design/master-brief.md`](../design/master-brief.md) | Shipped UI + `JUNE_V02_BRIEF.md` | Linked by CHANGELOG + design artifact README |

## Historical records (a record of a moment; never a live spec)

These are accurate for when they were written and are retained for provenance:
`docs/product/CLAUDE_HANDOFF_silence_and_trust.md`,
`docs/product/strategic-review-2026-07-01.md`,
`docs/product/sidecar-spike-findings.md`,
`docs/product/tauri-build-report.md`,
`docs/product/cold-start-notes.md`,
`docs/product/license-design.md`,
`docs/experiments/*`,
`docs/design/artifact/*`,
`docs/LOGO_PROMPT.md`.
