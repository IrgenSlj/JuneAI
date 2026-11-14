# 001 — The real Claude export

## Question
What does a real Claude data export contain, and can every conversation
round-trip into the vault's thread format with nothing lost?

## Hypothesis
The export is a zip of JSON files (conversations, projects, users) with stable
conversation and message ids, original timestamps, and attachment metadata.
All of it maps onto the thread format; project membership maps onto lanes.
Unknown: whether attachment contents and model names are included.

## Cheapest experiment
1. Founder requests the export (claude.ai: Settings, Privacy, Export data) and places the zip in `lab/private/`.
2. A throwaway script in `lab/scripts/` prints the file list, the JSON schema of each file (keys and types, no content), counts, and the date range.
3. Map each field to the thread format; list what has no home and what the format needs that the export lacks.
4. Cut a redacted fixture of three conversations for the M1 importer tests.

## Result
Pending.

## Decision
Pending. Expected to finalise the thread frontmatter in `PLAN.md` and unblock M1.
