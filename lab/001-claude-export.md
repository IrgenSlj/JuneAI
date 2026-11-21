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
Partial (2026-11-21).
- The 2026 export is no longer one zip. The founder's download was a manifest (`version` "1.0", `data_files`) listing nine zips: light_metadata, projects, memories, feedback, design_chats, frames, artifact_data, claude_code_cloud and conversations, each `<category>-000.zip`. The links expire after 24 hours and need the claude.ai login, so the manifest alone imports nothing; June now says so instead of failing.
- Built against the shape documented by an open-source importer (Nexus AI Chat Importer, read 2026-11-21): conversations with uuid, name, created_at, updated_at, chat_messages and optional model, project_uuid, current_leaf_message_uuid; messages with sender, text, content blocks (text, thinking, tool_use, tool_result), attachments with extracted_content, files. Not yet checked against the founder's real conversations zip.
- `lab/scripts/claude-shape.mjs` prints the real shape (fields and how often they appear, block and tool types, counts, date range, branch count) without any message text, for when the zips are in `lab/private/claude/`.

## Decision
Provisional, to confirm on the real zips:
- Read every export shape: a folder of zips, one zip, JSON or JSON Lines, recognising conversations and projects by shape rather than file name.
- A thread follows the conversation's current path; edited and regenerated versions stay in `.raw.json` beside it.
- Thinking and tool results stay in the raw JSON; artifacts show as code; attachment text is inlined in a fence; files the export leaves out are named.
- Projects become lanes. Memories, design chats and Claude Code sessions are listed as not imported yet.
