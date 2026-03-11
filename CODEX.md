# JuneAI Codex

## Product Strategy

JuneAI is an offline-first personal assistant for people who want one system that helps them think clearly, stay organized, maintain healthy routines, and preserve continuity across conversations.

The app is not a generic chatbot. It should behave like a personal operating layer:

- It remembers relevant details over time.
- It turns conversation into structure.
- It supports planning, wellness, taste, and follow-through.
- It feels private, calm, minimal, and dependable.

The product direction is:

- Offline-friendly by default.
- Local-first memory and storage.
- Broad personal assistant capabilities rather than narrow relationship coaching.
- Useful during normal daily life, not only during explicit “assistant” tasks.

Core user outcomes:

- Capture plans and commitments as they emerge in conversation.
- Build a persistent calendar-like memory without requiring manual entry for everything.
- Save recommendations and favorites for books, films, and other interests.
- Support active, healthy routines through gym schedules and food programs.
- Track goals, open loops, and preferences so the assistant becomes more useful over time.


## Experience Principles

The interface should feel like a quiet command center, not a productivity app full of noise.

Design principles:

- Minimal and spacious.
- White or very light backgrounds by default.
- Clear typography with personality, but not decorative excess.
- Soft surfaces, restrained contrast, and strong readability.
- Chat remains central, but surrounding memory surfaces stay visible.
- Every panel should earn its place through utility.

The assistant should feel:

- Calm
- Direct
- Competent
- Private
- Structured

Avoid:

- Marketing-style copy inside the app shell
- Overdesigned gradients and visual effects
- Loud dashboards
- “AI companion” fluff that reduces trust
- Empty decorative widgets


## UX Structure

The current app direction is a 2-column assistant layout:

- Left column: conversation
- Right column: reminders, chapter buttons, chapter content, workspace, and logs

This layout should remain the baseline unless there is a strong product reason to change it.

Important UI surfaces:

- Conversation
  - Main interaction loop
  - Streaming responses
  - Most important element in the app

- Workspace
  - Model-pinned structured notes
  - Checklists, focus summaries, current plan
  - Must be useful and sparse

- Notifications
  - Local reminders derived from calendar items and due plans
  - Should be visible quickly and easy to scan

- Chapters
  - Grid of square buttons for major life areas
  - Content opens inline below the grid and closes when toggled again
  - The app should always make it obvious what is actually stored

- Calendar
  - Conversation-derived events, reminders, and commitments

- Plans
  - Goals and open loops

- Gym / Food
  - Saved wellness structure and repeatable routines

- Dating / Family / Birthdays / Trips
  - Chaptered memory views for relational and life-event context

- Logs
  - Internal agent actions and tool usage
  - Useful for visibility, debugging, and verifying memory capture

- Capture Health
  - Small coverage panel that shows how much memory exists in each chapter
  - Helps verify tool usage and storage quality


## Assistant Behavior

June should act like a high-agency personal assistant with memory.

Behavioral rules:

- Listen for explicit and implicit structure in conversation.
- Save useful information proactively when confidence is high.
- Prefer concise, actionable responses.
- Use tools when they improve continuity, recall, or execution.
- Avoid unnecessary verbosity.
- Avoid pretending certainty when the user has not given enough information.

June should proactively capture:

- Calendar items when plans become concrete
- Preferences when the user states stable tastes or habits
- Favorites when the user wants to keep a recommendation
- Goals and open loops when the conversation creates a clear follow-up
- Gym and food plans when the user wants continuity in health routines
- Birthdays, trips, and family/dating context when the user clearly shares them

June should not:

- Spam memory with weak inferences
- Overuse UI tools for decoration
- Behave like a therapist by default
- Default to relationship advice unless the conversation is actually about that


## Current Skill Model

The app currently uses four primary skills:

- `assistant`
  - General executive/personal assistant mode

- `planner`
  - Scheduling, commitments, plans, follow-through

- `wellness`
  - Gym schedules, routines, food programs, sustainable structure

- `curator`
  - Taste learning, books, movies, saved recommendations

These skills are prompt overlays, not separate products. Future work should keep them coherent and compact.


## Memory Strategy

JuneAI is local-first. Memory is stored as JSON files per user in `.june_memory`.

Current memory domains:

- chat history
- moods
- journal entries
- relationship profiles
- goals
- open loops
- preferences
- calendar items
- favorites
- gym plans
- food programs
- app state for quote rotation and daily check-ins

Memory rules:

- Prefer simple JSON storage while the app remains local-first and single-user/small-scale.
- Reads must be resilient to malformed files.
- Writes should be atomic to reduce corruption risk.
- Memory structures should remain human-inspectable.
- New memory types should only be added when they have clear product value.


## Technical Stack

Current stack:

- UI: Streamlit
- Agent orchestration: LangGraph
- LLM client: LangChain + `langchain-openai`
- Model backend: OpenAI-compatible APIs, with local Ollama as a key target
- Storage: local JSON files
- Language: Python

Key files:

- `/Users/admin/JuneAI/JuneAI-app/app.py`
  - Streamlit interface

- `/Users/admin/JuneAI/JuneAI-app/src/agent/graph.py`
  - LangGraph agent definition

- `/Users/admin/JuneAI/JuneAI-app/src/agent/tools.py`
  - Agent tools and UI update tools

- `/Users/admin/JuneAI/JuneAI-app/src/agent/memory.py`
  - Persistent local memory layer

- `/Users/admin/JuneAI/JuneAI-app/src/agent/skills.py`
  - Skill registry and prompt construction


## Architecture Notes

The current architecture is intentionally simple:

1. Streamlit gathers user input.
2. The app sends state to the LangGraph agent.
3. The model decides whether to answer directly or use tools.
4. Tools update memory and optionally update the workspace UI state.
5. Streamed events are shown in the UI.

This simplicity is a feature. Avoid overengineering.

Preferred future direction:

- Keep app state understandable.
- Keep memory transparent.
- Keep tools focused and composable.
- Add features by extending memory + tool + UI surfaces together.

Avoid:

- adding backend complexity without real need
- building a large service layer too early
- introducing a database before the local JSON model becomes a real blocker


## Design Constraints For Future Work

When editing the UI:

- Keep the background white or very light unless there is a deliberate redesign decision.
- Do not add promotional headlines or aspirational product slogans to the main shell.
- Favor simple surfaces over complex cards-within-cards.
- Maintain strong spacing and readable typography.
- Preserve the feeling of a private desktop assistant.

When adding features:

- They must map to a real persistent surface or meaningful assistant behavior.
- They should strengthen the “personal operating layer” concept.
- They should not turn the app into a generic dashboard.

Good future features:

- Better typed storage for birthdays, trips, family, and dating
- Better calendar visualization
- Smarter reminders and recurring plans
- Routine adherence tracking
- Reading/watchlist refinement
- Daily or weekly review workflows
- Better local model setup and offline ergonomics

Bad future features:

- Social feed concepts
- Gimmicky avatars
- Excessive analytics panels
- Decorative animations without product value


## Development Rules

When working on JuneAI in future sessions:

- Preserve the offline-first, local-first product direction.
- Prefer incremental changes over rewrites.
- Keep the UI minimal and useful.
- Maintain compatibility with persisted local memory where practical.
- If memory schema changes, add graceful fallback or recovery logic.
- Update tests when memory or tools change.
- Verify both behavior and visual consistency, not just syntax.


## North Star

JuneAI should feel like a private, local, intelligent life console:

- one place to talk
- one place to remember
- one place to plan
- one place to stay organized and healthy

If a future change does not strengthen that direction, it should probably not be added.
