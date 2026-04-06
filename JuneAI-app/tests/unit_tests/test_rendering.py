from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from agent.memory import Memory
from agent_ui.rendering import (
    default_ui_state,
    render_memory_focus,
    render_save_summary_html,
    render_trust_panel,
    render_workspace,
    transcript_html,
)
from agent_ui.shell import card_html, empty_state_html


def test_transcript_renders_structured_assistant_blocks_and_collapses_long_text() -> None:
    html = transcript_html(
        [
            HumanMessage(content="hello June"),
            AIMessage(
                content=[
                    {"type": "summary", "text": "Daily summary"},
                    {"type": "analysis", "text": "A" * 1000},
                    {"type": "bullets", "items": ["Ship UI shell", "Tighten transcript"]},
                ]
            ),
        ]
    )

    assert "Daily summary" in html
    assert "Show more" in html
    assert "<details>" in html
    assert "Ship UI shell" in html
    assert "june-chip" in html
    assert "june-message-user" in html
    assert "june-message-assistant" in html


def test_transcript_strips_gemma_thought_blocks() -> None:
    html = transcript_html(
        [
            AIMessage(
                content="<|channel>thought\ninternal reasoning<channel|>Final answer only."
            )
        ]
    )

    assert "Final answer only." in html
    assert "internal reasoning" not in html
    assert "<|channel>thought" not in html


def test_transcript_renders_compact_save_summary_block() -> None:
    html = transcript_html(
        [AIMessage(content="Saved it.")],
        extra_messages=[
            {
                "role": "assistant",
                "content": {
                    "type": "save_summary",
                    "label": "What June saved",
                    "items": [
                        "Goal saved: Ship the onboarding flow.",
                        "Calendar item saved: Launch review on 2026-04-02.",
                    ],
                },
            }
        ],
    )

    assert "What June saved" in html
    assert "Goal saved: Ship the onboarding flow." in html
    assert "Calendar item saved: Launch review on 2026-04-02." in html
    assert "june-list" in html


def test_card_helpers_render_onboarding_and_collapsed_content() -> None:
    empty_html = empty_state_html(
        title="Workspace",
        body="Nothing is pinned yet.",
        tips=("Keep one open loop visible.", "Add the next concrete action."),
        actions=("Capture a note", "Set a next step"),
        note="June will use this for follow-up.",
        eyebrow="Workspace",
    )
    collapsed_html = card_html(
        "Today",
        "B" * 1000,
        collapsed=True,
        collapse_threshold=120,
        meta=("Focus mode",),
        chips=("Shell", "Reusable"),
    )

    assert "<ul>" in empty_html
    assert "Capture a note" in empty_html
    assert "June will use this for follow-up." in empty_html
    assert "<details>" in collapsed_html
    assert "Show more" in collapsed_html
    assert "Focus mode" in collapsed_html
    assert "Shell" in collapsed_html


def test_workspace_and_memory_focus_use_empty_state_helpers(tmp_path) -> None:
    ui_state = default_ui_state()
    ui_state.update(
        {
            "focus_title": "Workspace",
            "focus_body": "",
            "checklist_title": "Next steps",
            "checklist_items": [],
            "notice": "",
        }
    )

    workspace_html = render_workspace(ui_state)

    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        memory = Memory("rendering_test")
    memory_html = render_memory_focus(memory, "plans")

    assert "Nothing pinned yet." in workspace_html
    assert "Nothing saved yet." in memory_html
    assert "Ask June to capture something relevant here." in memory_html
    assert "Use this space to pin the active focus" in workspace_html


def test_trust_panel_surfaces_saved_context_and_assistant_actions(tmp_path) -> None:
    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        memory = Memory("trust_render_user")

    memory.save_goal("Ship onboarding helpers", next_step="Wire the new panels")
    memory.save_calendar_item("Launch review", "2026-04-02", details="Review onboarding copy")
    memory.record_save_event("goal", "Ship onboarding helpers")

    html = render_trust_panel(
        memory,
        [
            "route | assistant",
            "tool request | save_goal",
            "tool args | save_goal {title: Ship onboarding helpers}",
        ],
    )

    assert "Saved context" in html
    assert "What June saved" in html
    assert "Recent assistant actions" in html
    assert "Goal saved" in html or "Calendar item saved" in html
    assert "Route chosen" in html


def test_compact_save_summary_renders_inline_without_overwhelming_the_transcript() -> None:
    summary_html = render_save_summary_html(
        "What June saved",
        [
            ("Goal saved", "Ship onboarding helpers"),
            ("Calendar item saved", "Launch review"),
        ],
        note="Saved from this turn",
    )
    transcript_variant_html = transcript_html(
        [
            {
                "role": "assistant",
                "content": {
                    "variant": "save_summary",
                    "label": "What June saved",
                    "summary": "Saved from this turn",
                    "items": [
                        {"title": "Goal saved", "text": "Ship onboarding helpers"},
                        {"title": "Calendar item saved", "text": "Launch review"},
                    ],
                },
            }
        ]
    )

    assert "june-summary-card" in summary_html
    assert "june-summary-card" in transcript_variant_html
    assert "What June saved" in summary_html
    assert "Ship onboarding helpers" in summary_html
    assert "<details>" not in summary_html
    assert "Saved from this turn" in transcript_variant_html
