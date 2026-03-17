from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from agent.config import resolve_runtime_config
from agent.graph import create_june_agent
from agent.memory import Memory


class FakeLLM:
    def __init__(self, responses):
        self.responses = responses
        self.invocations = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        response = self.responses[self.invocations]
        self.invocations += 1
        return response


def test_resolve_runtime_config_for_local_mistral():
    with patch.dict(
        "os.environ",
        {
            "MODEL_PRESET": "local_mistral_3b",
            "LOCAL_SMALL_MODEL_NAME": "mistral-local-3b",
            "LLM_BASE_URL": "http://127.0.0.1:11434/v1",
            "LLM_API_KEY": "ollama",
        },
        clear=False,
    ):
        runtime = resolve_runtime_config()

    assert runtime.provider == "openai_compatible"
    assert runtime.label == "Local Mistral 3B"
    assert runtime.model == "mistral-local-3b"
    assert runtime.base_url == "http://127.0.0.1:11434/v1"


def test_resolve_runtime_config_for_claude():
    with patch.dict(
        "os.environ",
        {
            "MODEL_PRESET": "claude_high",
            "ANTHROPIC_API_KEY": "test-key",
            "CLAUDE_MODEL_NAME": "claude-test",
        },
        clear=False,
    ):
        runtime = resolve_runtime_config()

    assert runtime.provider == "anthropic"
    assert runtime.label == "Claude High Performance"
    assert runtime.model == "claude-test"
    assert runtime.api_key == "test-key"


def test_graph_tracks_tool_success():
    fake_llm = FakeLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "track_goal",
                        "args": {
                            "title": "Ship assistant runtime",
                            "category": "product",
                            "next_step": "Wire dual provider support",
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The goal is saved and ready."),
        ]
    )
    agent = create_june_agent(llm=fake_llm)

    with patch("agent.memory.MEMORY_DIR", "/tmp/juneai_test_memory"):
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Track this as a goal.")],
                "user_id": "tool_test_user",
                "skill": "assistant",
                "ui_state": {
                    "layout": "split",
                    "focus_title": "Workspace",
                    "focus_body": "",
                    "checklist_title": "Next steps",
                    "checklist_items": [],
                    "notice": "",
                },
                "tool_stats": {"requested": 0, "succeeded": 0, "failed": 0, "last_calls": []},
            }
        )

    assert result["tool_stats"]["requested"] == 1
    assert result["tool_stats"]["succeeded"] == 1
    assert result["tool_stats"]["failed"] == 0
    assert result["tool_stats"]["last_calls"][0]["name"] == "track_goal"


def test_graph_recovers_food_program_tool_call_from_json(tmp_path):
    fake_llm = FakeLLM(
        [
            AIMessage(
                content=(
                    '{"name":"save_food_program","parameters":'
                    '{"name":"Weekday Fuel","goal":"steady energy",'
                    '"daily_structure":"Protein breakfast, balanced lunch, light dinner"}}'
                )
            ),
            AIMessage(content="I saved your food schedule."),
        ]
    )
    agent = create_june_agent(llm=fake_llm)

    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Suggest and save a food schedule.")],
                "user_id": "food_test_user",
                "skill": "wellness",
                "ui_state": {
                    "layout": "split",
                    "focus_title": "Workspace",
                    "focus_body": "",
                    "checklist_title": "Next steps",
                    "checklist_items": [],
                    "notice": "",
                },
                "tool_stats": {"requested": 0, "succeeded": 0, "failed": 0, "last_calls": []},
            }
        )
        memory = Memory("food_test_user")

    assert result["tool_stats"]["succeeded"] == 1
    assert memory.get_food_programs()[0]["name"] == "Weekday Fuel"


def test_graph_remaps_birthday_journal_json_to_calendar_item(tmp_path):
    fake_llm = FakeLLM(
        [
            AIMessage(
                content=(
                    '{"name":"save_journal_entry","parameters":'
                    '{"entry":"{\\"event\\": \\"my son\'s birthday\\", '
                    '\\"date\\": \\"2026-08-24\\", '
                    '\\"note\\": \\"Reminder to celebrate my son\'s birthday on August 24th\\"}"}}'
                )
            ),
            AIMessage(content="I saved the birthday reminder."),
        ]
    )
    agent = create_june_agent(llm=fake_llm)

    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Remember my son's birthday 24 August")],
                "user_id": "birthday_test_user",
                "skill": "planner",
                "ui_state": {
                    "layout": "split",
                    "focus_title": "Workspace",
                    "focus_body": "",
                    "checklist_title": "Next steps",
                    "checklist_items": [],
                    "notice": "",
                },
                "tool_stats": {"requested": 0, "succeeded": 0, "failed": 0, "last_calls": []},
            }
        )
        memory = Memory("birthday_test_user")

    assert result["tool_stats"]["succeeded"] == 1
    assert memory.get_calendar_items()[0]["date"] == "2026-08-24"
    assert "birthday" in memory.get_calendar_items()[0]["title"].lower()


def test_graph_recovers_html_escaped_tool_json_with_wrapper_text(tmp_path):
    fake_llm = FakeLLM(
        [
            AIMessage(
                content=(
                    'I will save this now: {&quot;name&quot;:&quot;save_calendar_item&quot;,&quot;parameters&quot;:'
                    '{&quot;title&quot;:&quot;My son birthday&quot;,&quot;date&quot;:&quot;2026-08-24&quot;,'
                    '&quot;details&quot;:&quot;Birthday reminder&quot;}}'
                )
            ),
            AIMessage(content="Saved."),
        ]
    )
    agent = create_june_agent(llm=fake_llm)

    with patch("agent.memory.MEMORY_DIR", str(tmp_path)):
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Remember my son's birthday 24 August")],
                "user_id": "wrapped_json_user",
                "skill": "planner",
                "ui_state": {
                    "layout": "split",
                    "focus_title": "Workspace",
                    "focus_body": "",
                    "checklist_title": "Next steps",
                    "checklist_items": [],
                    "notice": "",
                },
                "tool_stats": {"requested": 0, "succeeded": 0, "failed": 0, "last_calls": []},
            }
        )
        memory = Memory("wrapped_json_user")

    assert result["tool_stats"]["succeeded"] == 1
    assert memory.get_calendar_items()[0]["title"] == "My son birthday"
