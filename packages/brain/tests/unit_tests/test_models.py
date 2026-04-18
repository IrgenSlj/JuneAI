from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import SecretStr

from june_brain.config import resolve_runtime_config, runtime_preset_options
from june_brain.graph import _build_memory_context, create_june_agent
from june_brain.memory import Memory
from june_brain.models import build_chat_model


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
    assert runtime.label == "Claude (API)"
    assert runtime.model == "claude-test"
    assert runtime.api_key == "test-key"


def test_resolve_runtime_config_accepts_explicit_preset_key():
    with patch.dict("os.environ", {}, clear=False):
        runtime = resolve_runtime_config("local_mistral_8b")

    assert runtime.preset_key == "local_mistral_8b"
    assert runtime.label == "Local Mistral 8B"


def test_runtime_preset_options_expose_known_presets():
    option_keys = [preset.key for preset in runtime_preset_options()]

    assert "local_gemma_4" in option_keys
    assert "local_mistral_3b" in option_keys
    assert "local_mistral_8b" in option_keys
    assert "claude_high" in option_keys


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

    with patch("june_brain.memory.MEMORY_DIR", "/tmp/juneai_test_memory"):
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

    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
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

    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
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

    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
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


def test_graph_recovers_single_quoted_multiple_tool_calls_with_aliases(tmp_path):
    fake_llm = FakeLLM(
        [
            AIMessage(
                content=(
                    "I'll save both now: "
                    "[{'tool':'save_goal','args':{'goal':'Book summer trip','next':'Pick flights'}},"
                    "{'tool':'open_chapter','args':{'section':'plans'}}]"
                )
            ),
            AIMessage(content="Saved."),
        ]
    )
    agent = create_june_agent(llm=fake_llm)

    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Track booking the trip and show my plans.")],
                "user_id": "alias_tool_user",
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
        memory = Memory("alias_tool_user")

    assert result["tool_stats"]["succeeded"] == 2
    assert memory.get_goals()[0]["title"] == "Book summer trip"
    assert result["ui_state"]["selected_chapter"] == "plans"


def test_graph_recovers_function_wrapper_tool_payload(tmp_path):
    fake_llm = FakeLLM(
        [
            AIMessage(
                content=(
                    '{"tool_calls":[{"function":{"name":"save_food_program",'
                    '"arguments":"{\\"title\\":\\"Cut Plan\\",\\"structure\\":\\"Protein breakfast, salad lunch\\",'
                    '\\"goal\\":\\"fat loss\\"}"}}]}'
                )
            ),
            AIMessage(content="Saved."),
        ]
    )
    agent = create_june_agent(llm=fake_llm)

    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Save my cut plan.")],
                "user_id": "function_wrapper_user",
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
        memory = Memory("function_wrapper_user")

    assert result["tool_stats"]["succeeded"] == 1
    assert memory.get_food_programs()[0]["name"] == "Cut Plan"
    assert memory.get_food_programs()[0]["daily_structure"] == "Protein breakfast, salad lunch"


def test_build_chat_model_uses_current_openai_signature():
    runtime = resolve_runtime_config()
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch("june_brain.models.ChatOpenAI", FakeChatOpenAI):
        build_chat_model(runtime)

    assert captured["model"] == runtime.model
    assert isinstance(captured["api_key"], SecretStr)
    assert captured["api_key"].get_secret_value() == runtime.api_key
    assert captured["base_url"] == runtime.base_url
    assert captured["max_completion_tokens"] == runtime.max_tokens
    assert captured["streaming"] is True
    assert captured["timeout"] == 120


def test_build_chat_model_uses_current_anthropic_signature():
    runtime = resolve_runtime_config()
    runtime = runtime.__class__(
        preset_key="claude_high",
        label="Claude (API)",
        provider="anthropic",
        model="claude-test",
        api_key="test-key",
        base_url="",
        temperature=0.3,
        max_tokens=777,
        tool_strategy="native",
        prompt_style="claude",
    )
    captured = {}

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch.dict("sys.modules", {"langchain_anthropic": type("M", (), {"ChatAnthropic": FakeChatAnthropic})}):
        build_chat_model(runtime)

    assert captured["model_name"] == "claude-test"
    assert isinstance(captured["api_key"], SecretStr)
    assert captured["api_key"].get_secret_value() == "test-key"
    assert captured["max_tokens_to_sample"] == 777
    assert captured["streaming"] is True


def test_memory_context_includes_detailed_body_metrics(tmp_path):
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        memory = Memory("body_context_user")
        memory.log_body_metrics(
            weight_kg=80.2,
            sleep_hours=7.2,
            sleep_quality=4,
            energy=3,
            stress=2,
            soreness=2,
            resting_hr=58,
            steps=9100,
            notes="Solid baseline day.",
        )
        memory.log_body_metrics(
            weight_kg=79.8,
            sleep_hours=6.5,
            sleep_quality=3,
            energy=2,
            stress=4,
            soreness=3,
            resting_hr=61,
            steps=8450,
            notes="Travel fatigue.",
        )

        context = _build_memory_context("body_context_user")

    assert "Today's body metrics" in context
    assert "sleep 6.5h" in context
    assert "stress 4/5" in context
    assert "resting HR 61" in context
    assert "Travel fatigue." in context
