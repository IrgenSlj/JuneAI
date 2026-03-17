from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from agent.config import resolve_runtime_config
from agent.graph import create_june_agent


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
