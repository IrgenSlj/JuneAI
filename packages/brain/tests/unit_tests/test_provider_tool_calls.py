"""Unit tests for provider-native tool-call types and OpenAI helpers (S5.1)."""

from __future__ import annotations

from types import SimpleNamespace

from june_brain.providers.base import (
    GenerateRequest,
    GenerateResult,
    Message,
    ToolCall,
    ToolSpec,
    parse_openai_tool_calls,
    tool_specs_to_openai,
)


def _spec() -> ToolSpec:
    return ToolSpec(
        name="get_weather",
        description="Look up the weather.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )


def test_tool_specs_to_openai_shape():
    out = tool_specs_to_openai([_spec()])
    assert out == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up the weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


def test_empty_parameters_defaults_to_object():
    out = tool_specs_to_openai([ToolSpec(name="ping")])
    assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def _message_with_calls(calls):
    return SimpleNamespace(tool_calls=calls)


def _call(name, arguments):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def test_parse_tool_calls_json_arguments():
    msg = _message_with_calls([_call("get_weather", '{"city": "NYC"}')])
    calls = parse_openai_tool_calls(msg)
    assert calls == [ToolCall(name="get_weather", arguments={"city": "NYC"})]


def test_parse_tool_calls_multiple():
    msg = _message_with_calls(
        [_call("a", '{"x": 1}'), _call("b", '{"y": 2}')]
    )
    calls = parse_openai_tool_calls(msg)
    assert [c.name for c in calls] == ["a", "b"]
    assert calls[1].arguments == {"y": 2}


def test_parse_tool_calls_garbled_arguments_degrade_to_empty():
    msg = _message_with_calls([_call("get_weather", "not json{{{")])
    calls = parse_openai_tool_calls(msg)
    assert calls == [ToolCall(name="get_weather", arguments={})]


def test_parse_tool_calls_skips_nameless():
    msg = _message_with_calls([_call("", "{}"), _call("real", "{}")])
    calls = parse_openai_tool_calls(msg)
    assert [c.name for c in calls] == ["real"]


def test_parse_tool_calls_none():
    assert parse_openai_tool_calls(SimpleNamespace(tool_calls=None)) == []
    assert parse_openai_tool_calls(SimpleNamespace()) == []


def test_request_and_result_carry_tool_fields():
    req = GenerateRequest(messages=[Message(role="user", content="hi")], max_tokens=8, tools=[_spec()])
    assert req.tools is not None and req.tools[0].name == "get_weather"
    res = GenerateResult(
        text="", input_tokens=1, output_tokens=1, latency_ms=1,
        model_id="m", tier="local-fast", tool_calls=[ToolCall(name="x")],
    )
    assert res.tool_calls[0].name == "x"
    # Default is an empty list.
    res2 = GenerateResult(text="", input_tokens=0, output_tokens=0, latency_ms=0, model_id="m", tier="local-fast")
    assert res2.tool_calls == []
