"""The Anthropic adapter against a stub SDK client returning real SDK response objects. No network."""

import anthropic
import httpx2
import pytest
from anthropic.types.beta import BetaMessage

from sim2real.models import Message, ModelError, ModelSettings, TextPart, ToolCall, ToolResult, ToolSpec
from sim2real.models.anthropic_client import AnthropicClient

KEY = "sk-ant-test-0123456789abcdefghij"
SETTINGS = ModelSettings(provider="anthropic", model="claude-opus-5-5", effort="high")
TOOL = ToolSpec(name="read_output", description="Read lines of a run's output file.",
                input_schema={"type": "object", "properties": {"file": {"type": "string"}}, "required": ["file"]})


def message(content, stop="tool_use", model="claude-opus-5-5", **extra) -> BetaMessage:
    return BetaMessage.model_validate({
        "id": "msg_01", "type": "message", "role": "assistant", "model": model, "content": content,
        "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 30, "cache_read_input_tokens": 100,
                  "cache_creation_input_tokens": 0},
        **extra,
    })


TOOL_TURN = [
    {"type": "thinking", "thinking": "", "signature": "opaque-signature"},
    {"type": "text", "text": "I will read the output."},
    {"type": "tool_use", "id": "toolu_01", "name": "read_output", "input": {"file": "aims.out"}},
]


class StubSDK:
    """Stands in for anthropic.Anthropic: records request parameters, returns or raises as told."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.requests = []
        self.beta = self
        self.messages = self

    def stream(self, **params):
        self.requests.append(params)
        stub = self

        class Stream:
            def __enter__(self):
                if isinstance(stub.outcome, BaseException):
                    raise stub.outcome
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return stub.outcome

        return Stream()


def client_for(outcome) -> tuple[AnthropicClient, StubSDK]:
    sdk = StubSDK(outcome)
    return AnthropicClient(KEY, client=sdk), sdk


def test_request_parameters():
    client, sdk = client_for(message(TOOL_TURN))
    client.complete(system="You are a hypothesis agent.", tools=[TOOL], settings=SETTINGS,
                    messages=[Message(role="user", parts=(TextPart(text="Check the run."),))])
    (params,) = sdk.requests
    assert params["model"] == "claude-opus-5-5" and params["max_tokens"] == 64000
    assert params["system"] == "You are a hypothesis agent."
    assert params["output_config"] == {"effort": "high"}
    assert params["thinking"] == {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}
    assert params["betas"] == ["thinking-binding-controls-2026-08-01", "server-side-fallback-2026-07-01"]
    assert params["fallbacks"] == "default"
    assert params["cache_control"] == {"type": "ephemeral"}
    assert params["tools"] == [{"name": "read_output", "description": TOOL.description,
                                "input_schema": TOOL.input_schema, "eager_input_streaming": True}]
    assert params["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Check the run."}]}]


def test_refusal_fallback_and_thinking_settings_can_be_switched_off():
    client, sdk = client_for(message(TOOL_TURN))
    hi = [Message(role="user", parts=(TextPart(text="hi"),))]
    client.complete(system="s", messages=hi, tools=[],
                    settings=SETTINGS.model_copy(update={"options": {"refusal_fallback": False}}))
    assert sdk.requests[0]["betas"] == ["thinking-binding-controls-2026-08-01"] and "fallbacks" not in sdk.requests[0]
    options = {"refusal_fallback": False, "thinking_prefix_mismatch": "off"}
    client.complete(system="s", messages=hi, tools=[],
                    settings=SETTINGS.model_copy(update={"options": options, "effort": None}))
    assert {"betas", "fallbacks", "output_config", "tools", "thinking"}.isdisjoint(sdk.requests[1])
    client.complete(system="s", messages=hi, tools=[],
                    settings=SETTINGS.model_copy(update={"options": {"thinking_prefix_mismatch": "error"}}))
    assert sdk.requests[2]["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "error"}
    with pytest.raises(ModelError, match="thinking_prefix_mismatch must be one of") as caught:
        client.complete(system="s", messages=hi, tools=[],
                        settings=SETTINGS.model_copy(update={"options": {"thinking_prefix_mismatch": "drop"}}))
    assert not caught.value.retryable


def test_dropped_reasoning_blocks_are_counted():
    dropped = [{"type": "thinking_dropped", "path": f"messages.1.content.{i}", "reason": "prefix_binding_mismatch"}
               for i in range(2)]
    client, _ = client_for(message(TOOL_TURN, input_transformations=dropped))
    reply = client.complete(system="s", tools=[TOOL], settings=SETTINGS,
                            messages=[Message(role="user", parts=(TextPart(text="go"),))])
    assert reply.dropped_reasoning_blocks == 2
    client, _ = client_for(message(TOOL_TURN))
    assert client.complete(system="s", tools=[TOOL], settings=SETTINGS,
                           messages=[Message(role="user", parts=(TextPart(text="go"),))]).dropped_reasoning_blocks == 0


def test_reply_keeps_native_blocks_and_neutral_parts():
    client, _ = client_for(message(TOOL_TURN, model="claude-opus-5-5"))
    reply = client.complete(system="s", tools=[TOOL], settings=SETTINGS,
                            messages=[Message(role="user", parts=(TextPart(text="go"),))])
    assert reply.stop == "tool_use" and reply.served_model == "claude-opus-5-5"
    assert reply.message.text == "I will read the output."
    assert reply.message.tool_calls == (ToolCall(id="toolu_01", name="read_output", arguments={"file": "aims.out"}),)
    assert reply.message.native[0] == {"type": "thinking", "thinking": "", "signature": "opaque-signature"}
    assert (reply.usage.input_tokens, reply.usage.output_tokens, reply.usage.cache_read_tokens) == (120, 30, 100)


def test_assistant_turns_go_back_verbatim_and_tool_results_lead():
    first, _ = client_for(message(TOOL_TURN))
    reply = first.complete(system="s", tools=[TOOL], settings=SETTINGS,
                           messages=[Message(role="user", parts=(TextPart(text="go"),))])
    client, sdk = client_for(message([{"type": "text", "text": "Done."}], stop="end_turn"))
    followup = Message(role="user", parts=(TextPart(text="Here it is."),
                                           ToolResult(call_id="toolu_01", content="line 1")))
    client.complete(system="s", tools=[TOOL], settings=SETTINGS,
                    messages=[Message(role="user", parts=(TextPart(text="go"),)), reply.message, followup])
    wire = sdk.requests[0]["messages"]
    assert wire[1] == {"role": "assistant", "content": list(reply.message.native)}
    assert wire[2]["content"][0] == {"type": "tool_result", "tool_use_id": "toolu_01", "content": "line 1",
                                     "is_error": False}
    assert wire[2]["content"][1] == {"type": "text", "text": "Here it is."}


def test_turns_from_another_provider_are_sent_as_neutral_blocks():
    client, sdk = client_for(message([{"type": "text", "text": "ok"}], stop="end_turn"))
    other = Message(role="assistant", provider="openai", native=({"role": "assistant"},),
                    parts=(TextPart(text="Calling."), ToolCall(id="call_9", name="read_output", arguments={"file": "a"})))
    client.complete(system="s", tools=[TOOL], settings=SETTINGS,
                    messages=[Message(role="user", parts=(TextPart(text="go"),)), other])
    assert sdk.requests[0]["messages"][1]["content"] == [
        {"type": "text", "text": "Calling."},
        {"type": "tool_use", "id": "call_9", "name": "read_output", "input": {"file": "a"}},
    ]


def test_refusal_reports_its_category():
    refused = message([], stop="refusal", stop_details={"type": "refusal", "category": "cyber", "explanation": None})
    client, _ = client_for(refused)
    reply = client.complete(system="s", tools=[], settings=SETTINGS,
                            messages=[Message(role="user", parts=(TextPart(text="go"),))])
    assert (reply.stop, reply.refusal_category, reply.message.parts) == ("refusal", "cyber", ())


def test_served_model_differs_after_a_fallback():
    client, _ = client_for(message([{"type": "text", "text": "ok"}], stop="end_turn", model="claude-opus-4-8"))
    reply = client.complete(system="s", tools=[], settings=SETTINGS,
                            messages=[Message(role="user", parts=(TextPart(text="go"),))])
    assert (reply.requested_model, reply.served_model) == ("claude-opus-5-5", "claude-opus-4-8")


def _status_error(cls, status):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    return cls(f"rejected key {KEY}", response=response, body=None)


@pytest.mark.parametrize("error, retryable", [
    (_status_error(anthropic.AuthenticationError, 401), False),
    (_status_error(anthropic.BadRequestError, 400), False),
    (_status_error(anthropic.RateLimitError, 429), True),
    (_status_error(anthropic.InternalServerError, 500), True),
    (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")), True),
    (ValueError("Unterminated string"), True),
])
def test_errors_are_classified_and_never_leak_the_key(error, retryable):
    client, _ = client_for(error)
    with pytest.raises(ModelError) as raised:
        client.complete(system="s", tools=[], settings=SETTINGS,
                        messages=[Message(role="user", parts=(TextPart(text="go"),))])
    assert raised.value.retryable is retryable
    assert KEY not in str(raised.value)
