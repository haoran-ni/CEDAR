"""The OpenAI adapter against the real SDK, served streams by a mock transport. No network."""

import json

import httpx2
import openai
import pytest

from cedar.models import (
    CredentialStore,
    MediaPart,
    Message,
    ModelError,
    ModelSettings,
    TextPart,
    ToolCall,
    ToolResult,
    ToolSpec,
    WebAccess,
    create_client,
)
from cedar.models.openai_client import OpenAIClient

KEY = "sk-proj-test-0123456789abcdefghijklmnop"
SETTINGS = ModelSettings(provider="openai", model="gpt-5.6-sol", effort="high")
TOOL = ToolSpec(name="read_output", description="Read lines of a run's output file.",
                input_schema={"type": "object", "properties": {"file": {"type": "string"}}, "required": ["file"]})
GO = [Message(role="user", parts=(TextPart(text="go"),))]

REASONING = {"type": "reasoning", "id": "rs_01", "summary": [], "encrypted_content": "opaque-reasoning"}
SAYS = {"type": "message", "id": "msg_01", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": "I will read the output.", "annotations": []}]}
CALLS = {"type": "function_call", "id": "fc_01", "call_id": "call_01", "name": "read_output",
         "arguments": "{\"file\": \"aims.out\"}", "status": "completed"}
TOOL_TURN = [REASONING, SAYS, CALLS]


def response(output, status="completed", model="gpt-5.6-sol", usage=None, **extra) -> dict:
    return {"id": "resp_01", "object": "response", "created_at": 0, "model": model, "status": status,
            "output": output, "error": None, "incomplete_details": None,
            "usage": usage or {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 600,
                                                                              "cache_write_tokens": 300},
                               "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 20},
                               "total_tokens": 1050}, **extra}


def finished(output, status="completed", **extra) -> list[dict]:
    """The events of a stream that ends with ``status``."""
    return [{"type": f"response.{status}", "sequence_number": 1, "response": response(output, status, **extra)}]


class Server:
    """Serves each request the next stream of events, or an HTTP error; records the request bodies."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        reply = self.replies.pop(0)
        if isinstance(reply, httpx2.Response):
            return reply
        body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in reply).encode()
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def client_for(*replies) -> tuple[OpenAIClient, Server]:
    server = Server(*replies)
    sdk = openai.OpenAI(api_key=KEY, max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(server)))
    return OpenAIClient(KEY, client=sdk), server


def test_the_openai_provider_gets_this_adapter_with_its_own_key(tmp_path):
    credentials = CredentialStore(tmp_path / "credentials.toml", environ={"OPENAI_API_KEY": KEY})
    client = create_client(ModelSettings(provider="openai", model="gpt-5.6-sol"), credentials)
    assert isinstance(client, OpenAIClient) and client._secrets == [KEY]


def test_request_parameters():
    client, server = client_for(finished(TOOL_TURN))
    client.complete(system="You are a hypothesis agent.", tools=[TOOL], settings=SETTINGS,
                    messages=[Message(role="user", parts=(TextPart(text="Check the run."),))])
    (params,) = server.requests
    assert params["model"] == "gpt-5.6-sol" and params["max_output_tokens"] == 64000
    assert params["instructions"] == "You are a hypothesis agent."
    assert params["reasoning"] == {"effort": "high"}
    assert params["store"] is False and params["stream"] is True
    assert params["tools"] == [{"type": "function", "name": "read_output", "description": TOOL.description,
                                "parameters": TOOL.input_schema, "strict": False}]
    assert params["input"] == [{"role": "user", "content": "Check the run."}]
    assert {"include", "max_tool_calls", "previous_response_id"}.isdisjoint(params)


def test_without_effort_no_reasoning_settings_are_sent():
    client, server = client_for(finished([SAYS]))
    client.complete(system="s", messages=GO, tools=[], settings=SETTINGS.model_copy(update={"effort": None}))
    assert {"reasoning", "tools"}.isdisjoint(server.requests[0])


def test_reply_keeps_native_items_and_neutral_parts():
    client, _ = client_for(finished(TOOL_TURN))
    reply = client.complete(system="s", tools=[TOOL], settings=SETTINGS, messages=GO)
    assert reply.stop == "tool_use" and reply.message.provider == "openai"
    assert reply.message.text == "I will read the output."
    assert reply.message.tool_calls == (ToolCall(id="call_01", name="read_output", arguments={"file": "aims.out"}),)
    assert [item["type"] for item in reply.message.native] == ["reasoning", "message", "function_call"]
    assert reply.message.native[0]["encrypted_content"] == "opaque-reasoning"
    assert (reply.requested_model, reply.served_model, reply.refusal_category) == ("gpt-5.6-sol", "gpt-5.6-sol", None)


def test_arguments_that_are_not_a_json_object_reach_the_runtime_unparsed():
    broken = {**CALLS, "arguments": "{\"file\": \"aims"}
    client, _ = client_for(finished([broken]))
    (call,) = client.complete(system="s", tools=[TOOL], settings=SETTINGS, messages=GO).message.tool_calls
    assert call.arguments == {"__unparsed__": "{\"file\": \"aims"}


def test_usage_separates_fresh_cached_and_written_input():
    client, _ = client_for(finished([SAYS]))
    usage = client.complete(system="s", tools=[], settings=SETTINGS, messages=GO).usage
    assert (usage.input_tokens, usage.cache_read_tokens, usage.cache_write_tokens, usage.output_tokens) == \
        (100, 600, 300, 50)


def test_assistant_turns_go_back_verbatim_and_tool_outputs_lead():
    client, server = client_for(finished(TOOL_TURN), finished([SAYS]))
    first = client.complete(system="s", tools=[TOOL], settings=SETTINGS, messages=GO)
    answer = Message(role="user", parts=(TextPart(text="Also check the forces."),
                                         ToolResult(call_id="call_01", content="| Total energy ...")))
    client.complete(system="s", tools=[TOOL], settings=SETTINGS, messages=[*GO, first.message, answer])
    sent = server.requests[1]["input"]
    assert sent[1:4] == TOOL_TURN
    assert sent[4:] == [{"type": "function_call_output", "call_id": "call_01", "output": "| Total energy ..."},
                        {"role": "user", "content": "Also check the forces."}]


def test_turns_from_another_provider_are_sent_as_neutral_items():
    client, server = client_for(finished([SAYS]))
    other = Message(role="assistant", provider="anthropic", native=({"type": "thinking"},),
                    parts=(TextPart(text="Reading."), ToolCall(id="toolu_01", name="read_output",
                                                              arguments={"file": "aims.out"})))
    result = Message(role="user", parts=(ToolResult(call_id="toolu_01", content="done", is_error=True),))
    client.complete(system="s", tools=[TOOL], settings=SETTINGS, messages=[*GO, other, result])
    assert server.requests[0]["input"][1:] == [
        {"role": "assistant", "content": "Reading."},
        {"type": "function_call", "call_id": "toolu_01", "name": "read_output", "arguments": "{\"file\": \"aims.out\"}"},
        {"type": "function_call_output", "call_id": "toolu_01", "output": "done"},
    ]


@pytest.mark.parametrize("reason, stop, category", [
    ("max_output_tokens", "max_tokens", None), ("content_filter", "refusal", "content_filter"), ("steered", "other", None),
])
def test_a_reply_cut_short_says_why(reason, stop, category):
    client, _ = client_for(finished([REASONING], "incomplete", incomplete_details={"reason": reason}))
    reply = client.complete(system="s", tools=[TOOL], settings=SETTINGS, messages=GO)
    assert (reply.stop, reply.refusal_category) == (stop, category)
    assert reply.message.native == (REASONING,)


def test_a_refusal_in_the_reply_is_a_refusal():
    refusal = {**SAYS, "content": [{"type": "refusal", "refusal": "I can't help with that."}]}
    client, _ = client_for(finished([refusal]))
    reply = client.complete(system="s", tools=[], settings=SETTINGS, messages=GO)
    assert (reply.stop, reply.refusal_category, reply.message.parts) == ("refusal", None, ())


def test_a_reply_failed_for_a_usage_policy_is_a_refusal_with_its_code():
    failed = finished(TOOL_TURN, "failed", error={"code": "bio_policy", "message": "declined"})
    client, _ = client_for(failed)
    reply = client.complete(system="s", tools=[TOOL], settings=SETTINGS, messages=GO)
    assert (reply.stop, reply.refusal_category) == ("refusal", "bio_policy")
    assert reply.message.parts == () and reply.message.native == ()


@pytest.mark.parametrize("code, retryable", [("server_error", True), ("invalid_prompt", False)])
def test_other_failed_replies_are_errors(code, retryable):
    client, _ = client_for(finished([], "failed", error={"code": code, "message": f"failed with {KEY}"}))
    with pytest.raises(ModelError, match=code) as raised:
        client.complete(system="s", tools=[], settings=SETTINGS, messages=GO)
    assert raised.value.retryable is retryable and KEY not in str(raised.value)


@pytest.mark.parametrize("code, retryable", [("server_error", True), ("rate_limit_exceeded", True),
                                             ("invalid_request", False)])
def test_errors_sent_partway_through_the_stream_are_classified_by_code(code, retryable):
    events = [{"type": "response.created", "sequence_number": 0, "response": response([], "in_progress")},
              {"type": "error", "sequence_number": 1, "code": code, "message": f"failed with {KEY}", "param": None}]
    client, _ = client_for(events)
    with pytest.raises(ModelError, match="partway through the reply") as raised:
        client.complete(system="s", tools=[], settings=SETTINGS, messages=GO)
    assert raised.value.retryable is retryable and KEY not in str(raised.value)


def test_a_stream_that_ends_without_a_response_can_be_retried():
    client, _ = client_for([{"type": "response.created", "sequence_number": 0, "response": response([], "in_progress")}])
    with pytest.raises(ModelError, match="ended without a response") as raised:
        client.complete(system="s", tools=[], settings=SETTINGS, messages=GO)
    assert raised.value.retryable


def http_error(status, code=None, type_=None):
    body = {"error": {"message": f"rejected key {KEY}", "type": type_, "code": code, "param": None}}
    return httpx2.Response(status, json=body)


@pytest.mark.parametrize("error, retryable", [
    (http_error(401, "invalid_api_key", "invalid_request_error"), False),
    (http_error(400, None, "invalid_request_error"), False),
    (http_error(429, "rate_limit_exceeded", "requests"), True),
    (http_error(429, "insufficient_quota", "insufficient_quota"), False),
    (http_error(500, None, "server_error"), True),
    (http_error(503, None, "server_error"), True),
])
def test_http_errors_are_classified_and_never_leak_the_key(error, retryable):
    client, _ = client_for(error)
    with pytest.raises(ModelError) as raised:
        client.complete(system="s", tools=[], settings=SETTINGS, messages=GO)
    assert raised.value.retryable is retryable
    assert KEY not in str(raised.value)


# -- web search, images and PDFs ----------------------------------------------------------------

SEARCHES = [{"type": "web_search_call", "id": "ws_01", "status": "completed",
             "action": {"type": "search", "query": "CO Cu(111) site"}},
            {"type": "web_search_call", "id": "ws_02", "status": "completed",
             "action": {"type": "open_page", "url": "https://example.org/co"}},
            {"type": "web_search_call", "id": "ws_03", "status": "completed",
             "action": {"type": "find_in_page", "url": "https://example.org/co", "pattern": "atop"}},
            {**SAYS, "content": [{"type": "output_text", "text": "A study reports atop adsorption.",
                                  "annotations": []}]}]


def test_web_access_adds_the_providers_own_search_with_one_limit_on_its_calls():
    client, server = client_for(finished([SAYS]))
    client.complete(system="s", messages=GO, tools=[TOOL], settings=SETTINGS,
                    web=WebAccess(allowed_domains=("fhi-aims.org",), max_searches=3, max_fetches=4))
    params = server.requests[0]
    assert params["tools"][1] == {"type": "web_search", "filters": {"allowed_domains": ["fhi-aims.org"]}}
    assert params["max_tool_calls"] == 7


def test_without_web_access_no_web_tool_is_sent_and_the_tool_type_can_be_chosen():
    client, server = client_for(finished([SAYS]), finished([SAYS]))
    client.complete(system="s", messages=GO, tools=[TOOL], settings=SETTINGS)
    assert [t["type"] for t in server.requests[0]["tools"]] == ["function"]
    older = SETTINGS.model_copy(update={"options": {"web_search_tool": "web_search_2025_08_26"}})
    client.complete(system="s", messages=GO, tools=[], settings=older, web=WebAccess())
    assert server.requests[1]["tools"] == [{"type": "web_search_2025_08_26"}]


def test_web_results_stay_native_and_only_searches_count_as_searches():
    client, _ = client_for(finished(SEARCHES))
    reply = client.complete(system="s", messages=GO, tools=[], settings=SETTINGS, web=WebAccess())
    assert reply.message.text == "A study reports atop adsorption." and len(reply.message.native) == 4
    assert (reply.usage.web_searches, reply.usage.web_fetches) == (1, 2)


def test_output_items_come_from_the_stream_when_the_final_event_leaves_them_out():
    events = [{"type": "response.output_item.done", "sequence_number": 1, "output_index": 1, "item": CALLS},
              {"type": "response.output_item.done", "sequence_number": 2, "output_index": 0, "item": REASONING}]
    final = response([])
    del final["output"]
    client, _ = client_for([*events, {"type": "response.completed", "sequence_number": 3, "response": final}])
    reply = client.complete(system="s", tools=[TOOL], settings=SETTINGS, messages=GO)
    assert reply.stop == "tool_use" and reply.message.native == (REASONING, CALLS)


def test_images_and_pdfs_go_inside_tool_outputs():
    client, server = client_for(finished([SAYS]))
    shown = ToolResult(call_id="call_01", content="Figure 2 and the paper.", media=(
        MediaPart(media_type="image/png", data="iVBORw0KGgo="),
        MediaPart(media_type="application/pdf", data="JVBERi0=", title="paper.pdf, pages 1-3 of 9"),
        MediaPart(media_type="application/pdf", data="JVBERi0=", title="notes.pdf")))
    client.complete(system="s", tools=[TOOL], settings=SETTINGS,
                    messages=[Message(role="user", parts=(shown,))])
    assert server.requests[0]["input"] == [{"type": "function_call_output", "call_id": "call_01", "output": [
        {"type": "input_text", "text": "Figure 2 and the paper."},
        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="},
        {"type": "input_file", "file_data": "data:application/pdf;base64,JVBERi0=",
         "filename": "paper.pdf, pages 1-3 of 9.pdf"},
        {"type": "input_file", "file_data": "data:application/pdf;base64,JVBERi0=", "filename": "notes.pdf"},
    ]}]
