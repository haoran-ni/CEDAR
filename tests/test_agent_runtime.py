"""The agent loop, driven by the ScriptedModel TEST DOUBLE."""

import json

import pytest
from pydantic import Field

from sim2real.agents import AgentRunner, AgentTool, RunLimits, ToolFailure, ToolOutcome, TranscriptStore
from sim2real.agents.tools import plain_schema
from sim2real.contracts.base import Contract
from sim2real.models import ModelError, ModelSettings, TextPart, ToolCall, ToolResult
from sim2real.models.scripted import ScriptedModel, reply

SETTINGS = ModelSettings(provider="scripted", model="scripted-model")
KEY = "sk-ant-test-0123456789abcdefghij"


class EchoArgs(Contract):
    text: str = Field(min_length=1)


def tools(log: list):
    def echo(args: EchoArgs) -> str:
        log.append(args.text)
        return f"echo: {args.text}"

    def leak(args: EchoArgs) -> str:
        return f"the key is {KEY}"

    def wait(args: EchoArgs) -> ToolOutcome:
        log.append("wait")
        return ToolOutcome("waiting", end_run=True)

    def fail(args: EchoArgs) -> str:
        raise ToolFailure("cannot do that")

    def crash(args: EchoArgs) -> str:
        raise RuntimeError("bug")

    return [AgentTool("echo", "Echo text.", EchoArgs, echo), AgentTool("leak", "Leak.", EchoArgs, leak),
            AgentTool("wait", "Wait.", EchoArgs, wait), AgentTool("fail", "Fail.", EchoArgs, fail),
            AgentTool("crash", "Crash.", EchoArgs, crash)]


def call(name, call_id="c1", **arguments):
    return ToolCall(id=call_id, name=name, arguments=arguments)


@pytest.fixture
def transcripts(tmp_path):
    return TranscriptStore(tmp_path / "transcripts")


def run(transcripts, script, log=None, limits=RunLimits(), secrets=(), session="s1", instruction="Do it."):
    model = ScriptedModel(list(script))
    runner = AgentRunner(model, SETTINGS, transcripts, secrets=secrets)
    result = runner.run(session, system="system prompt", tools=tools(log if log is not None else []),
                        instruction=instruction, limits=limits)
    return result, model


def tool_results(transcripts, session="s1"):
    return [p for m in transcripts.load(session) for p in m.parts if isinstance(p, ToolResult)]


def test_answer_without_tools_finishes(transcripts):
    result, model = run(transcripts, [reply(TextPart(text="All done."))])
    assert (result.outcome, result.final_text, result.turns) == ("finished", "All done.", 1)
    assert model.calls[0].system == "system prompt"
    assert {t.name for t in model.calls[0].tools} == {"echo", "leak", "wait", "fail", "crash"}


def test_parallel_tool_calls_answer_in_one_message(transcripts):
    log = []
    result, model = run(transcripts, [
        reply(call("echo", "a", text="one"), call("echo", "b", text="two")),
        reply(TextPart(text="Echoed both.")),
    ], log)
    assert log == ["one", "two"] and result.outcome == "finished" and result.turns == 2
    last_request = model.calls[1].messages
    assert [p.call_id for p in last_request[-1].parts] == ["a", "b"]
    assert result.usage.input_tokens == 20


def test_invalid_arguments_never_reach_the_handler(transcripts):
    log = []
    run(transcripts, [reply(call("echo", text="")), reply(TextPart(text="ok"))], log)
    (outcome,) = tool_results(transcripts)
    assert log == [] and outcome.is_error and "invalid_arguments" in outcome.content


def test_unknown_tools_and_expected_failures_become_error_results(transcripts):
    run(transcripts, [reply(call("nope", "a", text="x"), call("fail", "b", text="x")), reply(TextPart(text="ok"))])
    unknown, failure = tool_results(transcripts)
    assert unknown.is_error and "Unknown tool 'nope'" in unknown.content
    assert failure.is_error and failure.content == "ToolFailure: cannot do that"


def test_bugs_in_tools_are_not_hidden(transcripts):
    with pytest.raises(RuntimeError, match="bug"):
        run(transcripts, [reply(call("crash", text="x"))])


@pytest.mark.parametrize("stop, outcome", [("max_tokens", "truncated"), ("refusal", "refused")])
def test_tools_from_cut_off_or_refused_turns_are_not_run(transcripts, stop, outcome):
    log = []
    result, _ = run(transcripts, [reply(call("echo", text="hi"), stop=stop)], log)
    assert result.outcome == outcome and log == []
    (answer,) = tool_results(transcripts)
    assert answer.is_error and answer.content.startswith("Not run")


def test_model_errors_end_the_run_with_a_redacted_message(transcripts):
    def boom(_):
        raise ModelError(f"rejected {KEY}", retryable=False)

    result, _ = run(transcripts, [boom], secrets=[KEY])
    assert result.outcome == "model_error" and KEY not in result.error


def test_tool_output_is_redacted(transcripts):
    run(transcripts, [reply(call("leak", text="x")), reply(TextPart(text="ok"))], secrets=[KEY])
    (outcome,) = tool_results(transcripts)
    assert KEY not in outcome.content


def test_waiting_tool_ends_the_run(transcripts):
    log = []
    result, model = run(transcripts, [reply(TextPart(text="Waiting for jobs."), call("wait", text="x"))], log)
    assert result.outcome == "waiting" and result.final_text == "Waiting for jobs." and len(model.calls) == 1


def test_limits(transcripts):
    looping = [reply(call("echo", text="again")) for _ in range(5)]
    assert run(transcripts, looping, limits=RunLimits(max_turns=3))[0].outcome == "turn_limit"
    assert run(transcripts, looping, limits=RunLimits(max_tokens=20), session="s2")[0].outcome == "token_limit"


def test_transcripts_are_resumed_and_only_appended(transcripts):
    run(transcripts, [reply(TextPart(text="First answer."))], instruction="First.")
    first = transcripts.load("s1")
    _, model = run(transcripts, [reply(TextPart(text="Second answer."))], instruction="Second.")
    sent = model.calls[0].messages
    assert sent[:2] == tuple(first) and sent[2].text == "Second."
    assert transcripts.load("s1")[:2] == first and len(transcripts.load("s1")) == 4


def test_served_models_are_tracked(transcripts):
    script = [reply(call("echo", text="x"), model="model-a"), reply(TextPart(text="ok"), model="model-b")]
    assert run(transcripts, script)[0].served_models == ["model-a", "model-b"]


def test_schemas_are_inlined_and_keep_title_properties():
    class Inner(Contract):
        title: str

    class Outer(Contract):
        items: list[Inner]

    schema = plain_schema(Outer.model_json_schema())
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text
    assert schema["properties"]["items"]["items"]["properties"]["title"] == {"type": "string"}


def test_cached_input_counts_toward_the_token_limit(transcripts):
    from sim2real.models import Usage

    cached = reply(call("echo", text="x")).model_copy(update={"usage": Usage(input_tokens=1, cache_read_tokens=50)})
    result, _ = run(transcripts, [cached, reply(TextPart(text="ok"))], limits=RunLimits(max_tokens=40))
    assert result.outcome == "token_limit" and result.usage.total == 51
