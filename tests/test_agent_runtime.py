"""The agent loop, driven by the ScriptedModel TEST DOUBLE."""

import json

import pytest
from pydantic import Field

from cedar.agents import AgentRunner, AgentTool, RunLimits, ToolFailure, ToolOutcome, TranscriptStore
from cedar.agents.tools import plain_schema
from cedar.contracts.base import Contract
from cedar.agents.runtime import MAX_PAUSED_CONTINUATIONS
from cedar.models import (MediaPart, ModelError, ModelPrices, ModelSettings, TextPart, ToolCall, ToolResult, Usage,
                          WebAccess)
from cedar.models.scripted import ScriptedModel, reply

PRICES = ModelPrices(input=1.0, output=5.0, cache_read=0.1, cache_write=1.25)
"""TEST DOUBLE prices, dollars per million tokens."""
SETTINGS = ModelSettings(provider="scripted", model="scripted-model", prices=PRICES)
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


def run(transcripts, script, log=None, limits=RunLimits(), secrets=(), session="s1", instruction="Do it.",
        settings=SETTINGS):
    model = ScriptedModel(list(script))
    runner = AgentRunner(model, settings, transcripts, secrets=secrets)
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


def test_tool_calls_an_interrupted_run_left_unanswered_are_answered_when_the_session_resumes(transcripts):
    # A bug in a tool, a killed process or a closed chat can end a run between a turn's tool calls and their results;
    # the API refuses a conversation with a tool call left unanswered.
    log = []
    with pytest.raises(RuntimeError, match="bug"):
        run(transcripts, [reply(call("crash", text="x"), call("echo", "c2", text="y"))], log)
    assert log == [] and transcripts.load("s1")[-1].role == "assistant"
    _, model = run(transcripts, [reply(TextPart(text="ok"))], instruction="Next.")
    sent = model.calls[0].messages
    assert [m.role for m in sent] == ["user", "assistant", "user", "user"] and sent[3].text == "Next."
    answers = sent[2].parts
    assert [a.call_id for a in answers] == ["c1", "c2"] and all(a.is_error for a in answers)
    assert "may or may not have taken effect" in answers[0].content
    assert transcripts.load("s1")[2] == sent[2]


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


def test_dropped_reasoning_blocks_add_up_over_a_run(transcripts):
    first = reply(call("echo", text="x")).model_copy(update={"dropped_reasoning_blocks": 3})
    result, _ = run(transcripts, [first, reply(TextPart(text="done"))])
    assert result.dropped_reasoning_blocks == 3


def test_final_report_is_redacted_before_it_is_recorded(transcripts):
    result, _ = run(transcripts, [reply(TextPart(text=f"Done; the key was {KEY}."))], secrets=[KEY])
    assert result.final_text.startswith("Done; the key was ") and KEY not in result.final_text


def test_waiting_tool_ends_the_run(transcripts):
    log = []
    result, model = run(transcripts, [reply(TextPart(text="Waiting for jobs."), call("wait", text="x"))], log)
    assert result.outcome == "waiting" and result.final_text == "Waiting for jobs." and len(model.calls) == 1


def test_limits(transcripts):
    looping = [reply(call("echo", text="again")) for _ in range(5)]
    assert run(transcripts, looping, limits=RunLimits(max_turns=3))[0].outcome == "turn_limit"
    # Each scripted reply costs 10 input and 5 output tokens: $0.000035.
    spent = run(transcripts, looping, limits=RunLimits(max_cost_usd=0.00006), session="s2")[0]
    assert spent.outcome == "budget_limit" and spent.turns == 2 and spent.cost_usd == pytest.approx(0.00007)


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


def test_each_kind_of_token_costs_its_own_price(transcripts):
    from cedar.models import Usage

    usage = Usage(input_tokens=1_000, output_tokens=2_000, cache_read_tokens=500_000, cache_write_tokens=4_000,
                  web_searches=3)
    priced = ModelPrices(input=4.0, output=20.0, cache_read=0.2, cache_write=5.0, web_search=0.01)
    assert priced.cost(usage) == pytest.approx(0.004 + 0.04 + 0.1 + 0.02 + 0.03)
    cached = reply(call("echo", text="x")).model_copy(update={"usage": usage})
    result, _ = run(transcripts, [cached, reply(TextPart(text="ok"))], limits=RunLimits(max_cost_usd=0.1),
                    settings=SETTINGS.model_copy(update={"prices": priced}))
    assert result.outcome == "budget_limit" and result.cost_usd == pytest.approx(0.194)


def test_a_model_without_known_prices_does_not_run_under_a_budget(transcripts):
    unpriced = SETTINGS.model_copy(update={"prices": None})
    result, model = run(transcripts, [reply(TextPart(text="never"))], limits=RunLimits(max_cost_usd=1.0),
                        settings=unpriced)
    assert result.outcome == "budget_limit" and "prices of scripted scripted-model are not known" in result.error
    assert model.calls == []
    free, _ = run(transcripts, [reply(TextPart(text="ok"))], settings=unpriced, session="s2")
    assert free.outcome == "finished" and free.cost_usd is None


def test_published_prices_are_known_for_the_current_models():
    from cedar.models.prices import prices_for
    assert prices_for(ModelSettings(provider="anthropic", model="claude-opus-5-5")).cache_read == 0.2
    assert prices_for(ModelSettings(provider="openai", model="gpt-5.6-sol")).cache_read == 0.4
    assert prices_for(ModelSettings(provider="anthropic", model="claude-opus-9")) is None
    assert prices_for(ModelSettings(provider="openai", model="claude-opus-5-5")) is None


def test_a_run_stops_before_its_next_model_call_when_asked(transcripts):
    model = ScriptedModel([reply(call("echo", text="one")), reply(TextPart(text="never asked"))])
    log, asked = [], []

    def should_stop():
        asked.append(len(model.calls))
        return "the researcher halted it" if log else None

    result = AgentRunner(model, SETTINGS, transcripts).run(
        "s1", system="system prompt", tools=tools(log), instruction="Do it.", should_stop=should_stop)
    assert (result.outcome, result.error, result.turns) == ("stopped", "stopped: the researcher halted it", 1)
    assert asked == [0, 1] and len(model.calls) == 1
    # The transcript ends with the tool's result, so the session can be resumed.
    assert transcripts.load("s1")[-1].role == "user" and tool_results(transcripts)[-1].content == "echo: one"


# -- images, PDFs and the provider's web tools ---------------------------------------------------

FIGURE = MediaPart(media_type="image/png", data="aW1hZ2U=", title="fig.png")


def test_images_a_tool_shows_reach_the_model_with_its_text(transcripts):
    def show(args: EchoArgs) -> ToolOutcome:
        return ToolOutcome(f"Showing {args.text}", media=(FIGURE,))

    model = ScriptedModel([reply(call("show", text="fig.png")), reply(TextPart(text="It shows a tilted molecule."))])
    AgentRunner(model, SETTINGS, transcripts).run("s1", system="system prompt", instruction="Look.",
                                                  tools=[AgentTool("show", "Show.", EchoArgs, show)])
    (result,) = model.calls[1].messages[-1].parts
    assert result.content == "Showing fig.png" and result.media == (FIGURE,)
    assert tool_results(transcripts)[0].media == (FIGURE,)  # kept for the next run of the session


def test_web_access_is_passed_on_and_its_use_counted(transcripts):
    searched = reply(TextPart(text="Found it.")).model_copy(update={"usage": Usage(input_tokens=5, web_searches=2,
                                                                                   web_fetches=1)})
    model = ScriptedModel([searched])
    web = WebAccess(allowed_domains=("arxiv.org",))
    result = AgentRunner(model, SETTINGS, transcripts).run("s1", system="system prompt", tools=[], instruction="Go.",
                                                           web=web)
    assert model.calls[0].web == web
    assert (result.usage.web_searches, result.usage.web_fetches) == (2, 1)
    _, plain = run(transcripts, [reply(TextPart(text="ok"))], session="s2")
    assert plain.calls[0].web is None


def test_a_paused_turn_is_sent_back_unchanged_even_at_the_limits(transcripts):
    paused = reply(TextPart(text="Searching..."), stop="pause_turn")
    model = ScriptedModel([paused, paused, reply(TextPart(text="Done searching."))])
    result = AgentRunner(model, SETTINGS, transcripts).run("s1", system="system prompt", tools=[], instruction="Go.",
                                                           web=WebAccess(), limits=RunLimits(max_turns=1))
    assert (result.outcome, result.final_text, result.turns) == ("finished", "Done searching.", 3)
    # Each continuation ends on the paused assistant turn; no user message is added in between.
    assert [m.role for m in model.calls[2].messages] == ["user", "assistant", "assistant"]


def test_endless_pauses_end_the_run(transcripts):
    paused = reply(TextPart(text="Searching..."), stop="pause_turn")
    model = ScriptedModel([paused] * (MAX_PAUSED_CONTINUATIONS + 1))
    result = AgentRunner(model, SETTINGS, transcripts).run("s1", system="system prompt", tools=[], instruction="Go.",
                                                           web=WebAccess())
    assert result.outcome == "turn_limit" and "paused" in result.error
    assert len(model.calls) == MAX_PAUSED_CONTINUATIONS + 1
