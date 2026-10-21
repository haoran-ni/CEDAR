"""A hypothesis agent's full cycle with the ScriptedModel and FakeScheduler TEST DOUBLES.

The scripted model plays the agent: it requests a calculation with a control.in it wrote,
registers a prediction, waits, then reads the synthetic output and submits a cited reading.
"""

import re

import pytest
from factories import AGENT, authorize
from sites import aims_output, co_on_cu111, declared, geometry_text
from worlds import World

from sim2real.agents import AgentRunner, TranscriptStore, record_agent_run
from sim2real.agents.prompts import HYPOTHESIS_AGENT
from sim2real.agents.toolkits import FileAccess, ResearchContext, file_tools, research_tools
from sim2real.contracts import (
    CalculationState,
    EvidenceQuality,
    HypothesisStatus,
    PermissionAction,
    SchedulerState,
)
from sim2real.models import ModelSettings, TextPart, ToolCall
from sim2real.models.scripted import ScriptedModel, reply

SETTINGS = ModelSettings(provider="scripted", model="scripted-model")
ENERGY = -4321.987654321


@pytest.fixture
def world(ledger, store, artifacts, site):
    return World(ledger, store, artifacts, site)


@pytest.fixture
def agent(world, tmp_path):
    access = FileAccess(roots=[world.runs], deny=[tmp_path / "config"])
    ctx = ResearchContext(ledger=world.ledger, investigation_id=world.inv, hypothesis_id=world.hyp, actor=AGENT,
                          registry=world.registry, artifacts=world.artifacts, engine=world.engine,
                          validator=world.validator, files=access)
    return research_tools(ctx) + file_tools(access)


def latest_tool_text(call) -> str:
    return "\n".join(p.content for p in call.messages[-1].parts if hasattr(p, "content"))


def calc_id(call) -> str:
    return re.search(r"calc_[0-9a-f]{32}", latest_tool_text(call)).group(0)


def request_args(world, control_text=None, **overrides):
    slab, bottom = co_on_cu111()
    decl = declared(**overrides)
    args = {"protocol_name": "fhiaims.total_energy", "protocol_version": "0.1", "observable_name": "total energy",
            "estimand": "total energy of CO/Cu(111) atop", "system_label": "CO/Cu(111) atop (synthetic)",
            "structure_text": geometry_text(slab, fixed=bottom), "rationale": "baseline total energy", **decl}
    if control_text is not None:
        args["input_files"] = {"control.in": control_text}
    return args


def run_agent(world, agent, tmp_path, script, instruction):
    model = ScriptedModel(list(script))
    runner = AgentRunner(model, SETTINGS, TranscriptStore(tmp_path / "transcripts"))
    result = runner.run("agent-1", system=HYPOTHESIS_AGENT, tools=agent, instruction=instruction)
    record_agent_run(world.ledger, world.inv, AGENT, session_id="agent-1", role="hypothesis_agent",
                     settings=SETTINGS, result=result)
    return result, model


def test_full_cycle(world, agent, tmp_path):
    # Invocation 1: plan and request.
    first_script = [
        reply(ToolCall(id="t1", name="investigation_overview", arguments={}),
              ToolCall(id="t2", name="describe_protocol", arguments={"name": "fhiaims.total_energy", "version": "0.1"})),
        reply(ToolCall(id="t3", name="request_calculation", arguments=request_args(world))),
        lambda call: reply(ToolCall(id="t4", name="register_prediction", arguments={
            "prediction_id": "pred-1", "statement": "The run converges.", "observable_name": "total energy",
            "observable_definition": "final total energy", "observable_unit": "eV",
            "expected_outcome": "a validated total energy", "falsified_if": "the run fails its checks",
            "calculation_ids": [calc_id(call)]})),
        lambda call: reply(TextPart(text="Requested the baseline; waiting."), ToolCall(
            id="t5", name="wait_for_calculations",
            arguments={"calculation_ids": [re.search(r"calc_[0-9a-f]{32}", str(call.messages)).group(0)],
                       "reason": "baseline energy needed"})),
    ]
    result, model = run_agent(world, agent, tmp_path, first_script, "Plan a first test.")
    assert result.outcome == "waiting"
    assert "Physical (declare as intervention or control)" in latest_tool_text(model.calls[1])
    state = world.state()
    (cid,) = state.hypotheses[world.hyp].calculation_ids
    assert state.calculations[cid].state is CalculationState.VALIDATED_INPUT
    assert state.hypotheses[world.hyp].status is HypothesisStatus.WAITING_FOR_JOB
    assert "pred-1" in state.predictions

    # The researcher authorizes; the job runs (fake scheduler, synthetic output).
    authorize(world.ledger, world.inv, PermissionAction.SUBMIT, cid, intervention_id="ok-submit")
    job = world.manager.submit(world.inv, cid)
    output = aims_output(energy=ENERGY)
    (world.runs / state.calculations[cid].workdir / "aims.out").write_text(output)
    world.scheduler.set_state(job, SchedulerState.COMPLETED)
    world.monitor.poll(world.inv)

    # Invocation 2: read the output and report a cited value.
    line = next(i for i, l in enumerate(output.splitlines(), start=1) if "Total energy of the DFT" in l)
    quote = output.splitlines()[line - 1].strip()
    second_script = [
        reply(ToolCall(id="r1", name="run_outputs", arguments={"calculation_id": cid})),
        lambda call: reply(ToolCall(id="r2", name="search_file", arguments={
            "path": re.search(r"(/\S+/aims\.out)", latest_tool_text(call)).group(1), "pattern": "Total energy of"})),
        reply(ToolCall(id="r3", name="submit_reading", arguments={
            "calculation_id": cid, "value": ENERGY, "unit": "eV", "summary": "final total energy",
            "citations": [{"file": "aims.out", "line_start": line, "line_end": line, "quote": quote}]})),
        reply(TextPart(text="The baseline total energy is validated.")),
    ]
    result, model = run_agent(world, agent, tmp_path, second_script, "Your calculation finished.")
    assert result.outcome == "finished"
    assert f"{line}: " in latest_tool_text(model.calls[2])
    assert model.calls[0].messages[0].text == "Plan a first test."  # resumed from the transcript

    state = world.state()
    evidence = state.evidence[state.calculations[cid].evidence_id].evidence
    assert evidence.quality is EvidenceQuality.VALIDATED and evidence.value.value == ENERGY
    assert evidence.interpreted_by == AGENT
    runs = state.agent_runs
    assert [r.run.outcome for r in runs] == ["waiting", "finished"]
    assert runs[1].run.report == "The baseline total energy is validated."
    assert state.model_tokens_used == sum(r.run.input_tokens + r.run.output_tokens + r.run.cache_read_tokens
                                          + r.run.cache_write_tokens for r in runs) > 0


def test_examiner_findings_reach_the_agent(world, agent, tmp_path):
    bad_control = "xc pbe0\nspin none\ncharge 0\nrelativistic atomic_zora scalar\n"
    script = [reply(ToolCall(id="t1", name="request_calculation", arguments=request_args(world, bad_control))),
              reply(TextPart(text="Rejected; will fix."))]
    _, model = run_agent(world, agent, tmp_path, script, "Request with your own control.in.")
    text = latest_tool_text(model.calls[1])
    assert "REJECTED" in text and "contradicts_task" in text and "Fix the errors above" in text


def test_unverifiable_reading_is_refused(world, agent, tmp_path):
    cid = world.request()
    world.run(cid, output=aims_output(energy=ENERGY))
    script = [reply(ToolCall(id="t1", name="submit_reading", arguments={
                  "calculation_id": cid, "value": ENERGY, "unit": "eV", "summary": "made up",
                  "citations": [{"file": "aims.out", "line_start": 1, "line_end": 1, "quote": "not there"}]})),
              reply(TextPart(text="My citation was wrong."))]
    _, model = run_agent(world, agent, tmp_path, script, "Report the energy.")
    result_part = model.calls[1].messages[-1].parts[0]
    assert result_part.is_error and "Reading not accepted" in result_part.content
    assert world.calc(cid).evidence_id is None


@pytest.mark.parametrize("budget, shown", [(None, None), (5000, "Model tokens remaining for this investigation: 5000")])
def test_overview_shows_the_token_budget_when_one_is_set(ledger, store, artifacts, site, budget, shown):
    world = World(ledger, store, artifacts, site, **({"model_token_budget": budget} if budget else {}))
    access = FileAccess(roots=[world.runs])
    ctx = ResearchContext(ledger=world.ledger, investigation_id=world.inv, hypothesis_id=world.hyp, actor=AGENT,
                          registry=world.registry, artifacts=world.artifacts, engine=world.engine,
                          validator=world.validator, files=access)
    overview = {t.name: t for t in research_tools(ctx)}["investigation_overview"].invoke({}).content
    assert "Your hypothesis" in overview
    assert (shown in overview) if shown else ("Model tokens remaining" not in overview)
