"""A hypothesis agent's full cycle with the ScriptedModel and FakeScheduler TEST DOUBLES.

The scripted model plays the agent: it requests a calculation, registers a prediction,
waits, then reads the synthetic output, submits a cited reading, runs an analysis script on
the output and records its result. The analysis script really runs, under its guards.
"""

import re

import pytest
from factories import AGENT, authorize
from sites import aims_output, co_on_cu111, declared, geometry_text
from worlds import World

from cedar.agents import AgentRunner, TranscriptStore, record_agent_run
from cedar.agents.prompts import HYPOTHESIS_AGENT
from cedar.agents.toolkits import FileAccess, ResearchContext, file_tools, research_tools
from cedar.contracts import (
    CalculationState,
    EvidenceKind,
    EvidenceQuality,
    HypothesisStatus,
    PermissionAction,
    SchedulerState,
)
from cedar.models import ModelSettings, TextPart, ToolCall
from cedar.models.scripted import ScriptedModel, reply

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
                          validator=world.validator, files=access, analysis=world.analysis)
    return research_tools(ctx) + file_tools(access)


def latest_tool_text(call) -> str:
    return "\n".join(p.content for p in call.messages[-1].parts if hasattr(p, "content"))


def calc_id(call) -> str:
    return re.search(r"calc_[0-9a-f]{32}", latest_tool_text(call)).group(0)


def request_args(world, control_text=None, **overrides):
    slab, bottom = co_on_cu111()
    decl = declared(**overrides)
    args = {"protocol_name": "fhiaims.run", "protocol_version": "0.3", "observable_name": "total energy",
            "observable_definition": "final total energy printed by FHI-aims", "observable_unit": "eV",
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
              ToolCall(id="t2", name="describe_protocol", arguments={"name": "fhiaims.run", "version": "0.3"})),
        reply(ToolCall(id="t3", name="request_calculation", arguments=request_args(world))),
        lambda call: reply(ToolCall(id="t4", name="register_prediction", arguments={
            "prediction_id": "pred-1", "statement": "The run converges.", "observable_name": "total energy",
            "observable_definition": "final total energy", "observable_unit": "eV",
            "expected_outcome": "a validated total energy", "falsified_if": "the run fails its checks",
            "calculation_ids": [calc_id(call)]})),
        lambda call: reply(TextPart(text="Requested the baseline; waiting."), ToolCall(
            id="t5", name="submit_report",
            arguments={"outcome": "waiting_for_calculations", "summary": "baseline energy needed",
                       "wait_for": [re.search(r"calc_[0-9a-f]{32}", str(call.messages)).group(0)]})),
    ]
    result, model = run_agent(world, agent, tmp_path, first_script, "Plan a first test.")
    assert result.outcome == "waiting"
    assert "No keyword list limits what you may set" in latest_tool_text(model.calls[1])
    assert "The input reviewer, an agent, now reviews the inputs" in latest_tool_text(model.calls[2])
    state = world.state()
    (cid,) = state.hypotheses[world.hyp].calculation_ids
    assert state.calculations[cid].state is CalculationState.SPECIFIED  # waiting for the input reviewer
    assert state.hypotheses[world.hyp].status is HypothesisStatus.WAITING_FOR_JOB
    assert "pred-1" in state.predictions
    assert world.review(cid).content.startswith("Recorded review_")
    assert world.calc(cid).state is CalculationState.VALIDATED_INPUT

    # The researcher authorizes; the job runs (fake scheduler, synthetic output).
    authorize(world.ledger, world.inv, PermissionAction.SUBMIT, cid, intervention_id="ok-submit")
    job = world.manager.submit(world.inv, cid)
    output = aims_output(energy=ENERGY)
    (world.runs / state.calculations[cid].workdir / "aims.out").write_text(output)
    world.scheduler.set_state(job, SchedulerState.COMPLETED)
    world.monitor.poll(world.inv)
    world.validator.check_run(world.inv, cid)  # the dispatcher's job when a run ends

    # Invocation 2: read the output, report a cited value, and compute one with a script.
    line = next(i for i, l in enumerate(output.splitlines(), start=1) if "Total energy of the DFT" in l)
    quote = output.splitlines()[line - 1].strip()
    script = (f"import json, re\nline = next(l for l in open('inputs/{cid}/aims.out') if 'Total energy of the DFT' in l)\n"
              "ev = float(re.findall(r'[-+]?\\d+\\.\\d+', line)[-1])\nprint('matched:', line.strip())\n"
              "json.dump({'results': {'total_energy_ha': {'value': ev / 27.211386245988, 'unit': 'Ha'}},\n"
              "           'checks': {'one_final_energy_line': {'passed': True, 'detail': line.strip()}}},\n"
              "          open('results.json', 'w'))\n")
    second_script = [
        reply(ToolCall(id="r1", name="run_outputs", arguments={"calculation_id": cid})),
        lambda call: reply(ToolCall(id="r2", name="search_file", arguments={
            "path": re.search(r"Files in (/\S+) \(", latest_tool_text(call)).group(1) + "/aims.out",
            "pattern": "Total energy of"})),
        reply(ToolCall(id="r3", name="submit_reading", arguments={
            "calculation_id": cid, "observable_name": "total energy",
            "observable_definition": "final total energy printed by FHI-aims", "value": ENERGY, "unit": "eV",
            "summary": "final total energy",
            "citations": [{"file": "aims.out", "line_start": line, "line_end": line, "quote": quote}]})),
        reply(ToolCall(id="r4", name="run_analysis", arguments={
            "purpose": "final total energy in Hartree", "script": script, "inputs": [{"calculation_id": cid}]})),
        lambda call: reply(ToolCall(id="r5", name="record_result", arguments={
            "analysis_id": re.search(r"Analysis (an_[0-9a-f]{32})", latest_tool_text(call)).group(1),
            "result_name": "total_energy_ha", "observable_name": "total energy in Hartree",
            "observable_definition": "final total energy converted from eV with CODATA 2018",
            "summary": "the printed eV value divided by 27.211386245988"})),
        reply(TextPart(text="The baseline total energy is validated.")),
    ]
    result, model = run_agent(world, agent, tmp_path, second_script, "Your calculation finished.")
    assert result.outcome == "finished"
    assert "run check: healthy" in latest_tool_text(model.calls[1])
    assert f"{line}: " in latest_tool_text(model.calls[2])
    assert "matched: | Total energy of the DFT" in latest_tool_text(model.calls[4])
    assert "passed one_final_energy_line" in latest_tool_text(model.calls[4])
    assert model.calls[0].messages[0].text == "Plan a first test."  # resumed from the transcript

    state = world.state()
    reading, computed = (state.evidence[e].evidence for e in state.calculations[cid].evidence_ids)
    assert reading.quality is EvidenceQuality.VALIDATED and reading.value.value == ENERGY
    assert reading.interpreted_by == AGENT
    assert computed.kind is EvidenceKind.ANALYSIS and computed.quality is EvidenceQuality.VALIDATED
    assert computed.value.unit == "Ha" and computed.value.value == pytest.approx(ENERGY / 27.211386245988)
    assert "script:one_final_energy_line" in {c.name for c in computed.quality_checks}
    runs = [r for r in state.agent_runs if r.role == "hypothesis_agent"]  # the input reviewer's run is there too
    assert [r.run.outcome for r in runs] == ["waiting", "finished"]
    assert runs[1].run.report == "The baseline total energy is validated."
    assert state.model_tokens_used == sum(r.run.input_tokens + r.run.output_tokens + r.run.cache_read_tokens
                                          + r.run.cache_write_tokens for r in state.agent_runs) > 0


def test_examiner_findings_reach_the_agent(world, agent, tmp_path):
    bad_control = "xc pbe0\nspin none\ncharge 0\nrelativistic atomic_zora scalar\n"
    script = [reply(ToolCall(id="t1", name="request_calculation", arguments=request_args(world, bad_control))),
              reply(TextPart(text="Rejected; will fix."))]
    _, model = run_agent(world, agent, tmp_path, script, "Request with your own control.in.")
    text = latest_tool_text(model.calls[1])
    assert "REJECTED" in text and "contradicts_task" in text and "Fix the errors above" in text


def test_unverifiable_reading_is_refused(world, agent, tmp_path):
    cid = world.request()
    world.checked(cid, output=aims_output(energy=ENERGY))
    script = [reply(ToolCall(id="t1", name="submit_reading", arguments={
                  "calculation_id": cid, "observable_name": "total energy", "observable_definition": "final",
                  "value": ENERGY, "unit": "eV", "summary": "made up",
                  "citations": [{"file": "aims.out", "line_start": 1, "line_end": 1, "quote": "not there"}]})),
              reply(TextPart(text="My citation was wrong."))]
    _, model = run_agent(world, agent, tmp_path, script, "Report the energy.")
    result_part = model.calls[1].messages[-1].parts[0]
    assert result_part.is_error and "Reading not accepted" in result_part.content
    assert world.calc(cid).evidence_ids == []


def test_requests_define_their_observable(world, agent, tmp_path):
    undefined = {key: value for key, value in request_args(world).items() if key != "observable_definition"}
    script = [reply(ToolCall(id="t1", name="request_calculation", arguments=undefined)),
              reply(TextPart(text="Refused."))]
    _, model = run_agent(world, agent, tmp_path, script, "Request.")
    assert "observable_definition" in latest_tool_text(model.calls[1])
    assert world.state().calculations == {}


@pytest.mark.parametrize("budget, shown", [("none", None), (None, "Budget left for this investigation's model use: "
                                                                    "$15.00"),
                                           (40, "Budget left for this investigation's model use: $40.00")])
def test_overview_shows_the_budget_left_when_there_is_one(ledger, store, artifacts, site, budget, shown):
    constraints = {"budget_usd": None} if budget == "none" else {"budget_usd": budget} if budget else {}
    world = World(ledger, store, artifacts, site, **constraints)
    access = FileAccess(roots=[world.runs])
    ctx = ResearchContext(ledger=world.ledger, investigation_id=world.inv, hypothesis_id=world.hyp, actor=AGENT,
                          registry=world.registry, artifacts=world.artifacts, engine=world.engine,
                          validator=world.validator, files=access, analysis=world.analysis)
    overview = {t.name: t for t in research_tools(ctx)}["investigation_overview"].invoke({}).content
    assert "Your hypothesis" in overview
    assert (shown in overview) if shown else ("Budget left" not in overview)


def test_new_requests_use_the_reviewed_protocol(world, agent):
    request = {t.name: t for t in agent}["request_calculation"]
    outcome = request.invoke({**request_args(world), "protocol_version": "0.2"})
    assert outcome.is_error and "fhiaims.run 0.2 is kept only for the calculations recorded under it; request " \
                                 "fhiaims.run 0.3" in outcome.content
    described = {t.name: t for t in agent}["describe_protocol"].invoke({"name": "fhiaims.run", "version": "0.1"})
    assert described.content == ("fhiaims.run 0.1 is kept only for the calculations recorded under it; new requests "
                                 "use fhiaims.run 0.3.")
    assert world.state().calculations == {}
