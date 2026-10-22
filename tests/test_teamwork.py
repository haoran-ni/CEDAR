"""The agents as a team: what every role reads, and what the agents see of each other's work.

Everything here is SYNTHETIC: the models are scripted (TEST DOUBLES), the scheduler is the FakeScheduler TEST
DOUBLE, and every directory is under the test's temporary directory.
"""

from pathlib import Path

import pytest
from factories import CHIEF, USER, draft
from sites import aims_output
from test_orchestration import call, make_system
from test_swarm import INV, agents_answer, investigation, propose, says
from worlds import World

from cedar.agents.toolkits.files import agent_access
from cedar.contracts import AgentTaskQueued, Observable, UserInterventionRecorded
from cedar.engine import RunRegistrar
from cedar.models.scripted import ScriptedModel, reply
from cedar.models.types import TextPart
from cedar.summaries import short_id


@pytest.fixture
def system(tmp_path, site):
    return make_system(tmp_path, site)


def tool_result(call_record, name):
    """The text of the latest result the model got back for a call of ``name``."""
    found = [part.content for message in call_record.messages for part in message.parts
             if getattr(part, "call_id", "").startswith(name)]
    if not found:
        raise AssertionError(f"no result for {name}")
    return found[-1]


# -- every role reads any file but credentials ------------------------------------------------------

def test_the_chief_scientist_reads_any_file_but_credentials(system, tmp_path, monkeypatch):
    monkeypatch.setenv("CEDAR_CONFIG_DIR", str(tmp_path / "config"))
    (tmp_path / "config" / "credentials.toml").write_text("api_key = 'x'\n")
    notes = tmp_path / "earlier" / "notes.txt"
    notes.parent.mkdir()
    notes.write_text("PBE prefers fcc\n")
    chat = investigation(system, "hyp-a")
    model = ScriptedModel([
        reply(call("read_file", path=str(notes))),
        reply(call("read_file", path=str(tmp_path / "config" / "credentials.toml"))),
        reply(TextPart(text="Read your notes."))])
    system["clients"].models["chief_scientist"] = model
    assert "Read your notes." in chat.handle("What do my notes say?")
    assert "1: PBE prefers fcc" in tool_result(model.calls[1], "read_file")
    assert "not readable by agents" in tool_result(model.calls[2], "read_file")
    assert {"read_file", "list_directory", "find_files", "view_file"} <= {t.name for t in model.calls[0].tools}


def test_the_judge_and_the_approver_get_the_file_tools(system):
    chat = investigation(system, "hyp-a")
    system["clients"].add("evidence_judge", says("Nothing to assess yet."))
    chat.handle("/status")
    from cedar.dispatch import SYSTEM
    system["workspace"].ledger.record(INV, [draft(SYSTEM, AgentTaskQueued(
        task_id="t-judge", role="evidence_judge", hypothesis_id="hyp-a", instruction="Assess hyp-a."))])
    system["dispatcher"].tick()
    (judged,) = system["clients"].models["evidence_judge"].calls
    assert "read_file" in {t.name for t in judged.tools}


def test_agents_never_read_credentials_but_read_cedars_records(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home" / ".aws").mkdir(parents=True)
    (tmp_path / "home" / ".aws" / "credentials").write_text("secret\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "events.sqlite").write_text("log\n")
    access = agent_access(deny=[tmp_path / "config"])
    with pytest.raises(PermissionError, match="not readable by agents"):
        access.resolve(str(tmp_path / "home" / ".aws" / "credentials"))
    assert access.resolve(str(tmp_path / "data" / "events.sqlite")) == (tmp_path / "data" / "events.sqlite").resolve()


# -- the agents see each other ----------------------------------------------------------------------

def test_agents_are_numbered_in_the_order_they_start(system):
    chat = investigation(system, "hyp-a", "hyp-b")
    ledger = system["workspace"].ledger
    propose(ledger, "p-1", "hyp-a", "hyp-b")
    router = agents_answer(system, **{"hyp-a": [says("A.")], "hyp-b": [says("B.")]})
    chat.handle("/approve agents")
    agents = ledger.load(INV).agents
    assert [(a.agent_id, a.tag) for a in agents.values()] == [("agent-hyp-a", "A1"), ("agent-hyp-b", "A2")]
    system["dispatcher"].tick()
    (first,) = router.models["agent-hyp-a"].calls
    assert "you are A1 (agent-hyp-a)" in first.system and "other_agents" in {t.name for t in first.tools}


def test_a_cedar_run_directory_is_never_registered_as_a_run_from_outside(ledger, store, artifacts, site):
    world = World(ledger, store, artifacts, site)
    calc = world.request()
    world.checked(calc, output=aims_output(energy=-1.0))
    ledger.record(world.inv, [draft(USER, UserInterventionRecorded(intervention_id="msg-1", text="/register x"))])
    outcome = RunRegistrar(ledger, artifacts, profile=site).register(
        world.inv, world.runs / world.calc(calc).workdir, requested_by=USER, description="mine",
        intervention_id="msg-1")
    assert outcome.calculation_id is None and "use that calculation directly" in outcome.problems[0]


# -- each task starts afresh, briefed from the record -----------------------------------------------

def test_each_task_starts_a_new_conversation_briefed_from_the_record(system):
    chat = investigation(system, "hyp-a", "hyp-b")
    ledger = system["workspace"].ledger
    propose(ledger, "p-1", "hyp-a", "hyp-b")
    router = agents_answer(system, **{"hyp-a": [says("A: first look done."), says("A: second look done.")],
                                      "hyp-b": [says("B.")]})
    chat.handle("/approve agents")
    system["dispatcher"].tick()
    ledger.record(INV, [draft(CHIEF, AgentTaskQueued(task_id="t-2", role="hypothesis_agent",
                                                          hypothesis_id="hyp-a", agent_id="agent-hyp-a",
                                                          instruction="Look again."))])
    system["dispatcher"].tick()
    first, second = router.models["agent-hyp-a"].calls
    (briefing,) = second.messages
    text = briefing.text
    assert text.startswith("Briefing from the research record")
    assert "- Question: Why does the computed site preference disagree with the observed site?" in text
    assert "low-coverage measurement: observable 'preferred adsorption site'" in text
    assert "Your earlier reports, the latest last:" in text and "A: first look done." in text
    assert "A2 agent-hyp-b (hyp-b specialist) on hyp-b" in text
    (a_first, _) = [r for r in ledger.load(INV).agent_runs if r.actor_id == "agent-hyp-a"]
    earlier = system["workspace"].transcripts.path(a_first.session_id)
    assert f"your earlier conversations, one JSON message per line, the latest last: {earlier}" in text
    assert earlier.exists() and text.endswith("Look again.")


def test_the_chat_continues_in_a_briefed_conversation_once_it_grows_large(system):
    chat = investigation(system, "hyp-a")
    chat.fresh_after_tokens = 5  # each scripted reply reports 10 input tokens
    model = ScriptedModel([reply(TextPart(text="First answer.")), reply(TextPart(text="Second answer."))])
    system["clients"].models["chief_scientist"] = model
    first_session = chat.chief_session
    chat.handle("What is the question?")
    chat.handle("And what did you say before?")
    assert chat.chief_session != first_session
    (opening,) = model.calls[1].messages
    assert opening.text.startswith("Briefing from the research record")
    assert "[researcher, msg-1] What is the question?" in opening.text and "[you] First answer." in opening.text
    assert str(system["workspace"].transcripts.path(first_session)) in opening.text
    assert opening.text.endswith("[researcher, recorded as msg-2] And what did you say before?")


# -- runs that end early ----------------------------------------------------------------------------

def test_there_is_no_limit_on_model_calls(tmp_path):
    from cedar.agents import AgentRunner, TranscriptStore
    from cedar.models import ModelSettings, ToolCall
    calls = [reply(ToolCall(id=f"c{n}", name="list_directory", arguments={"path": str(tmp_path)})) for n in range(80)]
    model = ScriptedModel([*calls, reply(TextPart(text="Done after 80 steps."))])
    runner = AgentRunner(model, ModelSettings(provider="scripted", model="m"), TranscriptStore(tmp_path / "t"))
    from cedar.agents.toolkits.files import file_tools
    result = runner.run("s", system="x", tools=file_tools(agent_access()), instruction="Look.")
    assert result.outcome == "finished" and result.turns == 81 and result.final_text == "Done after 80 steps."


def wakeable_world(system):
    workspace = system["workspace"]
    world = World(workspace.ledger, workspace.store, workspace.artifacts, system["site"],
                  compute_profile=system["site"].name)
    return world


def record_run(world, agent_id, started_version, report="Requested a baseline."):
    from cedar.contracts import AgentRunRecorded
    from factories import agent_for
    world.ledger.record(world.inv, [draft(agent_for(world.hyp), AgentRunRecorded(
        session_id="s", role="hypothesis_agent", provider="p", requested_model="m", turns=1, input_tokens=1,
        output_tokens=1, cache_read_tokens=0, cache_write_tokens=0, outcome="finished", report=report,
        started_version=started_version))])


def test_an_agent_is_woken_for_a_result_it_did_not_say_it_waits_for(system):
    world = wakeable_world(system)
    calc = world.request()
    record_run(world, "agent-hyp-xc", world.state().version)  # it ended without wait_for_calculations
    assert system["dispatcher"]._wake(world.inv) == []  # nothing has ended yet
    world.checked(calc, output=aims_output(energy=-1.0))
    assert system["dispatcher"]._wake(world.inv) == [f"woke the agent for {world.hyp}"]
    (task,) = world.state().pending_tasks
    assert task.instruction.startswith("Your calculations have ended") and calc in task.instruction
    assert system["dispatcher"]._wake(world.inv) == []  # once is enough: the task is queued


def test_a_result_the_agent_was_told_of_does_not_wake_it(system):
    world = wakeable_world(system)
    calc = world.request()
    world.checked(calc, output=aims_output(energy=-1.0))
    record_run(world, "agent-hyp-xc", world.state().version)  # its briefing already showed the result
    assert system["dispatcher"]._wake(world.inv) == []


def test_a_run_cut_off_by_a_service_error_is_reported_from_the_record_and_continued(system):
    from cedar.models.types import ModelError
    chat = investigation(system, "hyp-a")
    ledger = system["workspace"].ledger
    propose(ledger, "p-1", "hyp-a")

    def overloaded(_call):
        raise ModelError("Anthropic error 529: overloaded", retryable=True)

    router = agents_answer(system, **{"hyp-a": [overloaded, overloaded, overloaded]})
    chat.handle("/approve agents")
    for _ in range(4):
        system["dispatcher"].tick()
    runs = ledger.load(INV).agent_runs
    assert [r.run.outcome for r in runs] == ["model_error"] * 3  # the first try and two continuations
    first = runs[0].run
    assert first.report_by_cedar and first.report.startswith("No report from the agent: it ended model_error "
                                                             "(Anthropic error 529: overloaded) after 0 model calls")
    state = ledger.load(INV)
    chain = [state.tasks[r.run.task_id] for r in runs]
    assert [t.continues for t in chain] == [None, chain[0].task_id, chain[1].task_id]
    assert chain[1].instruction.startswith(f"Continue task {chain[0].task_id}: the run carrying it out was cut off "
                                           "(a temporary model-service error")
    assert state.pending_tasks == [] and len(router.models["agent-hyp-a"].calls) == 3


def test_leaving_investigating_while_agents_work_needs_the_chiefs_confirmation(system):
    from cedar.agents.toolkits.chief import ChiefContext, chief_tools
    investigation(system, "hyp-a")
    warnings = []
    context = ChiefContext(ledger=system["workspace"].ledger, profiles=system["profiles"], registry_for=lambda n: None,
                           actor=CHIEF, investigation_id=INV, running=lambda: ["A1 (agent-hyp-a)"],
                           warn=warnings.append)
    tools = {t.name: t for t in chief_tools(context)}
    refused = tools["set_investigation_status"].invoke({"status": "REVIEWING", "reason": "time to review"})
    assert refused.is_error and "not moved: Moving the investigation to REVIEWING stops A1 (agent-hyp-a)" in refused.content
    assert warnings == [] and system["workspace"].ledger.load(INV).status.value == "INVESTIGATING"
    done = tools["set_investigation_status"].invoke({"status": "REVIEWING", "reason": "time to review",
                                                     "confirm_stopping_agents": True})
    assert not done.is_error and system["workspace"].ledger.load(INV).status.value == "REVIEWING"
    assert warnings == ["The Chief Scientist moved the investigation to REVIEWING, which stops A1 (agent-hyp-a) at "
                        "their next step; their tasks continue when it is INVESTIGATING again."]


def test_work_waiting_for_investigating_does_not_hold_a_review_back(system):
    from cedar.contracts import AgentRunRecorded, InvestigationStatusChanged
    from cedar.swarm import review_due
    chat = investigation(system, "hyp-a")
    ledger = system["workspace"].ledger
    propose(ledger, "p-1", "hyp-a")
    agents_answer(system, **{"hyp-a": [says("A reported.")]})
    chat.handle("/approve agents")
    system["dispatcher"].tick()
    ledger.record(INV, [draft(CHIEF, AgentTaskQueued(task_id="t-later", role="hypothesis_agent", hypothesis_id="hyp-a",
                                                     agent_id="agent-hyp-a", instruction="Later."))])
    assert not review_due(ledger.load(INV))  # it can run now, so the review waits for it
    ledger.record(INV, [draft(CHIEF, InvestigationStatusChanged(status="REVIEWING", reason="review"))])
    assert review_due(ledger.load(INV))  # it cannot run while REVIEWING


# -- structured reports -----------------------------------------------------------------------------

def test_an_agent_ends_its_task_with_a_structured_report(system):
    chat = investigation(system, "hyp-a")
    ledger = system["workspace"].ledger
    propose(ledger, "p-1", "hyp-a")
    agents_answer(system, **{"hyp-a": [
        reply(call("submit_report", outcome="needs_researcher_data", summary="The registered run is not enough.",
                   findings=[{"statement": "The PBE run prefers fcc."}],
                   open_questions=["Which coverage did the experiment use?"],
                   requests=[{"to": "researcher", "text": "Please give the measured coverage."}],
                   next_steps=["Repeat at the measured coverage."]))]})
    chat.handle("/approve agents")
    system["dispatcher"].tick()
    (run,) = ledger.load(INV).agent_runs
    assert run.run.outcome == "reported" and run.run.structured.outcome.value == "needs_researcher_data"
    assert run.run.report.startswith("[needs researcher data] The registered run is not enough.")
    assert "- The PBE run prefers fcc. (no recorded evidence)" in run.run.report
    texts = [n.text for n in chat.notices()]
    assert texts[:2] == ["[A1 agent-hyp-a: needs researcher data] The registered run is not enough.",
                         "[A1 agent-hyp-a: asks you] Please give the measured coverage."]
    assert "A1 (agent-hyp-a) on hyp-a asks: Please give the measured coverage." in chat.handle("/pending")


def test_a_report_cites_only_recorded_evidence(system):
    chat = investigation(system, "hyp-a")
    ledger = system["workspace"].ledger
    propose(ledger, "p-1", "hyp-a")
    router = agents_answer(system, **{"hyp-a": [
        reply(call("submit_report", outcome="concluded", summary="Done.",
                   findings=[{"statement": "It holds.", "evidence_ids": ["ev-invented"]}])),
        reply(call("submit_report", outcome="waiting_for_calculations", summary="Waiting.")),
        says("Giving up on the report.")]})
    chat.handle("/approve agents")
    system["dispatcher"].tick()
    calls = router.models["agent-hyp-a"].calls
    assert "no evidence ev-invented in the record" in tool_result(calls[1], "submit_report")
    assert "waiting_for_calculations" in tool_result(calls[2], "submit_report")
    (run,) = ledger.load(INV).agent_runs
    assert run.run.structured is None and run.run.report == "Giving up on the report."


# -- experimental evidence and preregistration -------------------------------------------------------

def research_tools_for(world, tmp_path):
    from cedar.agents.toolkits import ResearchContext, file_tools, research_tools
    from cedar.engine.external import ExternalEvidenceRecorder
    from factories import AGENT
    access = agent_access(deny=[tmp_path / "config"])
    ctx = ResearchContext(ledger=world.ledger, investigation_id=world.inv, hypothesis_id=world.hyp, actor=AGENT,
                          registry=world.registry, artifacts=world.artifacts, engine=world.engine,
                          validator=world.validator, files=access, analysis=world.analysis, runs_root=world.runs,
                          external=ExternalEvidenceRecorder(world.ledger, world.artifacts,
                                                            readable=lambda p: access.resolve(str(p))))
    return {t.name: t for t in research_tools(ctx) + file_tools(access)}


HEIGHT = {"kind": "measurement", "observable_name": "adsorption height",
          "observable_definition": "height of the molecule above the top surface layer", "value": 2.5, "unit": "Å",
          "summary": "the measured height, as the researcher gave it"}


def test_a_sentence_from_the_researcher_is_experimental_evidence(ledger, store, artifacts, site, tmp_path):
    world = World(ledger, store, artifacts, site)
    ledger.record(world.inv, [draft(USER, UserInterventionRecorded(
        intervention_id="msg-1", text="The experimental value of the adsorption height is 2.5 Å."))])
    tools = research_tools_for(world, tmp_path)
    done = tools["record_external_evidence"].invoke({**HEIGHT, "source_kind": "researcher_message", "source": "msg-1",
                                                      "quote": "the adsorption height is 2.5 Å"})
    assert "VALIDATED, adsorption height = 2.5 Å" in done.content, done.content
    (record,) = world.state().evidence.values()
    ev = record.evidence
    assert ev.kind.value == "published_experiment" and ev.interpreted_by.id == "agent-hyp-xc"
    assert [c.name for c in ev.quality_checks] == ["quote_verified"]
    wrong = tools["record_external_evidence"].invoke({**HEIGHT, "source_kind": "researcher_message", "source": "msg-1",
                                                       "quote": "the adsorption height is 2.7 Å"})
    assert wrong.is_error and "not found, word for word" in wrong.content
    other = tools["record_external_evidence"].invoke({**HEIGHT, "value": 2.6, "source_kind": "researcher_message",
                                                       "source": "msg-1", "quote": "the adsorption height is 2.5 Å"})
    assert other.is_error and "does not contain the value 2.6" in other.content


def test_files_are_checked_and_web_pages_stay_provisional(ledger, store, artifacts, site, tmp_path):
    world = World(ledger, store, artifacts, site)
    paper = tmp_path / "paper.txt"
    paper.write_text("Table 2. The CO adsorption height above Cu(111) is 2.5 Å at 0.1 ML.\n")
    tools = research_tools_for(world, tmp_path)
    from_file = tools["record_external_evidence"].invoke({**HEIGHT, "source_kind": "file", "source": str(paper),
                                                           "quote": "adsorption height above Cu(111) is 2.5 Å",
                                                           "locator": "Table 2", "conditions": {"coverage": "0.1 ML"}})
    assert "VALIDATED" in from_file.content
    from_web = tools["record_external_evidence"].invoke({**HEIGHT, "source_kind": "url",
                                                          "source": "https://example.org/co-cu", "title": "A page",
                                                          "quote": "height 2.5 Å"})
    assert "PROVISIONAL" in from_web.content and "could not be checked" in from_web.content
    filed, webbed = world.state().evidence.values()
    assert any(getattr(s, "sha256", None) == artifacts.put_file(paper).sha256 for s in filed.evidence.sources)
    assert webbed.evidence.quality.value == "PROVISIONAL"


def test_existing_results_challenge_but_never_support(ledger, store, artifacts, site, tmp_path):
    from cedar.contracts import AssessmentRecorded
    from cedar.contracts.status import Assessment, Completeness
    from cedar.errors import RuleViolation
    from factories import JUDGE
    world = World(ledger, store, artifacts, site)
    ledger.record(world.inv, [draft(USER, UserInterventionRecorded(
        intervention_id="msg-1", text="The experimental value of the adsorption height is 2.5 Å."))])
    tools = research_tools_for(world, tmp_path)
    tools["record_external_evidence"].invoke({**HEIGHT, "source_kind": "researcher_message", "source": "msg-1",
                                              "quote": "the adsorption height is 2.5 Å"})
    (evidence_id,) = world.state().evidence

    def assess(assessment):
        return ledger.record(world.inv, [draft(JUDGE, AssessmentRecorded(
            hypothesis_id=world.hyp, assessment=assessment, completeness=Completeness.PARTIAL,
            evidence_ids=(evidence_id,), rationale="test"))])

    with pytest.raises(RuleViolation, match="SUPPORTED needs evidence from a calculation that a prediction"):
        assess(Assessment.SUPPORTED)
    assert assess(Assessment.CHALLENGED).state.hypotheses[world.hyp].assessment is Assessment.CHALLENGED

    from cedar.agents.toolkits.judge import JudgeContext, judge_tools
    dossier = {t.name: t for t in judge_tools(JudgeContext(ledger=ledger, investigation_id=world.inv, actor=JUDGE))}
    text = dossier["hypothesis_dossier"].invoke({"hypothesis_id": world.hyp}).content
    assert "not a test of a prediction (it existed before, or was not predicted)" in text
    assert "- Question: Why does the computed site preference disagree" in text


# -- analyses see only what they stage --------------------------------------------------------------

PEEK = """
import json, os, sqlite3
print("outside visible:", os.path.exists({outside!r}))
try:
    sqlite3.connect("file:{db}?mode=ro", uri=True).execute("select 1").fetchall()
    print("database: read")
except Exception as exc:
    print("database:", type(exc).__name__)
data = open("inputs/data/spectrum.csv").read().split()
json.dump({{"results": {{"peak": {{"value": float(data[-1]), "unit": "cm^-1"}}}}}}, open("results.json", "w"))
"""


def test_an_analysis_sees_only_its_staged_inputs(ledger, store, artifacts, site, tmp_path):
    import sqlite3
    from cedar.contracts import MaterialAttached
    from cedar.engine.analysis import AnalysisRunner
    from cedar.engine.sandbox import bwrap_path
    world = World(ledger, store, artifacts, site)
    outside = tmp_path / "elsewhere.sqlite"
    sqlite3.connect(outside).execute("create table t(x)")
    spectrum = artifacts.put_bytes(b"wavenumber\n2077.0\n", label="spectrum.csv")
    ledger.record(world.inv, [draft(USER, UserInterventionRecorded(intervention_id="msg-1", text="/attach spectrum")),
                              draft(USER, MaterialAttached(material_id="mat-1", name="spectrum.csv",
                                                           artifact=spectrum, intervention_id="msg-1"))])
    for isolate in ([None] if bwrap_path() else []) + [False]:
        runner = AnalysisRunner(ledger, artifacts, world.runs, tmp_path / f"analyses-{isolate}", isolate=isolate)
        report = runner.run(world.inv, world.hyp, purpose="peek", script=PEEK.format(outside=str(outside), db=outside),
                            inputs={}, data={"spectrum.csv": (spectrum, "material mat-1")})
        assert report.run.outcome.value == "completed", report.shown_output
        assert "database: PermissionError" in report.shown_output or "database: OperationalError" in report.shown_output
        if isolate is None:
            assert "outside visible: False" in report.shown_output
            assert "isolated with bubblewrap" in report.run.environment
        else:
            assert "guarded by an audit hook" in report.run.environment
        assert report.run.data[0].origin == "material mat-1" and report.run.result("peak").value.value == 2077.0
    outcome = world.validator.record_analysis_result(
        world.inv, report.analysis_id, "peak", observable=Observable(name="peak", definition="measured CO stretch"),
        summary="the measured peak", reader=world_agent())
    ev = outcome.recorded.events[0].payload.evidence
    assert ev.calculation_ids == () and spectrum in ev.sources
    assert any("data from outside CEDAR's calculations: spectrum.csv (material mat-1)" in t for t in ev.limitations)


def world_agent():
    from factories import AGENT
    return AGENT


# -- the researcher's approvals ---------------------------------------------------------------------

def test_pending_shows_what_a_submission_costs_and_approvals_go_by_agent_or_all(system):
    from cedar.chat import ChatSession
    from factories import add_hypothesis, agent_for
    workspace = system["workspace"]
    world = World(workspace.ledger, workspace.store, workspace.artifacts, system["site"],
                  compute_profile=system["site"].name)
    other = add_hypothesis(world.ledger, world.inv, "hyp-b")
    mine = world.request()
    theirs = world.ledger.request_calculation(world.inv, agent_for(other), other, world.task(
        numerical={"k_grid": [6, 6, 1]}), "denser k-grid").calculation_id
    another = world.request(numerical={"k_grid": [8, 8, 1]})
    for calc in (mine, theirs, another):
        world.engine.examine(world.inv, calc)
    chat = ChatSession(workspace, system["profiles"], system["clients"], investigation_id=world.inv,
                       backend=system["scheduler"], clock=system["clock"])
    pending = chat.handle("/pending")
    exam = world.calc(mine).examination
    assert (f"/approve submit {short_id(mine)}  (A1 agent-hyp-xc on hyp-xc; CO/Cu(111) atop (synthetic): total "
            f"energy of CO/Cu(111) atop; through Slurm, partition {exam.partition}; {exam.requested_tasks} tasks × "
            f"{exam.requested_minutes} min = {exam.requested_core_hours:g} core-hours") in pending
    directory = Path(system["site"].runs_root) / world.calc(mine).workdir
    assert f"examined inputs (control.in, geometry.in, job.sh) in the run directory {directory}" in pending

    answer = chat.handle("/approve submit A2")
    assert answer.startswith("Approved 1 calculation of A2 agent-hyp-b to run") and theirs in answer
    assert chat.handle("/approve submit A2") == "No calculation of A2 agent-hyp-b waits for your approval to run."
    answer = chat.handle("/approve submit all")
    assert answer.startswith("Approved 2 calculations to run") and mine in answer and another in answer
    assert "/approve submit" not in chat.handle("/pending")


# -- budgets in dollars -----------------------------------------------------------------------------

def test_an_agent_whose_model_has_no_known_prices_waits_while_a_budget_applies(system, monkeypatch):
    from cedar.models import ModelSettings
    chat = investigation(system, "hyp-a")  # opened with the default budget of $15
    assert system["workspace"].ledger.load(INV).budget_usd == 15
    propose(system["workspace"].ledger, "p-1", "hyp-a")
    agents_answer(system, **{"hyp-a": [says("A.")]})
    chat.handle("/approve agents")
    monkeypatch.setattr(system["clients"], "settings", lambda role: ModelSettings(provider="scripted", model="unpriced"))
    notes = system["dispatcher"].tick()
    assert (f"{INV}: the prices of scripted unpriced, the hypothesis agent's model, are not known, so its spending "
            "cannot be kept within the budget; set them in models.toml") in notes
    chat.handle("/budget off")
    assert any("hypothesis_agent ran" in n for n in system["dispatcher"].tick())
    (run,) = system["workspace"].ledger.load(INV).agent_runs
    assert run.run.cost_usd is None and system["workspace"].ledger.load(INV).unpriced_runs == 1
