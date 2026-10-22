"""The hypothesis-agent swarm: agents the Chief Scientist proposes and the researcher reviews, one agent
per hypothesis, agents that cannot reach each other's work, agents running at the same time, and the
Chief Scientist's reviews of what they report.

ScriptedModel and FakeScheduler TEST DOUBLES on a SYNTHETIC site; no model or job runs.
"""

import itertools
import os
import threading
from collections import Counter

import pytest
from factories import (
    JUDGE,
    USER,
    add_hypothesis,
    agent_for,
    analysis_evidence,
    analysis_run,
    computed_evidence,
    draft,
    hypothesis_spec,
    start_investigation,
)
from pydantic import ValidationError
from sites import aims_output
from test_orchestration import HARTREE_SCRIPT, make_system
from worlds import World

from cedar.agents import RunLimits
from cedar.agents.toolkits import ResearchContext, agent_access, file_tools, research_tools
from cedar.agents.toolkits.chief import ChiefContext, chief_tools
from cedar.chat import ChatSession
from cedar.contracts import (
    Actor,
    ActorKind,
    AgentRetired,
    AgentRunRecorded,
    AgentSpec,
    AgentsProposed,
    AgentTaskQueued,
    AnalysisRecorded,
    Assessment,
    AssessmentRecorded,
    Completeness,
    EvidenceRecorded,
    HypothesisProposed,
    HypothesisStatus,
    HypothesisStatusChanged,
    InvestigationStatus,
    InvestigationStatusChanged,
    PlanRevised,
    PredictionRegistered,
    ResearchPlan,
    SchedulerState,
    UserInterventionRecorded,
)
from cedar.contracts.hypothesis import Prediction
from cedar.dispatch import SYSTEM, Dispatcher
from cedar.errors import RuleViolation
from cedar.models import TextPart, ToolCall
from cedar.models.scripted import ScriptedModel, reply
from cedar.notebook import render_notebook, round_file
from cedar.state.projection import ProposalStatus, hypothesis_evidence
from cedar.summaries import calculation_summary
from cedar.swarm import CHIEF, REVIEW_INSTRUCTION, review_due, review_material, start_drafts

INV = "swarm-test"
ENERGY = -4321.987654321


@pytest.fixture
def system(tmp_path, site):
    return make_system(tmp_path, site)


class ByAgent:
    """Each model call goes to the script of the agent named in its instructions (TEST DOUBLE).

    It counts how many runs of each agent, and of all agents, are in a model call at once.
    """

    def __init__(self, scripts: dict[str, list]):
        self.models = {agent: ScriptedModel(list(steps)) for agent, steps in scripts.items()}
        self.provider = "scripted"
        self.lock = threading.Lock()
        self.busy: Counter = Counter()
        self.peak: Counter = Counter()
        self.peak_total = 0
        self.order: list[str] = []

    def complete(self, *, system, messages, tools, settings, web=None):
        agent = next(a for a in self.models if f"({a})," in system)
        with self.lock:
            self.busy[agent] += 1
            self.peak[agent] = max(self.peak[agent], self.busy[agent])
            self.peak_total = max(self.peak_total, sum(self.busy.values()))
            self.order.append(agent)
        try:
            return self.models[agent].complete(system=system, messages=messages, tools=tools, settings=settings,
                                               web=web)
        finally:
            with self.lock:
                self.busy[agent] -= 1


def investigation(system, *hypotheses, budget=None, profile=True) -> ChatSession:
    """An investigation on the system's site (or none) with activated hypotheses and no agents yet; the researcher's
    chat."""
    ledger = system["workspace"].ledger
    start_investigation(ledger, INV, compute_profile=system["site"].name if profile else None,
                        **({"budget_usd": budget} if budget is not None else {}))
    for hypothesis in hypotheses:
        add_hypothesis(ledger, INV, hypothesis, agent=False)
    return ChatSession(system["workspace"], system["profiles"], system["clients"], investigation_id=INV,
                       backend=system["scheduler"], clock=system["clock"])


def propose(ledger, proposal_id, *hypotheses, **limits):
    return ledger.record(INV, [draft(CHIEF, AgentsProposed(proposal_id=proposal_id, rationale="test", agents=tuple(
        AgentSpec(hypothesis_id=h, role=f"{h} specialist", brief=f"test {h}", **limits) for h in hypotheses)))])


def agents_answer(system, **scripts) -> ByAgent:
    router = ByAgent({f"agent-{h}": steps for h, steps in scripts.items()})
    system["clients"].models["hypothesis_agent"] = router
    return router


def says(text):
    return reply(TextPart(text=text))


# -- the researcher's review of proposed agents ---------------------------------------------------

def test_proposed_agents_start_after_the_review_window_unless_the_researcher_replies(system):
    ledger, clock, dispatcher = system["workspace"].ledger, system["clock"], system["dispatcher"]
    investigation(system, "hyp-a")
    propose(ledger, "p-1", "hyp-a")
    agents_answer(system, **{"hyp-a": [says("Looked at hyp-a.")]})

    # Nothing starts early, and the record refuses an early start by the system.
    clock.advance(minutes=9)
    assert not any("p-1" in n for n in dispatcher.tick())  # not even attempted
    state = ledger.load(INV)
    assert state.agents == {} and state.pending_proposal.proposal_id == "p-1"
    with pytest.raises(RuleViolation, match="review window"):
        ledger.record(INV, start_drafts(state, state.pending_proposal, SYSTEM))

    # A reply holds the proposal, even once the window has passed.
    ledger.record(INV, [draft(USER, UserInterventionRecorded(intervention_id="msg-1", text="Why not coverage?"))])
    clock.advance(minutes=5)
    assert not any("started" in n for n in dispatcher.tick())
    state = ledger.load(INV)
    assert state.agents == {} and state.pending_proposal.feedback_ids == ["msg-1"]
    with pytest.raises(RuleViolation, match="researcher replied"):
        ledger.record(INV, start_drafts(state, state.pending_proposal, SYSTEM))

    # Proposing again replaces it and opens a new window, after which the agent starts by itself.
    propose(ledger, "p-2", "hyp-a")
    assert ledger.load(INV).proposals["p-1"].status is ProposalStatus.SUPERSEDED
    clock.advance(minutes=9, seconds=59)
    assert not any("started" in n for n in dispatcher.tick())
    clock.advance(seconds=1)
    notes = dispatcher.tick()
    assert f"{INV}: the review window of p-2 passed without a reply; started agent-hyp-a" in notes
    assert any("hypothesis_agent ran" in n and "finished" in n for n in notes)
    state = ledger.load(INV)
    agent = state.agents["agent-hyp-a"]
    assert agent.started_by == SYSTEM and agent.authorization_id is None and agent.spec.role == "hyp-a specialist"
    assert state.proposals["p-2"].status is ProposalStatus.STARTED and state.pending_proposal is None
    (run,) = state.agent_runs
    assert state.tasks[run.run.task_id].instruction == "Your first task, from the Chief Scientist: test hyp-a"

    pages = render_notebook(system["workspace"].store.read_stream(INV))
    assert ("| A1 agent-hyp-a | [hyp-a](hypotheses/hyp-a.md) | hyp-a specialist | review window passed | active | 1 |"
            in pages["index.md"])
    assert "- p-1 (event #" in pages["index.md"] and "replaced by a later proposal; researcher replied: msg-1" \
        in pages["index.md"]
    round_page = pages[round_file(state.round)]
    assert "- Agents proposed by Chief Scientist (p-2): “test”" in round_page
    assert ("- Agents of p-2 started after the researcher's review window passed without a reply: agent-hyp-a on "
            "[hyp-a](hypotheses/hyp-a.md)") in round_page
    assert "- **agent-hyp-a**, hyp-a specialist: started after the review window passed" in pages["hypotheses/hyp-a.md"]
    assert "p-2 started after the review window: agent-hyp-a" in pages["timeline.md"]


def test_the_researcher_can_start_proposed_agents_at_once(system):
    ledger = system["workspace"].ledger
    chat = investigation(system, "hyp-a", "hyp-b")
    assert chat.handle("/approve agents") == "No agents are waiting to start."
    propose(ledger, "p-1", "hyp-a", "hyp-b", budget_usd=2.5)
    agents = chat.handle("/agents")
    assert "Proposed by the Chief Scientist (p-1), starts at" in agents and "unless the researcher replies" in agents
    assert "  - hyp-a: hyp-a specialist (budget $2.50)\n    first task: test hyp-a" in agents
    assert any(n.text.startswith("[waiting for you] /approve agents") for n in chat.notices())

    # Only the researcher's own command starts agents early.
    with pytest.raises(RuleViolation, match="requires a user authorization"):
        ledger.record(INV, start_drafts(ledger.load(INV), ledger.load(INV).pending_proposal, USER))
    assert chat.handle("/approve agents") == ("Started agent-hyp-a on hyp-a, agent-hyp-b on hyp-b (msg-1). The "
                                              "dispatcher runs their first tasks.")
    state = ledger.load(INV)
    assert {a.authorization_id for a in state.agents.values()} == {"msg-1"}
    assert {t.agent_id for t in state.pending_tasks} == {"agent-hyp-a", "agent-hyp-b"}
    assert state.pending_proposal is None and not state.proposals["p-1"].feedback_ids  # a command is not a reply
    assert "Agents:\n  - A1 agent-hyp-a (hyp-a specialist) on hyp-a; $0.00 spent of $2.50" in chat.handle("/agents")


def test_each_hypothesis_has_at_most_one_agent(system):
    ledger = system["workspace"].ledger
    chat = investigation(system, "hyp-a")
    with pytest.raises(ValidationError, match="at most one agent"):
        AgentsProposed(proposal_id="p", rationale="x", agents=(
            AgentSpec(hypothesis_id="hyp-a", role="r", brief="b"), AgentSpec(hypothesis_id="hyp-a", role="s", brief="c")))
    ledger.record(INV, [draft(CHIEF, HypothesisStatusChanged(hypothesis_id="hyp-a", status=HypothesisStatus.PAUSED,
                                                             reason="test"))])
    propose(ledger, "p-1", "hyp-a")
    chat.handle("/approve agents")
    with pytest.raises(RuleViolation, match="already has an agent, agent-hyp-a"):
        propose(ledger, "p-2", "hyp-a")

    ledger.record(INV, [draft(CHIEF, HypothesisProposed(hypothesis_id="hyp-new", spec=hypothesis_spec()))])
    with pytest.raises(RuleViolation, match="agents work on activated hypotheses"):
        propose(ledger, "p-3", "hyp-new")

    # Retiring an agent frees its hypothesis for a new one, which gets a new id; the old one can no longer act.
    ledger.record(INV, [draft(CHIEF, AgentRetired(agent_id="agent-hyp-a", reason="wrong specialization"))])
    assert ledger.load(INV).pending_tasks == []  # its first task was dropped
    propose(ledger, "p-4", "hyp-a")
    chat.handle("/approve agents")
    assert ledger.load(INV).active_agent("hyp-a").agent_id == "agent-hyp-a-2"
    with pytest.raises(RuleViolation, match="agent-hyp-a is not the active agent of hyp-a"):
        ledger.record(INV, [draft(agent_for("hyp-a"), HypothesisStatusChanged(
            hypothesis_id="hyp-a", status=HypothesisStatus.PLANNING_TEST, reason="test"))])

    ledger.record(INV, [draft(CHIEF, InvestigationStatusChanged(status=InvestigationStatus.REVIEWING,
                                                                reason="test"))])
    with pytest.raises(RuleViolation, match="while INVESTIGATING"):
        propose(ledger, "p-5", "hyp-new")


# -- agents record only for their own hypothesis -------------------------------------------------

def test_an_agent_acts_only_on_its_own_hypothesis(ledger, store, artifacts, site):
    world = World(ledger, store, artifacts, site)
    a, b = world.hyp, add_hypothesis(ledger, world.inv, "hyp-b")
    agent_a = agent_for(a)
    b_calc = ledger.request_calculation(world.inv, agent_for(b), b, world.task(), "b's test").calculation_id
    attempts = [
        HypothesisStatusChanged(hypothesis_id=b, status=HypothesisStatus.NEEDS_DATA, reason="test"),
        PredictionRegistered(prediction_id="p-x", hypothesis_id=b, prediction=Prediction(
            statement="s", observable=world.task().observable, expected_outcome="e", falsified_if="f")),
    ]
    for payload in attempts:
        with pytest.raises(RuleViolation, match="agent-hyp-xc is not the active agent of hyp-b"):
            ledger.record(world.inv, [draft(agent_a, payload)])
    with pytest.raises(RuleViolation, match="not the active agent of hyp-b"):
        ledger.request_calculation(world.inv, agent_a, b, world.task(interventions={"xc": "pbe0"}), "test")

    world.checked(b_calc, output=aims_output(energy=ENERGY))
    validator = Actor(kind=ActorKind.VALIDATOR, id="validator")
    # A may read B's run for its own hypothesis; the reading is A's.
    ledger.record(world.inv, [draft(validator, EvidenceRecorded(evidence_id="ev-x", evidence=computed_evidence(
        b_calc).model_copy(update={"interpreted_by": agent_a})))])
    assert [r.evidence_id for r in hypothesis_evidence(ledger.load(world.inv), a)] == ["ev-x"]
    assert hypothesis_evidence(ledger.load(world.inv), b) == []
    ledger.record(world.inv, [draft(Actor(kind=ActorKind.ANALYSIS_RUNNER, id="analysis-runner"), AnalysisRecorded(
        analysis_id="an-b", hypothesis_id=b, run=analysis_run(b_calc, files=dict(world.calc(b_calc).run_check.outputs))))])
    with pytest.raises(RuleViolation, match="analysis an-b was not run for hyp-xc"):
        ledger.record(world.inv, [draft(validator, EvidenceRecorded(evidence_id="ev-y", evidence=analysis_evidence(
            "an-b", b_calc).model_copy(update={"interpreted_by": agent_a})))])
    ledger.record(world.inv, [draft(Actor(kind=ActorKind.ANALYSIS_RUNNER, id="analysis-runner"), AnalysisRecorded(
        analysis_id="an-x", hypothesis_id=a, run=analysis_run(b_calc, files=dict(world.calc(b_calc).run_check.outputs))))])
    with pytest.raises(RuleViolation, match="no active agent agent-nobody"):
        ledger.record(world.inv, [draft(USER, AgentTaskQueued(task_id="t-x", role="hypothesis_agent", hypothesis_id=a,
                                                              agent_id="agent-nobody", instruction="x"))])
    with pytest.raises(RuleViolation, match="works on hyp-b, not hyp-xc"):
        ledger.record(world.inv, [draft(USER, AgentTaskQueued(task_id="t-y", role="hypothesis_agent", hypothesis_id=a,
                                                              agent_id="agent-hyp-b", instruction="x"))])
    with pytest.raises(RuleViolation, match="only the dispatcher queues the Chief Scientist's reviews"):
        ledger.record(world.inv, [draft(CHIEF, AgentTaskQueued(task_id="t-z", role="chief_scientist",
                                                               instruction="review"))])
    ledger.record(world.inv, [draft(USER, AgentTaskQueued(task_id="t-b", role="hypothesis_agent", hypothesis_id=b,
                                                          agent_id="agent-hyp-b", instruction="x"))])
    with pytest.raises(RuleViolation, match="task t-b is for agent-hyp-b, not agent-hyp-xc"):
        ledger.record(world.inv, [draft(agent_a, AgentRunRecorded(
            session_id="s", role="hypothesis_agent", task_id="t-b", provider="p", requested_model="m", turns=1,
            input_tokens=1, output_tokens=1, cache_read_tokens=0, cache_write_tokens=0, outcome="finished"))])


def tools_for(world, hypothesis, actor, tmp_path):
    access = agent_access(deny=[tmp_path / "config"])
    ctx = ResearchContext(ledger=world.ledger, investigation_id=world.inv, hypothesis_id=hypothesis, actor=actor,
                          registry=world.registry, artifacts=world.artifacts, engine=world.engine,
                          validator=world.validator, files=access, analysis=world.analysis, runs_root=world.runs)
    return {tool.name: tool for tool in research_tools(ctx) + file_tools(access)}


def reading(calculation_id, output):
    lines = output.splitlines()
    line = next(i for i, text in enumerate(lines, start=1) if "Total energy of the DFT" in text)
    return {"calculation_id": calculation_id, "observable_name": "total energy",
            "observable_definition": "final total energy printed by FHI-aims", "value": ENERGY, "unit": "eV",
            "summary": "baseline", "citations": [{"file": "aims.out", "line_start": line, "line_end": line,
                                                  "quote": lines[line - 1].strip()}]}


def test_agents_see_and_use_each_others_work(ledger, store, artifacts, site, tmp_path):
    world = World(ledger, store, artifacts, site)
    a, b = world.hyp, add_hypothesis(ledger, world.inv, "hyp-b")
    tools_a = tools_for(world, a, agent_for(a), tmp_path)
    tools_b = tools_for(world, b, agent_for(b), tmp_path)
    theirs = ledger.request_calculation(world.inv, agent_for(b), b, world.task(), "b's baseline").calculation_id
    output = aims_output(energy=ENERGY)
    world.checked(theirs, output=output)
    directory = world.runs / world.calc(theirs).workdir
    assert "Reading recorded as evidence" in tools_b["submit_reading"].invoke(reading(theirs, output)).content
    (b_reading,) = world.state().evidence

    # A sees where B's run is and what B found, reads its files, and uses it for its own hypothesis.
    team = tools_a["other_agents"].invoke({}).content
    assert f"files in {directory}" in team and b_reading in team and "(yours)" in team
    assert not tools_a["read_file"].invoke({"path": str(directory / "aims.out")}).is_error
    assert theirs in tools_a["run_outputs"].invoke({"calculation_id": theirs}).content
    assert "Reading recorded as evidence" in tools_a["submit_reading"].invoke(reading(theirs, output)).content
    assert not tools_a["run_analysis"].invoke({"purpose": "energy in Hartree", "script": HARTREE_SCRIPT.format(
        calc=theirs), "inputs": [{"calculation_id": theirs}]}).is_error
    state = world.state()
    a_reading = next(e for e in state.evidence if e != b_reading)
    assert [r.evidence_id for r in hypothesis_evidence(state, b)] == [b_reading]
    assert [r.evidence_id for r in hypothesis_evidence(state, a)] == [a_reading]
    assert f"{b_reading} for {b}" in calculation_summary(state, state.calculations[theirs])

    # What A records is still only for A: B's analyses are B's to record.
    tools_b["run_analysis"].invoke({"purpose": "energy in Hartree", "script": HARTREE_SCRIPT.format(calc=theirs),
                                    "inputs": [{"calculation_id": theirs}]})
    b_analysis = next(r.analysis_id for r in world.state().analyses.values() if r.hypothesis_id == b)
    outcome = tools_a["record_result"].invoke({"analysis_id": b_analysis, "result_name": "total_energy_ha",
                                               "observable_name": "e", "observable_definition": "e", "summary": "s"})
    assert outcome.is_error and f"{b_analysis} is not one of your analyses" in outcome.content

    # Asking for the identical calculation links the run instead of running it again.
    assert ledger.request_calculation(world.inv, agent_for(a), a, world.task(), "reuse").reused


# -- agents running at the same time ---------------------------------------------------------------

def test_agents_run_at_the_same_time_but_never_twice_at_once(system):
    ledger, dispatcher = system["workspace"].ledger, system["dispatcher"]
    chat = investigation(system, "hyp-a", "hyp-b")
    propose(ledger, "p-1", "hyp-a", "hyp-b")
    chat.handle("/approve agents")
    ledger.record(INV, [draft(CHIEF, AgentTaskQueued(task_id="t-again", role="hypothesis_agent",
                                                     hypothesis_id="hyp-a", agent_id="agent-hyp-a",
                                                     instruction="look again"))])
    together = threading.Barrier(2, timeout=10)

    def meet(text):
        def step(_call):
            together.wait()  # passes only if both agents are in a model call at the same time
            return says(text)
        return step

    router = agents_answer(system, **{"hyp-a": [meet("A first."), says("A again.")], "hyp-b": [meet("B.")]})
    notes = dispatcher.tick()
    assert sum("hypothesis_agent ran" in n and "finished" in n for n in notes) == 2, notes
    assert [t.task_id for t in ledger.load(INV).pending_tasks] == ["t-again"]  # its agent was busy
    dispatcher.tick()
    assert ledger.load(INV).pending_tasks == []
    assert router.peak_total == 2 and router.peak["agent-hyp-a"] == 1
    reports = [r.run.report for r in ledger.load(INV).agent_runs]
    assert sorted(reports) == ["A again.", "A first.", "B."]
    runs = ledger.load(INV).agent_runs
    assert {r.session_id for r in runs} == {f"{INV}.{r.actor_id}.{r.run.task_id}" for r in runs}  # one per task
    assert len({r.session_id for r in runs}) == 3


def test_job_polling_goes_on_while_agents_work(system):
    from test_orchestration import HYP, INV as ORCHESTRATION, agent_requests, latest_calc, open_and_start

    workspace, chat, clients, dispatcher, scheduler = (system[k] for k in (
        "workspace", "chat", "clients", "dispatcher", "scheduler"))
    open_and_start(system)
    clients.add("hypothesis_agent", *agent_requests(system))
    dispatcher.tick()
    calc = latest_calc(workspace)
    chat.handle(f"/approve submit {calc}")
    scheduler.on_submit = lambda job, workdir: (workdir / "aims.out").write_text(aims_output(energy=ENERGY))
    dispatcher.tick()
    workspace.ledger.record(ORCHESTRATION, [draft(CHIEF, AgentTaskQueued(
        task_id="t-slow", role="hypothesis_agent", hypothesis_id=HYP, agent_id=f"agent-{HYP}",
        instruction="Think about the next test."))])
    working, release = threading.Event(), threading.Event()

    def slow(_call):
        working.set()
        assert release.wait(10)
        return says("Thought it over.")

    clients.add("hypothesis_agent", slow, says("Woken."))
    stop, notes = threading.Event(), []
    loop = threading.Thread(target=dispatcher.run_forever, args=(0.02,), kwargs={"log": notes.append, "stop": stop})
    loop.start()
    try:
        assert working.wait(10)
        scheduler.set_state(next(iter(scheduler.jobs)), SchedulerState.COMPLETED)
        for _ in range(500):  # the job ends and its run is checked while the agent is still working
            if workspace.ledger.load(ORCHESTRATION).calculations[calc].run_check is not None:
                break
            threading.Event().wait(0.01)
        assert workspace.ledger.load(ORCHESTRATION).calculations[calc].run_check.healthy
        assert not release.is_set() and (workspace.root / "activity" / "t-slow.json").exists()
    finally:
        release.set()
        stop.set()
        loop.join(10)
    assert not loop.is_alive() and any("ran t-slow" in n and "finished" in n for n in notes)


def test_running_agents_share_the_budget(system):
    ledger, clients = system["workspace"].ledger, system["clients"]
    chat = investigation(system, "hyp-a", "hyp-b", budget=0.05)
    dispatcher = Dispatcher(system["workspace"], system["profiles"], clients, backend=system["scheduler"],
                            clock=system["clock"], chief_reviews=False)
    propose(ledger, "p-1", "hyp-a", "hyp-b")
    chat.handle("/approve agents")
    looking = [reply(ToolCall(id=f"c{n}", name="investigation_overview", arguments={})) for n in range(3)]
    agents_answer(system, **{"hyp-a": [*looking, says("A.")], "hyp-b": [*looking, says("B.")]})
    dispatcher.tick()  # both start: nothing is set aside; each checks what all have spent before every call
    runs = ledger.load(INV).agent_runs
    assert sorted(r.actor_id for r in runs) == ["agent-hyp-a", "agent-hyp-b"]
    assert {r.run.outcome for r in runs} == {"budget_limit"}
    # Each reply costs $0.035: together they overspend $0.05 by at most one call each.
    assert ledger.load(INV).model_cost <= 0.05 + 2 * 0.035 + 1e-9
    ledger.record(INV, [draft(CHIEF, AgentTaskQueued(task_id="t-more", role="hypothesis_agent", hypothesis_id="hyp-a",
                                                     agent_id="agent-hyp-a", instruction="more"))])
    assert f"{INV}: the budget is spent; queued agent work waits until the researcher raises it with /budget" in \
        dispatcher.tick()


def test_an_agent_stops_at_its_own_budget(system):
    ledger, dispatcher = system["workspace"].ledger, system["dispatcher"]
    chat = investigation(system, "hyp-a")
    propose(ledger, "p-1", "hyp-a", budget_usd=0.02)
    chat.handle("/approve agents")
    agents_answer(system, **{"hyp-a": [says("A.")]})
    dispatcher.tick()
    ledger.record(INV, [draft(CHIEF, AgentTaskQueued(task_id="t-more", role="hypothesis_agent", hypothesis_id="hyp-a",
                                                     agent_id="agent-hyp-a", instruction="more"))])
    assert f"{INV}: agent-hyp-a has spent its budget of $0.020; its queued work waits" in dispatcher.tick()
    assert [t.task_id for t in ledger.load(INV).pending_tasks] == ["t-more"]


def test_priority_decides_who_runs_first_and_agents_wait_outside_investigating(system):
    ledger, clients = system["workspace"].ledger, system["clients"]
    chat = investigation(system, "hyp-a", "hyp-b")
    dispatcher = Dispatcher(system["workspace"], system["profiles"], clients, backend=system["scheduler"],
                            clock=system["clock"], chief_reviews=False, max_agents=1)
    propose(ledger, "p-1", "hyp-a", "hyp-b")
    chat.handle("/approve agents")
    ledger.record(INV, [draft(CHIEF, PlanRevised(version=2, reason="b first", plan=ResearchPlan(
        objective="test", hypothesis_priorities={"hyp-b": "high"})))])
    router = agents_answer(system, **{"hyp-a": [says("A.")], "hyp-b": [says("B.")]})

    ledger.record(INV, [draft(CHIEF, InvestigationStatusChanged(status=InvestigationStatus.REVIEWING,
                                                                reason="test"))])
    dispatcher.tick()
    assert router.order == [] and len(ledger.load(INV).pending_tasks) == 2
    ledger.record(INV, [draft(CHIEF, InvestigationStatusChanged(status=InvestigationStatus.INVESTIGATING,
                                                                reason="test"))])
    dispatcher.tick()
    dispatcher.tick()
    assert router.order == ["agent-hyp-b", "agent-hyp-a"]


# -- the Chief Scientist collects and compares --------------------------------------------------------

def test_the_chief_scientist_reviews_once_every_agent_has_reported(tmp_path, site):
    system = make_system(tmp_path, site, reviews=True)
    ledger, clients, dispatcher = system["workspace"].ledger, system["clients"], system["dispatcher"]
    chat = investigation(system, "hyp-a", "hyp-b")
    propose(ledger, "p-1", "hyp-a", "hyp-b")
    chat.handle("/approve agents")
    tasks = {t.agent_id: t.task_id for t in ledger.load(INV).pending_tasks}

    # Not while any agent given work has yet to report.
    ledger.record(INV, [draft(agent_for("hyp-a"), AgentRunRecorded(
        session_id="s", role="hypothesis_agent", task_id=tasks["agent-hyp-a"], provider="p", requested_model="m",
        turns=1, input_tokens=1, output_tokens=1, cache_read_tokens=0, cache_write_tokens=0, outcome="finished",
        report="A found the hollow site lower by a synthetic amount."))])
    assert not review_due(ledger.load(INV))

    agents_answer(system, **{"hyp-b": [says("B found no coverage effect at the synthetic coverages.")]})
    notes = dispatcher.tick()
    assert any("hypothesis_agent ran" in n for n in notes) and review_due(ledger.load(INV))
    clients.add("chief_scientist", says("Both agents reported. Their results do not conflict; I asked for nothing "
                                        "new."))
    notes = dispatcher.tick()
    assert f"{INV}: queued the Chief Scientist's review of the agents' reports" in notes
    (review,) = [r for r in ledger.load(INV).agent_runs if r.role == "chief_scientist"]
    assert review.session_id == f"{INV}.chief-review.{review.run.task_id}"
    instruction = clients.models["chief_scientist"].calls[0].messages[-1].text
    assert "[A1 agent-hyp-a (hyp-a specialist) on hyp-a; run finished]\nA found the hollow site" in instruction
    assert "[A2 agent-hyp-b (hyp-b specialist) on hyp-b; run finished]\nB found no coverage effect" in instruction
    assert "background review" in instruction and "comparing across hypotheses" in instruction
    assert not any("review" in n for n in dispatcher.tick())  # nothing new came in
    assert not review_due(ledger.load(INV))

    # The researcher sees the review in the chat, and the Chief knows it there.
    (shown,) = [n for n in chat.notices() if n.review]
    assert shown.text == "Both agents reported. Their results do not conflict; I asked for nothing new."
    clients.add("chief_scientist", says("Nothing conflicts so far."))
    assert chat.handle("What came in?") == "Nothing conflicts so far."
    asked = clients.models["chief_scientist"].calls[-1].messages[-1].text
    assert asked.startswith("[record: your background review of the agents' reports, run finished]\n"
                            "Both agents reported.")
    assert asked.endswith("[researcher, recorded as msg-2] What came in?")
    assert chat.chief_session == f"{INV}.chief"
    round_page = render_notebook(system["workspace"].store.read_stream(INV))[round_file(ledger.load(INV).round)]
    assert "(chief scientist; review of the agents' reports; task" in round_page


def test_a_review_can_start_another_round_that_builds_on_the_first(tmp_path, site):
    """The review finds the question open: a new hypothesis and agent, a follow-up for an agent, a new round; the
    next review waits for all of them and sees only what is new, with its earlier reviews in mind."""
    system = make_system(tmp_path, site, reviews=True)
    ledger, clients, dispatcher = system["workspace"].ledger, system["clients"], system["dispatcher"]
    chat = investigation(system, "hyp-a", "hyp-b")
    propose(ledger, "p-1", "hyp-a", "hyp-b")
    chat.handle("/approve agents")
    router = agents_answer(system, **{
        "hyp-a": [says("A: the effect halves with method X but remains."),
                  says("A: with B1 controlled, the effect still remains.")],
        "hyp-b": [says("B: confounder B1 explains part of the effect.")],
        "hyp-c": [says("C: tested C with the context I was given.")]})
    dispatcher.tick()
    first_round = ledger.load(INV).round
    ids = itertools.count()

    def chief(*steps):
        return reply(*(ToolCall(id=f"c{next(ids)}", name=name, arguments=arguments) for name, arguments in steps))

    clients.add(
        "chief_scientist",
        chief(("set_investigation_status", {"status": "REVIEWING", "reason": "both reported"}),
              ("propose_hypothesis", {"hypothesis_id": "hyp-c", "title": "C", "statement": "C explains the rest.",
                                      "rationale": "B explains only part of A's effect.",
                                      "falsifiers": ["no effect remains once B1 is controlled"]}),
              ("set_hypothesis_status", {"hypothesis_id": "hyp-c", "status": "SCREENED", "reason": "testable"}),
              ("set_hypothesis_status", {"hypothesis_id": "hyp-c", "status": "ACTIVE", "reason": "worth testing"}),
              ("retire_agent", {"hypothesis_id": "hyp-b", "reason": "B1 found; passed on"}),
              ("recommend_next_actions", {"reason": "Both steps tell the remaining explanations apart.",
                                          "chosen": ["a1", "a2"], "candidates": [
                  {"candidate_id": "a1", "kind": "calculation", "action": "A repeats its test controlling for B1",
                   "hypothesis_ids": ["hyp-a", "hyp-c"], "rationale": "separates B1 from the rest", "cost": "one run",
                   "feasibility": "now", "priority": "high", "priority_reason": "cheapest"},
                  {"candidate_id": "a2", "kind": "calculation", "action": "test C", "hypothesis_ids": ["hyp-c"],
                   "rationale": "C is untested", "cost": "one agent", "feasibility": "after approval",
                   "priority": "high", "priority_reason": "the remaining explanation"}]})),
        lambda call: chief(
            ("set_investigation_status", {"status": "INVESTIGATING", "reason": "an effect remains: test C"}),
            ("delegate", {"hypothesis_id": "hyp-a", "recommendation_id": next(iter(ledger.load(INV).recommendations)),
                          "instruction": "agent-hyp-b found confounder B1. Repeat your test controlling for it."}),
            ("propose_agents", {"rationale": "C may explain what B1 does not",
                                "recommendation_id": next(iter(ledger.load(INV).recommendations)), "agents": [
                                    {"hypothesis_id": "hyp-c", "role": "C specialist",
                                     "brief": "Test C. Context: A saw the effect halve with X; B found confounder "
                                              "B1."}]})),
        says("Not resolved: an effect remains. Round 2: A controls for B1, and I propose an agent for C."))
    dispatcher.tick()
    state = ledger.load(INV)
    assert state.round == first_round + 1  # back to INVESTIGATING from REVIEWING: a new round
    assert not state.agents["agent-hyp-b"].active and state.pending_proposal is not None
    (rec_id,) = state.recommendations
    assert [t.recommendation_id for t in state.pending_tasks] == [rec_id]  # A's follow-up carries out a1
    texts = [n.text for n in chat.notices()]
    review, recommended = texts.index("Not resolved: an effect remains. Round 2: A controls for B1, and I propose "
                                      "an agent for C."), texts.index(next(t for t in texts if rec_id in t))
    assert review < recommended and "Taking: a1, a2." in texts[recommended]

    dispatcher.tick()  # A's follow-up runs at once, in a new conversation briefed from the record
    (follow_up,) = router.models["agent-hyp-a"].calls[-1].messages
    assert "Your earlier reports, the latest last:" in follow_up.text
    assert "A: the effect halves with method X but remains." in follow_up.text
    assert follow_up.text.endswith("agent-hyp-b found confounder B1. Repeat your test controlling for it.")
    assert not any("review" in n for n in dispatcher.tick())  # not while C's agent waits to start
    assert not review_due(ledger.load(INV))

    chat.handle("/approve agents")
    dispatcher.tick()  # the new agent knows only what the Chief wrote
    assert router.models["agent-hyp-c"].calls[0].messages[-1].text.endswith(
        "Test C. Context: A saw the effect halve with X; B found confounder B1.")
    clients.add("chief_scientist", says("A and C both reported; C accounts for the rest."))
    assert f"{INV}: queued the Chief Scientist's review of the agents' reports" in dispatcher.tick()
    assert len([r for r in ledger.load(INV).agent_runs if r.role == "chief_scientist"]) == 2
    second = clients.models["chief_scientist"].calls[-1].messages
    material = second[-1].text
    assert "[A1 agent-hyp-a (hyp-a specialist) on hyp-a; run finished]\nA: with B1 controlled" in material
    assert "[A3 agent-hyp-c (C specialist) on hyp-c; run finished]\nC: tested C" in material
    assert "B: confounder B1" not in material and "the effect halves" not in material  # only what is new
    assert len(second) == 1  # a new conversation, briefed with what its first review concluded
    assert "What your latest background reviews concluded" in material and "Not resolved: an effect remains." in material


def test_proposed_agents_hold_the_review_back_only_while_they_can_start(system):
    ledger = system["workspace"].ledger
    chat = investigation(system, "hyp-a", "hyp-b")
    propose(ledger, "p-1", "hyp-a")
    chat.handle("/approve agents")
    (task,) = ledger.load(INV).pending_tasks
    ledger.record(INV, [draft(agent_for("hyp-a"), AgentRunRecorded(
        session_id="s", role="hypothesis_agent", task_id=task.task_id, provider="p", requested_model="m", turns=1,
        input_tokens=1, output_tokens=1, cache_read_tokens=0, cache_write_tokens=0, outcome="finished",
        report="A reported."))])
    assert review_due(ledger.load(INV))
    propose(ledger, "p-2", "hyp-b")
    assert not review_due(ledger.load(INV))  # B's agent starts soon; both are reviewed together

    def move(status):
        ledger.record(INV, [draft(CHIEF, InvestigationStatusChanged(status=status, reason="test"))])

    move(InvestigationStatus.REVIEWING)
    assert review_due(ledger.load(INV))  # it cannot start while REVIEWING
    move(InvestigationStatus.INVESTIGATING)
    assert not review_due(ledger.load(INV))
    ledger.record(INV, [draft(USER, UserInterventionRecorded(intervention_id="msg-9", text="Not B yet."))])
    assert review_due(ledger.load(INV))  # the researcher's reply holds it


def test_a_review_shows_the_evidence_and_assessments_recorded_since_the_last_one(system):
    from test_orchestration import ENERGY as RUN_ENERGY, HYP, INV as OPEN, agent_reads, agent_requests, latest_calc
    from test_orchestration import open_and_start

    ledger, clients, dispatcher, scheduler = (system["workspace"].ledger, system["clients"], system["dispatcher"],
                                              system["scheduler"])
    open_and_start(system)
    clients.add("hypothesis_agent", *agent_requests(system))
    dispatcher.tick()
    calc = latest_calc(system["workspace"])
    system["chat"].handle(f"/approve submit {calc}")
    output = aims_output(energy=RUN_ENERGY)
    scheduler.on_submit = lambda job, workdir: (workdir / "aims.out").write_text(output)
    dispatcher.tick()
    scheduler.set_state(next(iter(scheduler.jobs)), SchedulerState.COMPLETED)
    clients.add("hypothesis_agent", *agent_reads(system, output))
    dispatcher.tick()
    state = ledger.load(OPEN)
    reading, computed = state.calculations[calc].evidence_ids
    ledger.record(OPEN, [draft(JUDGE, AssessmentRecorded(
        hypothesis_id=HYP, assessment=Assessment.INCONCLUSIVE, completeness=Completeness.PARTIAL,
        evidence_ids=(reading,), rationale="One energy cannot rank adsorption sites."))])

    def review(task_id):
        ledger.record(OPEN, [draft(SYSTEM, AgentTaskQueued(task_id=task_id, role="chief_scientist",
                                                           instruction=REVIEW_INSTRUCTION))])
        state = ledger.load(OPEN)
        return review_material(state, state.tasks[task_id])

    material = review("review-1")
    analysis = ledger.load(OPEN).evidence[computed].evidence
    assert ("Evidence recorded since your last review, as the record holds it:\n"
            f"- {reading} for {HYP} (reading of {calc}): total energy = {RUN_ENERGY!r} eV; VALIDATED\n"
            f"- {computed} for {HYP} (analysis {analysis.analysis_id} of {calc}): total energy in Hartree = "
            f"{analysis.value.value!r} {analysis.value.unit}; VALIDATED") in material
    assert ("The Evidence Judge's assessments since your last review:\n"
            f"- {HYP}: INCONCLUSIVE (partial), citing {reading}: One energy cannot rank adsorption sites.") in material
    assert "A report is its agent's account, not evidence" in material
    later = review("review-2")  # nothing was recorded since the first
    assert "Evidence recorded since your last review, as the record holds it:\nNone." in later
    assert "The Evidence Judge's assessments since" not in later


def test_a_review_proposes_closing_and_only_the_researcher_closes(tmp_path, site):
    """The Chief Scientist's review proposes closing and says so; the researcher's /approve close closes it."""
    system = make_system(tmp_path, site, reviews=True)
    ledger, clients, dispatcher = system["workspace"].ledger, system["clients"], system["dispatcher"]
    chat = investigation(system, "hyp-a")
    propose(ledger, "p-1", "hyp-a")
    chat.handle("/approve agents")
    agents_answer(system, **{"hyp-a": [says("A answered its question.")]})
    dispatcher.tick()

    def chief(call_id, name, **arguments):
        return reply(ToolCall(id=call_id, name=name, arguments=arguments))

    message = "I propose closing the investigation as COMPLETED: A answered it. /approve close closes it."
    clients.add("chief_scientist", chief("c1", "set_investigation_status", status="REVIEWING", reason="A reported"),
                chief("c2", "propose_closing", status="COMPLETED", reason="A answered the question."),
                chief("c3", "record_synthesis", summary="Only A was tested; it answers the question.",
                      hypotheses=[{"hypothesis_id": "hyp-a", "standing": "answered by A's test"}],
                      residual="nothing known remains"),
                chief("c4", "propose_closing", status="COMPLETED", reason="A answered the question."), says(message))
    dispatcher.tick()
    state = ledger.load(INV)
    assert state.status is InvestigationStatus.REVIEWING
    closing = state.pending_closing
    assert (closing.status, closing.reason) == (InvestigationStatus.COMPLETED, "A answered the question.")
    (review,) = [r for r in state.agent_runs if r.role == "chief_scientist"]
    assert (review.run.outcome, review.run.turns) == ("finished", 5)
    first_try = clients.models["chief_scientist"].calls[2].messages[-1].parts[0].content
    assert "closing needs a comparison across hypotheses (record_synthesis)" in first_try
    assert "If you proposed closing the investigation, say so first" in \
        clients.models["chief_scientist"].calls[0].messages[-1].text

    # The researcher sees the Chief's message, and the request for approval whatever that message says.
    notices = [n.text for n in chat.notices()]
    assert message in notices
    assert (f"[waiting for you] /approve close  (the Chief Scientist proposes closing the investigation as COMPLETED "
            f"({closing.closing_id}): A answered the question.)") in notices

    answer = chat.handle("/approve close")
    assert answer.splitlines() == [
        f"{INV} is now COMPLETED, closed for good (msg-2). The Chief Scientist proposed it ({closing.closing_id}): "
        "A answered the question.",
        "  - retired agent-hyp-a"]
    state = ledger.load(INV)
    assert state.status is InvestigationStatus.COMPLETED and state.pending_closing is None
    assert not state.agents["agent-hyp-a"].active
    assert chat.handle("/pending") == "Nothing waits for your approval."
    assert chat.handle("/approve close").startswith("Nothing to close")
    pages = render_notebook(system["workspace"].store.read_stream(INV))
    assert f"{closing.closing_id}: closing proposed as COMPLETED: “A answered the question.”" in pages["timeline.md"]
    assert "Investigation moved to COMPLETED, authorized by msg-2" in pages["timeline.md"]
    assert f"- Closing as COMPLETED proposed by Chief Scientist ({closing.closing_id})" in \
        pages[round_file(state.round)]
    assert "Comparison across hypotheses by Chief Scientist" in pages[round_file(state.round)]
    assert "- Unexplained: “nothing known remains”" in pages[round_file(state.round)]


def test_closing_needs_the_chiefs_proposal_and_the_researchers_command(system):
    ledger = system["workspace"].ledger
    chat = investigation(system, "hyp-a")
    tools = {t.name: t for t in chief_tools(ChiefContext(
        ledger=ledger, profiles=system["profiles"], registry_for=lambda name: None, actor=CHIEF, investigation_id=INV,
        clock=system["clock"]))}

    def use(name, **arguments):
        return tools[name].invoke(arguments)

    def close(actor, status):
        ledger.record(INV, [draft(actor, InvestigationStatusChanged(status=status, reason="done"))])

    assert use("set_investigation_status", status="COMPLETED", reason="done").is_error  # not the Chief's to set
    outcome = use("propose_closing", status="COMPLETED", reason="done")
    assert outcome.is_error and "closing is proposed while REVIEWING, not INVESTIGATING" in outcome.content
    assert chat.handle("/approve close").startswith("Nothing to close")

    use("set_investigation_status", status="REVIEWING", reason="reports in")
    with pytest.raises(RuleViolation, match="only the researcher closes an investigation"):
        close(CHIEF, InvestigationStatus.COMPLETED)
    refused = use("propose_closing", status="INCONCLUSIVE", reason="the budget is spent")
    assert refused.is_error and "closing needs a comparison across hypotheses" in refused.content
    missing = use("record_synthesis", summary="s", hypotheses=[{"hypothesis_id": "hyp-x", "standing": "?"}],
                  residual="all of it")
    assert missing.is_error and "unknown hypotheses hyp-x" in missing.content
    assert not use("record_synthesis", summary="Untested.", residual="all of it",
                   hypotheses=[{"hypothesis_id": "hyp-a", "standing": "never tested"}]).is_error
    assert "as INCONCLUSIVE" in use("propose_closing", status="INCONCLUSIVE", reason="the budget is spent").content
    with pytest.raises(RuleViolation, match="no proposal to close the investigation as COMPLETED"):
        close(USER, InvestigationStatus.COMPLETED)
    with pytest.raises(RuleViolation, match="requires a user authorization naming it"):
        close(USER, InvestigationStatus.INCONCLUSIVE)  # the researcher's command carries one

    use("set_investigation_status", status="INVESTIGATING", reason="one more test after all")
    assert ledger.load(INV).pending_closing is None  # any other status change withdraws the proposal
    assert chat.handle("/approve close").startswith("Nothing to close")


def test_closing_waits_for_jobs_the_dispatcher_still_follows(system):
    from test_orchestration import INV as OPEN

    from cedar.contracts import ClosingProposed

    from cedar.contracts import HypothesisStanding, SynthesisRecorded

    calc = halted_system(system)
    ledger = system["workspace"].ledger
    open_ones = [h for h, hyp in ledger.load(OPEN).hypotheses.items() if hyp.status.value not in ("DISCARDED",
                                                                                                  "SUPERSEDED")]
    ledger.record(OPEN, [draft(CHIEF, InvestigationStatusChanged(status=InvestigationStatus.REVIEWING, reason="r")),
                         draft(CHIEF, SynthesisRecorded(synthesis_id="synth-1", summary="s", residual="r", hypotheses=tuple(
                             HypothesisStanding(hypothesis_id=h, standing="s") for h in open_ones))),
                         draft(CHIEF, ClosingProposed(closing_id="close-1", status=InvestigationStatus.COMPLETED,
                                                      reason="done"))])
    answer = system["chat"].handle("/approve close")
    assert answer == (f"The investigation could not be closed: {calc} still at the scheduler or waiting for a run "
                      "check; wait for them, or cancel them, before closing")
    assert ledger.load(OPEN).status is InvestigationStatus.REVIEWING


def test_halt_still_stops_a_review_in_progress(tmp_path, site, monkeypatch):
    import cedar.chat

    monkeypatch.setattr(cedar.chat, "stop_dispatcher", lambda ws, timeout, force: "Stopped.")
    system = make_system(tmp_path, site, reviews=True)
    ledger, clients, dispatcher = system["workspace"].ledger, system["clients"], system["dispatcher"]
    chat = investigation(system, "hyp-a")
    propose(ledger, "p-1", "hyp-a")
    chat.handle("/approve agents")
    agents_answer(system, **{"hyp-a": [says("A answered its question.")]})
    dispatcher.tick()

    def halt_while_thinking(_call):
        chat.handle("/halt")  # the researcher, while the review is in a model call
        return reply(ToolCall(id="c1", name="investigation_state", arguments={}))

    clients.add("chief_scientist", halt_while_thinking, says("never asked"))
    dispatcher.tick()
    (review,) = [r for r in ledger.load(INV).agent_runs if r.role == "chief_scientist"]
    assert review.run.outcome == "stopped" and review.run.report_by_cedar
    assert review.run.report.startswith("No report from the agent: it ended stopped (stopped: the investigation is "
                                        "PAUSED) after 1 model call")
    assert len(clients.models["chief_scientist"].calls) == 1


def test_the_chief_scientist_proposes_delegates_and_retires_agents(system):
    ledger = system["workspace"].ledger
    chat = investigation(system, "hyp-a", "hyp-b")
    tools = {t.name: t for t in chief_tools(ChiefContext(
        ledger=ledger, profiles=system["profiles"], registry_for=lambda name: None, actor=CHIEF, investigation_id=INV,
        clock=system["clock"]))}

    def use(name, **arguments):
        return tools[name].invoke(arguments)

    outcome = use("delegate", hypothesis_id="hyp-a", instruction="go")
    assert outcome.is_error and "hyp-a has no agent; propose one with propose_agents" in outcome.content
    outcome = use("propose_agents", rationale="XC first; coverage can wait",
                  agents=[{"hypothesis_id": "hyp-a", "role": "exchange-correlation specialist",
                           "brief": "Compare PBE and PBE0 site energies.", "budget_usd": 3}])
    assert not outcome.is_error and "10 minutes from now) unless the researcher replies" in outcome.content
    outcome = use("delegate", hypothesis_id="hyp-a", instruction="go")
    assert outcome.is_error and "proposed but has not started" in outcome.content
    listing = use("agents").content
    assert "  - hyp-a: exchange-correlation specialist (budget $3.00)" in listing
    assert listing.endswith("Open hypotheses without an agent: hyp-b")
    first = ledger.load(INV).pending_proposal.proposal_id
    outcome = use("propose_agents", rationale="both after all", agents=[
        {"hypothesis_id": "hyp-a", "role": "exchange-correlation specialist", "brief": "Compare functionals."},
        {"hypothesis_id": "hyp-b", "role": "coverage specialist", "brief": "Vary the coverage."}])
    assert f"It replaces {first}, which never started." in outcome.content

    chat.handle("/approve agents")
    assert not use("delegate", hypothesis_id="hyp-a", instruction="Now relax the slab.").is_error
    assert "Retired A1 agent-hyp-a (exchange-correlation specialist)" in use(
        "retire_agent", hypothesis_id="hyp-a", reason="its question is settled").content
    state = ledger.load(INV)
    assert [t.agent_id for t in state.pending_tasks] == ["agent-hyp-b"]  # hyp-a's queued work was dropped
    assert state.agents["agent-hyp-a"].retire_reason == "its question is settled"
    use("set_investigation_status", status="REVIEWING", reason="test")
    outcome = use("propose_agents", rationale="x", agents=[{"hypothesis_id": "hyp-a", "role": "r", "brief": "b"}])
    assert outcome.is_error and "move it to INVESTIGATING before proposing agents" in outcome.content


def test_writers_working_at_the_same_time_all_get_through(ledger):
    """Agents running in parallel, the dispatcher and the chat append to one investigation at once."""
    investigation = start_investigation(ledger)

    def write(writer):
        for n in range(10):
            ledger.record_with(investigation, lambda state: [draft(USER, UserInterventionRecorded(
                intervention_id=f"w{writer}-{n}-{state.version}", text=f"writer {writer}, message {n}"))])

    threads = [threading.Thread(target=write, args=(w,)) for w in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    state = ledger.load(investigation)
    assert len(state.interventions) == 60
    assert sorted({i.text for i in state.interventions.values()}) == sorted(
        f"writer {w}, message {n}" for w in range(6) for n in range(10))


def test_agents_work_without_a_compute_profile(system):
    """Planning, predictions and reports need no compute profile; only calculations do."""
    ledger, dispatcher = system["workspace"].ledger, system["dispatcher"]
    chat = investigation(system, "hyp-a", "hyp-b", profile=False)
    propose(ledger, "p-1", "hyp-a", "hyp-b")
    chat.handle("/approve agents")
    asked = {"protocol_name": "fhiaims.run", "protocol_version": "0.1", "observable_name": "e",
             "observable_definition": "e", "estimand": "e", "system_label": "s", "structure_text": "x",
             "rationale": "r"}
    router = agents_answer(system, **{
        "hyp-a": [says("Planned a test.")],
        "hyp-b": [lambda call: reply(ToolCall(id="c1", name="request_calculation", arguments=asked)),
                  says("No compute here; I registered nothing.")]})
    notes = dispatcher.tick()
    assert sum("hypothesis_agent ran" in n and "finished" in n for n in notes) == 2, notes
    refused = router.models["agent-hyp-b"].calls[1].messages[-1].parts[0]
    assert refused.is_error and "no approved compute profile" in refused.content
    assert not any("compute profile" in n for n in notes)  # nothing to submit, so nothing to complain about


# -- the dispatcher starts when there is work -------------------------------------------------------

def test_what_counts_as_work_for_a_dispatcher(system):
    from cedar.dispatch import needs_dispatcher

    ledger = system["workspace"].ledger
    chat = investigation(system, "hyp-a")
    assert not needs_dispatcher(ledger.load(INV))  # hypotheses alone: nothing to carry out
    propose(ledger, "p-1", "hyp-a")
    assert needs_dispatcher(ledger.load(INV))  # its review window must be watched
    ledger.record(INV, [draft(USER, UserInterventionRecorded(intervention_id="msg-9", text="Wait."))])
    assert not needs_dispatcher(ledger.load(INV))  # held for the Chief Scientist and the researcher
    propose(ledger, "p-2", "hyp-a")
    chat.handle("/approve agents")
    assert needs_dispatcher(ledger.load(INV))  # the agent's first task
    ledger.record(INV, [draft(CHIEF, PlanRevised(version=2, reason="hold", plan=ResearchPlan(
        objective="test", hypothesis_priorities={"hyp-a": "on_hold"})))])
    assert not needs_dispatcher(ledger.load(INV))  # on hold: nothing may run
    ledger.record(INV, [draft(CHIEF, InvestigationStatusChanged(status=InvestigationStatus.CANCELLED,
                                                                reason="test"))])
    assert not needs_dispatcher(ledger.load(INV))


def test_the_chat_starts_a_dispatcher_only_when_there_is_work_and_none_runs(system, monkeypatch):
    import subprocess
    import sys

    import cedar.chat

    ledger, workspace = system["workspace"].ledger, system["workspace"]
    started = []
    chat = investigation(system, "hyp-a")
    chat.auto_dispatch, chat.start_dispatcher = True, lambda ws: started.append(ws) or 4242
    assert chat.ensure_dispatcher() is None and started == []  # nothing to do yet
    propose(ledger, "p-1", "hyp-a")
    note = chat.ensure_dispatcher()
    assert note.startswith("[dispatcher] Started in the background (process 4242)") and len(started) == 1
    assert "stops by itself after 30 minutes with nothing to do" in note
    assert chat.ensure_dispatcher() is None and len(started) == 1  # not restarted within a minute

    monkeypatch.setattr(cedar.chat, "RESTART_SECONDS", 0)
    holder = subprocess.Popen(  # another process holds the lock, as a running dispatcher does
        [sys.executable, "-c", "import fcntl, sys; f = open(sys.argv[1], 'a'); fcntl.lockf(f, fcntl.LOCK_EX); "
         "print('locked', flush=True); sys.stdin.read()", str(workspace.root / "dispatcher.lock")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "locked"
        assert chat.ensure_dispatcher() is None and len(started) == 1
    finally:
        holder.communicate("", timeout=10)
    assert chat.ensure_dispatcher() is not None and len(started) == 2  # it stopped while work remained
    chat.auto_dispatch = False
    assert chat.ensure_dispatcher() is None and len(started) == 2


def test_an_idle_dispatcher_stops_by_itself_but_not_while_there_is_work(system):
    workspace, dispatcher = system["workspace"], system["dispatcher"]
    notes = []
    idle = threading.Thread(target=dispatcher.run_forever, args=(0.01,),  # an empty data directory
                            kwargs={"log": notes.append, "idle_seconds": 0.05}, daemon=True)
    idle.start()
    idle.join(10)
    assert not idle.is_alive(), "an idle dispatcher should stop by itself"
    assert any("nothing to do for" in n for n in notes) and not (workspace.root / "dispatcher.pid").exists()

    investigation(system, "hyp-a")
    propose(workspace.ledger, "p-1", "hyp-a")  # a review window to watch
    stop = threading.Event()
    loop = threading.Thread(target=dispatcher.run_forever, args=(0.01,),
                            kwargs={"log": notes.append, "stop": stop, "idle_seconds": 0.05})
    loop.start()
    try:
        loop.join(0.5)
        assert loop.is_alive() and (workspace.root / "dispatcher.pid").read_text() == str(os.getpid())
    finally:
        stop.set()
        loop.join(10)
    assert not loop.is_alive()


def test_a_background_dispatcher_runs_detached_and_stops_on_request(tmp_path):
    """A real process, with the tests' isolated configuration and no API key."""
    import signal

    from cedar.dispatch import dispatcher_running, start_background_dispatcher, stop_dispatcher
    from cedar.models import write_starter
    from cedar.paths import config_dir
    from cedar.workspace import Workspace

    write_starter(config_dir() / "models.toml")
    workspace = Workspace.open(tmp_path / "background")
    pid = start_background_dispatcher(workspace, idle_minutes=5)
    try:
        for _ in range(300):
            if dispatcher_running(workspace) and (workspace.root / "dispatcher.pid").exists():
                break
            threading.Event().wait(0.1)
        assert dispatcher_running(workspace)
        assert (workspace.root / "dispatcher.pid").read_text() == str(pid)
        assert stop_dispatcher(workspace, timeout=30) == f"Stopped the dispatcher (process {pid})."
        assert not dispatcher_running(workspace) and not (workspace.root / "dispatcher.pid").exists()
        assert "stops after 5 min with nothing to do" in (workspace.root / "dispatcher.log").read_text()
        assert stop_dispatcher(workspace) == "No dispatcher is running on this data directory."
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


# -- the researcher halts everything --------------------------------------------------------------

def halted_system(system):
    """The orchestration story up to a submitted job, plus a second hypothesis whose agent is proposed."""
    from test_orchestration import agent_requests, latest_calc, open_and_start

    open_and_start(system)
    system["clients"].add("hypothesis_agent", *agent_requests(system))
    system["dispatcher"].tick()
    calc = latest_calc(system["workspace"])
    system["chat"].handle(f"/approve submit {calc}")
    system["dispatcher"].tick()
    return calc


def test_halt_cancels_the_jobs_retires_the_agents_and_pauses(system, monkeypatch):
    from test_orchestration import HYP, INV as OPEN

    import cedar.chat
    from cedar.contracts import CalculationState
    from cedar.dispatch import needs_dispatcher
    from cedar.state.projection import ProposalStatus

    stops = []
    monkeypatch.setattr(cedar.chat, "stop_dispatcher", lambda ws, timeout, force: stops.append(force) or
                        "Stopped the dispatcher (process 4242).")
    ledger, chat, scheduler = system["workspace"].ledger, system["chat"], system["scheduler"]
    calc = halted_system(system)
    job = ledger.load(OPEN).calculations[calc].scheduler_job_id
    assert scheduler.jobs[job]["state"] is not SchedulerState.CANCELLED
    add_hypothesis(ledger, OPEN, "hyp-coverage", agent=False)
    ledger.record(OPEN, [draft(CHIEF, AgentsProposed(proposal_id="p-coverage", rationale="test", agents=(
        AgentSpec(hypothesis_id="hyp-coverage", role="coverage specialist", brief="vary coverage"),)))])

    reply_text = chat.handle("/halt the test went wrong")
    message = next(m for m in reversed(ledger.load(OPEN).interventions.values()))
    assert message.text == "/halt the test went wrong"
    assert reply_text.splitlines() == [
        f"{OPEN}: halted ({message.intervention_id}).",
        f"  - cancelled 1 job: {calc} (job {job})",
        f"  - retired agent-{HYP}; a run in progress stops at its next model call",
        "  - withdrew the proposal p-coverage",
        "  - paused: nothing new starts until you ask the Chief Scientist to resume it",
        "Stopped the dispatcher (process 4242)."]
    state = ledger.load(OPEN)
    assert scheduler.jobs[job]["state"] is SchedulerState.CANCELLED
    assert state.calculations[calc].state is CalculationState.CANCELLED
    assert message.authorizations[0].target_ids == (calc,)
    assert not any(a.active for a in state.agents.values()) and state.pending_tasks == []
    assert state.proposals["p-coverage"].status is ProposalStatus.WITHDRAWN
    assert state.status is InvestigationStatus.PAUSED and not needs_dispatcher(state)
    assert stops == [True]
    reason = f"halted by the researcher ({message.intervention_id}): the test went wrong"
    assert state.agents[f"agent-{HYP}"].retire_reason == reason
    assert f"p-coverage withdrawn: “{reason}”" in render_notebook(system["workspace"].store.read_stream(OPEN))[
        "timeline.md"]


def test_a_paused_investigation_submits_nothing(system):
    from test_orchestration import agent_requests, latest_calc, open_and_start
    from test_orchestration import INV as OPEN

    open_and_start(system)
    system["clients"].add("hypothesis_agent", *agent_requests(system))
    system["dispatcher"].tick()
    calc = latest_calc(system["workspace"])
    system["chat"].handle(f"/approve submit {calc}")
    system["workspace"].ledger.record(OPEN, [draft(CHIEF, InvestigationStatusChanged(
        status=InvestigationStatus.PAUSED, reason="test"))])
    assert not any("submitted" in n for n in system["dispatcher"].tick())
    assert system["scheduler"].submissions == []
    from cedar.dispatch import needs_dispatcher
    assert not needs_dispatcher(system["workspace"].ledger.load(OPEN))  # authorized, but nothing to do while paused


def test_halt_leaves_other_investigations_alone_and_halt_all_stops_everything(system, monkeypatch):
    from test_orchestration import INV as OPEN

    import cedar.chat

    stops = []
    monkeypatch.setattr(cedar.chat, "stop_dispatcher", lambda ws, timeout, force: stops.append(force) or
                        "Stopped the dispatcher (process 4242).")
    ledger, chat = system["workspace"].ledger, system["chat"]
    halted_system(system)
    investigation(system, "hyp-a")  # another investigation, whose proposal needs the dispatcher
    propose(ledger, "p-1", "hyp-a")

    assert chat.handle("/halt").splitlines()[-1] == f"The background dispatcher keeps running for {INV}."
    assert stops == [] and ledger.load(INV).status is InvestigationStatus.INVESTIGATING
    assert "already" not in chat.handle("/halt all")
    assert ledger.load(INV).status is InvestigationStatus.PAUSED and stops == [True]
    assert ledger.load(INV).proposals["p-1"].status.value == "withdrawn"
    assert ledger.load(OPEN).status is InvestigationStatus.PAUSED  # still paused, not halted twice


def test_halt_stops_an_agent_run_in_progress_at_its_next_model_call(system, monkeypatch):
    import cedar.chat

    monkeypatch.setattr(cedar.chat, "stop_dispatcher", lambda ws, timeout, force: "Stopped.")
    ledger, dispatcher = system["workspace"].ledger, system["dispatcher"]
    chat = investigation(system, "hyp-a")
    propose(ledger, "p-1", "hyp-a")
    chat.handle("/approve agents")

    def halt_while_thinking(_call):
        chat.handle("/halt")  # the researcher, while the agent is in a model call
        return reply(ToolCall(id="c1", name="investigation_overview", arguments={}))

    router = agents_answer(system, **{"hyp-a": [halt_while_thinking, says("never asked")]})
    notes = dispatcher.tick()
    assert any("ran" in n and "stopped" in n for n in notes), notes
    (run,) = [r for r in ledger.load(INV).agent_runs if r.role == "hypothesis_agent"]
    assert run.run.outcome == "stopped" and run.run.report_by_cedar
    assert run.run.report.startswith("No report from the agent: it ended stopped (stopped: agent-hyp-a was retired)")
    assert "- its hypothesis is now PLANNING_TEST" in run.run.report
    assert len(router.models["agent-hyp-a"].calls) == 1 and run.run.turns == 1  # its tokens still count


def test_a_dispatcher_that_will_not_stop_is_killed_when_forced(tmp_path):
    import signal
    import subprocess
    import sys

    from cedar.dispatch import dispatcher_running, stop_dispatcher
    from cedar.workspace import Workspace

    workspace = Workspace.open(tmp_path / "stubborn")
    (workspace.root / "activity").mkdir()
    (workspace.root / "activity" / "t-1.json").write_text("{}")
    stubborn = subprocess.Popen(
        [sys.executable, "-c", "import fcntl, os, signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
         "f = open(sys.argv[1] + '/dispatcher.lock', 'a'); fcntl.lockf(f, fcntl.LOCK_EX); "
         "open(sys.argv[1] + '/dispatcher.pid', 'w').write(str(os.getpid())); print('ready', flush=True); "
         "time.sleep(60)", str(workspace.root)], stdout=subprocess.PIPE, text=True)
    try:
        assert stubborn.stdout.readline().strip() == "ready"
        assert stop_dispatcher(workspace, timeout=0.5) == (
            f"Asked the dispatcher (process {stubborn.pid}) to stop; it is still finishing agent runs.")
        assert stop_dispatcher(workspace, timeout=0.5, force=True).startswith(
            f"Stopped the dispatcher (process {stubborn.pid}) by force")
        assert stubborn.wait(10) == -signal.SIGKILL and not dispatcher_running(workspace)
        assert not (workspace.root / "dispatcher.pid").exists() and not list((workspace.root / "activity").iterdir())
    finally:
        stubborn.kill()
