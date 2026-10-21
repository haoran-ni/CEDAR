"""The hypothesis-agent swarm: agents the Chief Scientist proposes and the researcher reviews, one agent
per hypothesis, agents that cannot reach each other's work, agents running at the same time, and the
Chief Scientist's reviews of what they report.

ScriptedModel and FakeScheduler TEST DOUBLES on a SYNTHETIC site; no model or job runs.
"""

import threading
from collections import Counter

import pytest
from factories import (
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

from sim2real.agents import RunLimits
from sim2real.agents.toolkits import ResearchContext, file_tools, research_tools
from sim2real.agents.toolkits.chief import ChiefContext, chief_tools
from sim2real.agents.toolkits.research import agent_file_access
from sim2real.chat import ChatSession
from sim2real.contracts import (
    Actor,
    ActorKind,
    AgentRetired,
    AgentRunRecorded,
    AgentSpec,
    AgentsProposed,
    AgentTaskQueued,
    AnalysisRecorded,
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
from sim2real.contracts.hypothesis import Prediction
from sim2real.dispatch import SYSTEM, Dispatcher
from sim2real.errors import RuleViolation
from sim2real.models import TextPart
from sim2real.models.scripted import ScriptedModel, reply
from sim2real.notebook import render_notebook, round_file
from sim2real.state.projection import ProposalStatus
from sim2real.summaries import calculation_summary
from sim2real.swarm import CHIEF, review_due, start_drafts

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

    def complete(self, *, system, messages, tools, settings):
        agent = next(a for a in self.models if f"you are {a}," in system)
        with self.lock:
            self.busy[agent] += 1
            self.peak[agent] = max(self.peak[agent], self.busy[agent])
            self.peak_total = max(self.peak_total, sum(self.busy.values()))
            self.order.append(agent)
        try:
            return self.models[agent].complete(system=system, messages=messages, tools=tools, settings=settings)
        finally:
            with self.lock:
                self.busy[agent] -= 1


def investigation(system, *hypotheses, budget=None) -> ChatSession:
    """An investigation on the system's site with activated hypotheses and no agents yet; the researcher's chat."""
    ledger = system["workspace"].ledger
    start_investigation(ledger, INV, compute_profile=system["site"].name, model_token_budget=budget)
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
    assert ("| agent-hyp-a | [hyp-a](hypotheses/hyp-a.md) | hyp-a specialist | review window passed | active | 1 |"
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
    propose(ledger, "p-1", "hyp-a", "hyp-b", token_budget=5000)
    agents = chat.handle("/agents")
    assert "Proposed by the Chief Scientist (p-1), starts at" in agents and "unless the researcher replies" in agents
    assert "  - hyp-a: hyp-a specialist (5,000 tokens)\n    first task: test hyp-a" in agents
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
    assert "Agents:\n  - agent-hyp-a (hyp-a specialist) on hyp-a; 0 tokens used of 5,000" in chat.handle("/agents")


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


# -- agents cannot reach each other's work --------------------------------------------------------

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
    with pytest.raises(RuleViolation, match="agent-hyp-xc reads only calculations linked to hyp-xc"):
        ledger.record(world.inv, [draft(validator, EvidenceRecorded(evidence_id="ev-x", evidence=computed_evidence(
            b_calc).model_copy(update={"interpreted_by": agent_a})))])
    ledger.record(world.inv, [draft(Actor(kind=ActorKind.ANALYSIS_RUNNER, id="analysis-runner"), AnalysisRecorded(
        analysis_id="an-b", hypothesis_id=b, run=analysis_run(b_calc, files=dict(world.calc(b_calc).run_check.outputs))))])
    with pytest.raises(RuleViolation, match="analysis an-b was not run for hyp-xc"):
        ledger.record(world.inv, [draft(validator, EvidenceRecorded(evidence_id="ev-y", evidence=analysis_evidence(
            "an-b", b_calc).model_copy(update={"interpreted_by": agent_a})))])
    with pytest.raises(RuleViolation, match="an analysis uses only its own hypothesis's runs"):
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
    access = agent_file_access(world.ledger, world.inv, hypothesis, world.runs, deny=[tmp_path / "config"])
    ctx = ResearchContext(ledger=world.ledger, investigation_id=world.inv, hypothesis_id=hypothesis, actor=actor,
                          registry=world.registry, artifacts=world.artifacts, engine=world.engine,
                          validator=world.validator, files=access, analysis=world.analysis)
    return {tool.name: tool for tool in research_tools(ctx) + file_tools(access)}


def reading(calculation_id, output):
    lines = output.splitlines()
    line = next(i for i, text in enumerate(lines, start=1) if "Total energy of the DFT" in text)
    return {"calculation_id": calculation_id, "observable_name": "total energy",
            "observable_definition": "final total energy printed by FHI-aims", "value": ENERGY, "unit": "eV",
            "summary": "baseline", "citations": [{"file": "aims.out", "line_start": line, "line_end": line,
                                                  "quote": lines[line - 1].strip()}]}


def test_agents_see_only_their_own_work(ledger, store, artifacts, site, tmp_path):
    world = World(ledger, store, artifacts, site)
    a, b = world.hyp, add_hypothesis(ledger, world.inv, "hyp-b")
    tools_a = tools_for(world, a, agent_for(a), tmp_path)
    tools_b = tools_for(world, b, agent_for(b), tmp_path)
    shared = ledger.request_calculation(world.inv, agent_for(b), b, world.task(), "b's baseline").calculation_id
    output = aims_output(energy=ENERGY)
    world.checked(shared, output=output)
    aims_out = str(world.runs / world.calc(shared).workdir / "aims.out")

    # The other hypothesis's calculation is out of reach in every way.
    for name, arguments in (("run_outputs", {"calculation_id": shared}), ("submit_reading", reading(shared, output)),
                            ("run_analysis", {"purpose": "peek", "script": "print(1)",
                                              "inputs": [{"calculation_id": shared}]})):
        outcome = tools_a[name].invoke(arguments)
        assert outcome.is_error and "is not one of your hypothesis's calculations" in outcome.content
    outcome = tools_a["read_file"].invoke({"path": aims_out})
    assert outcome.is_error and "outside the readable roots" in outcome.content

    # Its own agent reads it, computes from it and records both.
    assert not tools_b["read_file"].invoke({"path": aims_out}).is_error
    assert "Reading recorded as evidence" in tools_b["submit_reading"].invoke(reading(shared, output)).content
    assert not tools_b["run_analysis"].invoke({"purpose": "energy in Hartree", "script": HARTREE_SCRIPT.format(
        calc=shared), "inputs": [{"calculation_id": shared}]}).is_error
    analysis = next(iter(world.state().analyses))
    outcome = tools_a["record_result"].invoke({"analysis_id": analysis, "result_name": "total_energy_ha",
                                               "observable_name": "e", "observable_definition": "e", "summary": "s"})
    assert outcome.is_error and f"{analysis} is not one of your analyses" in outcome.content

    # Asking for the identical calculation shares the run, but not what the other agent made of it.
    assert ledger.request_calculation(world.inv, agent_for(a), a, world.task(), "reuse").reused
    assert not tools_a["read_file"].invoke({"path": aims_out}).is_error
    state = world.state()
    (b_reading,) = state.calculations[shared].evidence_ids
    overview = tools_a["investigation_overview"].invoke({}).content
    assert shared in overview and b_reading not in overview and "hyp-b" not in overview
    assert b_reading in tools_b["investigation_overview"].invoke({}).content
    assert b_reading in calculation_summary(state, state.calculations[shared])  # the Chief and the Judge see all
    assert "Reading recorded as evidence" in tools_a["submit_reading"].invoke(reading(shared, output)).content
    assert len(world.state().calculations[shared].evidence_ids) == 2


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
    sessions = {r.session_id for r in ledger.load(INV).agent_runs}
    assert sessions == {f"{INV}.agent-hyp-a", f"{INV}.agent-hyp-b"}


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


def test_running_agents_set_their_tokens_aside(system):
    ledger, clients = system["workspace"].ledger, system["clients"]
    chat = investigation(system, "hyp-a", "hyp-b", budget=100)
    dispatcher = Dispatcher(system["workspace"], system["profiles"], clients, backend=system["scheduler"],
                            clock=system["clock"], chief_reviews=False,
                            limits=RunLimits(max_turns=5, max_tokens=100))
    propose(ledger, "p-1", "hyp-a", "hyp-b")
    chat.handle("/approve agents")
    agents_answer(system, **{"hyp-a": [says("A.")], "hyp-b": [says("B.")]})
    dispatcher.tick()  # hyp-a's run sets aside all 100 tokens, so hyp-b waits
    assert [r.actor_id for r in ledger.load(INV).agent_runs] == ["agent-hyp-a"]
    dispatcher.tick()  # hyp-a used 15; hyp-b may use the remaining 85
    assert [r.actor_id for r in ledger.load(INV).agent_runs] == ["agent-hyp-a", "agent-hyp-b"]


def test_an_agent_stops_at_its_own_budget(system):
    ledger, dispatcher = system["workspace"].ledger, system["dispatcher"]
    chat = investigation(system, "hyp-a")
    propose(ledger, "p-1", "hyp-a", token_budget=10)
    chat.handle("/approve agents")
    agents_answer(system, **{"hyp-a": [says("A.")]})
    dispatcher.tick()
    ledger.record(INV, [draft(CHIEF, AgentTaskQueued(task_id="t-more", role="hypothesis_agent", hypothesis_id="hyp-a",
                                                     agent_id="agent-hyp-a", instruction="more"))])
    assert f"{INV}: agent-hyp-a has used its budget of 10 tokens; its queued work waits" in dispatcher.tick()
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
    assert review.session_id == f"{INV}.chief-review" and review.run.task_id is not None
    instruction = clients.models["chief_scientist"].calls[0].messages[-1].text
    assert "[agent-hyp-a (hyp-a specialist) on hyp-a; run finished]\nA found the hollow site" in instruction
    assert "[agent-hyp-b (hyp-b specialist) on hyp-b; run finished]\nB found no coverage effect" in instruction
    assert "background review" in instruction and "Agents never see each other's work" in instruction
    assert not any("review" in n for n in dispatcher.tick())  # nothing new came in
    assert not review_due(ledger.load(INV))

    # The researcher sees the review in the chat, and the Chief knows it there.
    (shown,) = [n for n in chat.notices() if n.review]
    assert shown.text == "Both agents reported. Their results do not conflict; I asked for nothing new."
    clients.add("chief_scientist", says("Nothing conflicts so far."))
    assert chat.handle("What came in?") == "Nothing conflicts so far."
    asked = clients.models["chief_scientist"].calls[-1].messages[-1].text
    assert asked.startswith("[record: your background review of the agents' reports, run finished, concluded]\n"
                            "Both agents reported.")
    assert asked.endswith("[researcher, recorded as msg-2] What came in?")
    assert chat.chief_session == f"{INV}.chief"
    round_page = render_notebook(system["workspace"].store.read_stream(INV))[round_file(ledger.load(INV).round)]
    assert "(chief scientist; review of the agents' reports; task" in round_page


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
                           "brief": "Compare PBE and PBE0 site energies.", "token_budget": 50_000}])
    assert not outcome.is_error and "10 minutes from now) unless the researcher replies" in outcome.content
    outcome = use("delegate", hypothesis_id="hyp-a", instruction="go")
    assert outcome.is_error and "proposed but has not started" in outcome.content
    listing = use("agents").content
    assert "  - hyp-a: exchange-correlation specialist (50,000 tokens)" in listing
    assert listing.endswith("Open hypotheses without an agent: hyp-b")
    first = ledger.load(INV).pending_proposal.proposal_id
    outcome = use("propose_agents", rationale="both after all", agents=[
        {"hypothesis_id": "hyp-a", "role": "exchange-correlation specialist", "brief": "Compare functionals."},
        {"hypothesis_id": "hyp-b", "role": "coverage specialist", "brief": "Vary the coverage."}])
    assert f"It replaces {first}, which never started." in outcome.content

    chat.handle("/approve agents")
    assert not use("delegate", hypothesis_id="hyp-a", instruction="Now relax the slab.").is_error
    assert "Retired agent-hyp-a (exchange-correlation specialist)" in use(
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
