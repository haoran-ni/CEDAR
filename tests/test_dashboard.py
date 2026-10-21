"""What the live panel shows, computed from the record. Synthetic data throughout."""

from datetime import datetime, timedelta, timezone

import pytest
from factories import (
    AGENT,
    CHIEF,
    ENGINE,
    MANAGER,
    MONITOR,
    USER,
    VALIDATOR,
    add_hypothesis,
    authorize,
    draft,
    examine,
    move,
    record_evidence,
    run_to_parsing,
    set_hypothesis_status,
    start_agent,
    start_investigation,
    task,
)

from sim2real.contracts import (
    AgentRetired,
    AgentRunRecorded,
    AgentSpec,
    AgentsProposed,
    AgentTaskQueued,
    CalculationState,
    EvidenceQuality,
    ExaminationVerdict,
    HypothesisStatus,
    PermissionAction,
    QualityCheck,
    SchedulerState,
    UserInterventionRecorded,
)
from sim2real.dashboard import (
    age,
    agent_status,
    dispatcher_status,
    job_status,
    model_line,
    short_id,
    status_line,
    subagent_status,
    swarm_rows,
    work_summary,
)
from sim2real.models import ModelSettings
from sim2real.dispatch import Activity
from sim2real.workspace import Workspace

LATER = datetime.now(timezone.utc) + timedelta(minutes=5)


@pytest.fixture
def workspace(tmp_path):
    return Workspace.open(tmp_path / "data")


@pytest.fixture
def ledger(workspace):
    return workspace.ledger


def test_ages_are_coarse_and_never_negative():
    assert [age(s) for s in (-3, 5, 125, 3700, 7200, 200_000)] == ["0 s", "5 s", "2 min", "1 h 1 min", "2 h", "2 d"]


def test_job_status_follows_the_calculation_and_the_scheduler(ledger):
    inv = start_investigation(ledger)
    hyp = add_hypothesis(ledger, inv)

    def new(**kwargs):
        return ledger.request_calculation(inv, AGENT, hyp, task(**kwargs), "test").calculation_id

    def status(cid):
        text = job_status(ledger.load(inv), ledger.load(inv).calculations[cid])
        assert text.startswith(short_id(cid) + " ")
        return text.removeprefix(short_id(cid) + " ")

    a = new(xc="PBE")
    assert status(a) == "being examined"
    examine(ledger, inv, a)
    move(ledger, inv, a, ENGINE, CalculationState.VALIDATED_INPUT)
    assert status(a) == "needs /approve submit"
    authorize(ledger, inv, PermissionAction.SUBMIT, a, intervention_id="ok-a")
    assert status(a) == "authorized, not yet submitted"
    move(ledger, inv, a, MANAGER, CalculationState.SUBMISSION_PENDING)
    assert status(a) == "submitting"
    move(ledger, inv, a, MANAGER, CalculationState.SUBMITTED, scheduler_job_id="77", authorization_id="ok-a")
    assert status(a) == "job 77 SUBMITTED"
    move(ledger, inv, a, MONITOR, CalculationState.QUEUED)
    assert status(a) == "job 77 PENDING"
    move(ledger, inv, a, MONITOR, CalculationState.RUNNING)
    assert status(a) == "job 77 RUNNING"
    move(ledger, inv, a, MONITOR, CalculationState.SCHEDULER_TERMINAL, scheduler_state=SchedulerState.COMPLETED)
    assert status(a) == "job 77 COMPLETED, run not yet checked"
    move(ledger, inv, a, VALIDATOR, CalculationState.PARSING)
    record_evidence(ledger, inv, a, "ev-a")
    assert status(a) == "job 77 COMPLETED"

    b = new(xc="PBE0")
    examine(ledger, inv, b, ExaminationVerdict.NEEDS_REVIEW)
    assert status(b) == "needs /approve inputs"
    c = new(xc="HSE06")
    examine(ledger, inv, c, ExaminationVerdict.REJECTED)
    move(ledger, inv, c, ENGINE, CalculationState.REJECTED, reason="bad inputs")
    assert status(c) == "inputs rejected"


def test_rows_keep_jobs_evidence_and_assessment_apart(ledger):
    inv = start_investigation(ledger)
    hyp = add_hypothesis(ledger, inv)
    ok = ledger.request_calculation(inv, AGENT, hyp, task(), "test").calculation_id
    run_to_parsing(ledger, inv, ok, job_id="1")
    record_evidence(ledger, inv, ok, "ev-1")
    failed = ledger.request_calculation(inv, AGENT, hyp, task(xc="PBE0"), "test").calculation_id
    run_to_parsing(ledger, inv, failed, job_id="2", scheduler_state=SchedulerState.FAILED)
    record_evidence(ledger, inv, failed, "ev-2", quality=EvidenceQuality.INVALID,
                    checks=(QualityCheck(name="scf_converged", passed=False),))
    state = ledger.load(inv)
    (row,) = swarm_rows(state, [], [], LATER)
    assert row.hypothesis_id == hyp and row.lifecycle == "PLANNING_TEST" and row.priority == "normal"
    assert row.jobs == f"{short_id(ok)} job 1 COMPLETED; {short_id(failed)} job 2 FAILED, run failed its checks"
    assert row.evidence == "1 VALIDATED, 1 INVALID"
    assert row.assessment == "NOT_TESTED" and row.agent == "test specialist: not run yet"
    assert work_summary(state) == "1 hypothesis · jobs: none active · evidence: 1 VALIDATED, 1 INVALID"


def test_agent_status_prefers_what_is_happening_now(workspace):
    ledger = workspace.ledger
    inv = start_investigation(ledger)
    hyp = add_hypothesis(ledger, inv, agent=False)

    def status(*activities):
        state = ledger.load(inv)
        times = {e.stream_version: e.occurred_at for e in workspace.store.read_stream(inv)}
        return agent_status(state, state.hypotheses[hyp], times, activities, LATER)

    assert status() == "no agent"
    ledger.record(inv, [draft(CHIEF, AgentsProposed(proposal_id="p-1", rationale="test", agents=(
        AgentSpec(hypothesis_id=hyp, role="exchange-correlation specialist", brief="compare functionals"),)))])
    assert status().startswith("proposed exchange-correlation specialist; starts in ")
    ledger.record(inv, [draft(USER, UserInterventionRecorded(intervention_id="msg-1", text="Use PBE0 too."))])
    assert status() == "proposed exchange-correlation specialist; held for your reply"

    agent = start_agent(ledger, inv, hyp, role="exchange-correlation specialist")
    assert status() == "exchange-correlation specialist: not run yet"
    ledger.record(inv, [draft(USER, AgentTaskQueued(task_id="t-1", role="hypothesis_agent", hypothesis_id=hyp,
                                                    agent_id=agent.id, instruction="test it"))])
    assert status().startswith("exchange-correlation specialist: queued ") and status().endswith(" ago")
    working = Activity(investigation_id=inv, task_id="t-1", role="hypothesis_agent", hypothesis_id=hyp,
                       agent_id=agent.id, started_at=LATER - timedelta(minutes=3), last_step="request_calculation",
                       last_step_at=LATER)
    assert status(working) == "exchange-correlation specialist: working for 3 min; last step request_calculation"
    elsewhere = Activity(investigation_id="other", task_id="t-9", role="hypothesis_agent", hypothesis_id=hyp,
                         agent_id=agent.id, started_at=LATER)
    assert status(elsewhere).startswith("exchange-correlation specialist: queued")
    judge = Activity(investigation_id=inv, task_id="t-8", role="evidence_judge", hypothesis_id=hyp, agent_id=None,
                     started_at=LATER - timedelta(minutes=1))
    assert status(working, judge) == "judge working for 1 min"

    calc = ledger.request_calculation(inv, agent, hyp, task(), "test").calculation_id
    examine(ledger, inv, calc)
    set_hypothesis_status(ledger, inv, hyp, HypothesisStatus.WAITING_FOR_JOB, awaiting=(calc,))
    ledger.record(inv, [draft(agent, AgentRunRecorded(
        session_id="s", role="hypothesis_agent", task_id="t-1", provider="scripted", requested_model="m", turns=2,
        input_tokens=1, output_tokens=1, cache_read_tokens=0, cache_write_tokens=0, outcome="waiting"))])
    assert status() == "exchange-correlation specialist: waiting for jobs"
    set_hypothesis_status(ledger, inv, hyp, HypothesisStatus.ANALYZING)
    assert status().startswith("exchange-correlation specialist: last ran ") and status().endswith(" ago (waiting)")
    ledger.record(inv, [draft(CHIEF, AgentRetired(agent_id=agent.id, reason="done"))])
    assert status() == "agent retired"


def test_dispatcher_status_and_status_line(ledger):
    now = LATER
    assert dispatcher_status(None, [], now) == ("no dispatcher has run: start `sim2real dispatch`", True)
    assert dispatcher_status(601, [], now)[1] and "silent for 10 min" in dispatcher_status(601, [], now)[0]
    assert dispatcher_status(30, [], now) == ("dispatcher seen 30 s ago", False)

    def running(role, n):
        return Activity(investigation_id="i", task_id=f"t-{n}", role=role, hypothesis_id=f"h{n}", agent_id=None,
                        started_at=now)

    assert dispatcher_status(5, [running("evidence_judge", 1)], now) == ("dispatcher running the Evidence Judge", False)
    assert dispatcher_status(5, [running("hypothesis_agent", 1), running("hypothesis_agent", 2),
                                 running("chief_scientist", 3)], now)[0] == \
        "dispatcher running 2 hypothesis agents and the Chief Scientist's review"

    inv = start_investigation(ledger, model_token_budget=1000)
    hyp = add_hypothesis(ledger, inv)
    calc = ledger.request_calculation(inv, AGENT, hyp, task(), "test").calculation_id
    examine(ledger, inv, calc)
    move(ledger, inv, calc, ENGINE, CalculationState.VALIDATED_INPUT)
    assert status_line(ledger.load(inv), now) == ("inv-1 · human-in-the-loop · INVESTIGATING · setup · plan v1 · 1 "
                                                  "agent · 0 of 1,000 tokens · 1 waiting for you (/pending)")
    other = add_hypothesis(ledger, inv, "hyp-other", agent=False)
    ledger.record(inv, [draft(CHIEF, AgentsProposed(proposal_id="p-9", rationale="test", agents=(
        AgentSpec(hypothesis_id=other, role="coverage specialist", brief="vary coverage"),)))])
    line = status_line(ledger.load(inv), now)
    assert "1 agent proposed: start in " in line and line.endswith("unless you reply (/agents) · 2 waiting for you "
                                                                   "(/pending)")


def test_the_panel_says_which_sub_agents_are_working(ledger):
    assert subagent_status(None, [], LATER) == "No working sub-agents"
    inv = start_investigation(ledger)
    hyp = add_hypothesis(ledger, inv)
    state = ledger.load(inv)
    assert subagent_status(state, [], LATER) == "No working sub-agents"

    def running(role, agent=None, hypothesis=hyp, investigation=inv, minutes=2):
        return Activity(investigation_id=investigation, task_id=f"t-{role}", role=role, hypothesis_id=hypothesis,
                        agent_id=agent, started_at=LATER - timedelta(minutes=minutes))

    review = running("chief_scientist", hypothesis=None, minutes=1)
    assert subagent_status(state, [running("hypothesis_agent", AGENT.id, investigation="other")], LATER) == \
        "No working sub-agents"
    assert subagent_status(state, [review], LATER) == ("No working sub-agents · the Chief Scientist is reviewing "
                                                       "their reports")
    assert subagent_status(state, [running("hypothesis_agent", AGENT.id), running("evidence_judge", minutes=1)],
                           LATER) == f"Working sub-agents: test specialist on {hyp} (2 min); Evidence Judge on {hyp} (1 min)"


def test_the_panel_names_the_models_in_use(ledger):
    opus = ModelSettings(provider="anthropic", model="claude-opus-5-5", effort="high")
    assert model_line(lambda role: opus, None) == "Model: claude-opus-5-5, effort high (anthropic)"

    def missing(role):
        raise KeyError(role)

    assert model_line(missing, None) == "Model: none configured (sim2real models init)"
    mixed = {"hypothesis_agent": ModelSettings(provider="anthropic", model="claude-sonnet-5-5")}
    assert model_line(lambda role: mixed.get(role, opus), None) == (
        "Models: Chief Scientist claude-opus-5-5, effort high (anthropic) · hypothesis agents claude-sonnet-5-5 "
        "(anthropic) · Evidence Judge claude-opus-5-5, effort high (anthropic) · approver claude-opus-5-5, effort "
        "high (anthropic)")

    inv = start_investigation(ledger)
    hyp = add_hypothesis(ledger, inv)
    ledger.record(inv, [draft(AGENT, AgentRunRecorded(
        session_id="s", role="hypothesis_agent", provider="anthropic", requested_model="claude-opus-5-5",
        served_models=("claude-opus-5-5", "claude-sonnet-5-5"), turns=2, input_tokens=1, output_tokens=1,
        cache_read_tokens=0, cache_write_tokens=0, outcome="finished"))])
    assert model_line(lambda role: opus, ledger.load(inv)).endswith(
        "· hypothesis agents: latest run answered by claude-opus-5-5, claude-sonnet-5-5")


def test_the_panel_says_when_queued_work_waits_for_a_dispatcher(ledger):
    inv = start_investigation(ledger)
    hyp = add_hypothesis(ledger, inv)
    ledger.record(inv, [draft(USER, AgentTaskQueued(task_id="t-1", role="hypothesis_agent", hypothesis_id=hyp,
                                                    agent_id=AGENT.id, instruction="test it"))])
    state = ledger.load(inv)
    assert subagent_status(state, [], LATER, dispatcher_running=False) == (
        "No working sub-agents · 1 task queued, waiting for a dispatcher: start `sim2real dispatch`")
    assert subagent_status(state, [], LATER) == "No working sub-agents · 1 task queued"
    working = Activity(investigation_id=inv, task_id="t-1", role="hypothesis_agent", hypothesis_id=hyp,
                       agent_id=AGENT.id, started_at=LATER)
    assert subagent_status(state, [working], LATER) == f"Working sub-agents: test specialist on {hyp} (0 s)"


def test_with_automatic_dispatch_a_missing_dispatcher_needs_no_attention(ledger):
    assert dispatcher_status(None, [], LATER, automatic=True) == (
        "no dispatcher running; one starts by itself when there is work", False)
    assert dispatcher_status(30, [], LATER, automatic=True) == ("dispatcher seen 30 s ago", False)
    inv = start_investigation(ledger)
    hyp = add_hypothesis(ledger, inv)
    ledger.record(inv, [draft(USER, AgentTaskQueued(task_id="t-1", role="hypothesis_agent", hypothesis_id=hyp,
                                                    agent_id=AGENT.id, instruction="test it"))])
    assert subagent_status(ledger.load(inv), [], LATER, dispatcher_running=False, automatic=True) == (
        "No working sub-agents · 1 task queued, waiting for the dispatcher to start")
