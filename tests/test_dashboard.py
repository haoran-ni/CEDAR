"""What the live panel shows, computed from the record. Synthetic data throughout."""

from datetime import datetime, timedelta, timezone

import pytest
from factories import (
    AGENT,
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
    start_investigation,
    task,
)

from sim2real.contracts import (
    AgentRunRecorded,
    AgentTaskQueued,
    CalculationState,
    EvidenceQuality,
    ExaminationVerdict,
    HypothesisStatus,
    PermissionAction,
    QualityCheck,
    SchedulerState,
)
from sim2real.dashboard import (
    age,
    agent_status,
    dispatcher_status,
    job_status,
    short_id,
    status_line,
    swarm_rows,
    work_summary,
)
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
    (row,) = swarm_rows(state, [], None, LATER)
    assert row.hypothesis_id == hyp and row.lifecycle == "PLANNING_TEST" and row.priority == "normal"
    assert row.jobs == f"{short_id(ok)} job 1 COMPLETED; {short_id(failed)} job 2 FAILED, run failed its checks"
    assert row.evidence == "1 VALIDATED, 1 INVALID"
    assert row.assessment == "NOT_TESTED" and row.agent == "no agent yet"
    assert work_summary(state) == "1 hypothesis · jobs: none active · evidence: 1 VALIDATED, 1 INVALID"


def test_agent_status_prefers_what_is_happening_now(workspace):
    ledger = workspace.ledger
    inv = start_investigation(ledger)
    hyp = add_hypothesis(ledger, inv)

    def status(activity=None):
        state = ledger.load(inv)
        times = {e.stream_version: e.occurred_at for e in workspace.store.read_stream(inv)}
        return agent_status(state, state.hypotheses[hyp], times, activity, LATER)

    assert status() == "no agent yet"
    ledger.record(inv, [draft(USER, AgentTaskQueued(task_id="t-1", role="hypothesis_agent", hypothesis_id=hyp,
                                                    instruction="test it"))])
    assert status().startswith("agent queued ") and status().endswith(" ago")
    working = Activity(investigation_id=inv, task_id="t-1", role="hypothesis_agent", hypothesis_id=hyp,
                       started_at=LATER - timedelta(minutes=3), last_step="request_calculation",
                       last_step_at=LATER)
    assert status(working) == "agent working for 3 min; last step request_calculation"
    elsewhere = Activity(investigation_id="other", task_id="t-9", role="hypothesis_agent", hypothesis_id=hyp,
                         started_at=LATER)
    assert status(elsewhere).startswith("agent queued")

    calc = ledger.request_calculation(inv, AGENT, hyp, task(), "test").calculation_id
    examine(ledger, inv, calc)
    set_hypothesis_status(ledger, inv, hyp, HypothesisStatus.WAITING_FOR_JOB, awaiting=(calc,))
    ledger.record(inv, [draft(AGENT, AgentRunRecorded(
        session_id="s", role="hypothesis_agent", task_id="t-1", provider="scripted", requested_model="m", turns=2,
        input_tokens=1, output_tokens=1, cache_read_tokens=0, cache_write_tokens=0, outcome="waiting"))])
    assert status() == "agent waiting for jobs"
    set_hypothesis_status(ledger, inv, hyp, HypothesisStatus.ANALYZING)
    assert status().startswith("agent last ran ") and status().endswith(" ago (waiting)")


def test_dispatcher_status_and_status_line(ledger):
    now = LATER
    assert dispatcher_status(None, None, now) == ("no dispatcher has run: start `sim2real dispatch`", True)
    assert dispatcher_status(601, None, now)[1] and "silent for 10 min" in dispatcher_status(601, None, now)[0]
    assert dispatcher_status(30, None, now) == ("dispatcher seen 30 s ago", False)
    judge = Activity(investigation_id="i", task_id="t", role="evidence_judge", hypothesis_id="h", started_at=now)
    assert dispatcher_status(5, judge, now) == ("dispatcher running the Evidence Judge", False)

    inv = start_investigation(ledger, model_token_budget=1000)
    hyp = add_hypothesis(ledger, inv)
    calc = ledger.request_calculation(inv, AGENT, hyp, task(), "test").calculation_id
    examine(ledger, inv, calc)
    move(ledger, inv, calc, ENGINE, CalculationState.VALIDATED_INPUT)
    assert status_line(ledger.load(inv)) == "inv-1 · INVESTIGATING · setup · plan v1 · 0 of 1,000 tokens · " \
                                            "1 waiting for you (/pending)"
