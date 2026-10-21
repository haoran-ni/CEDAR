import pytest

from factories import (
    AGENT,
    CHIEF,
    ENGINE,
    JUDGE,
    MANAGER,
    MONITOR,
    PREFERRED_SITE,
    SITE_ENERGY,
    USER,
    VALIDATOR,
    add_hypothesis,
    authorize,
    computed_evidence,
    draft,
    examine,
    examination,
    hypothesis_spec,
    investigation_spec,
    move,
    plan,
    RAW_OUTPUT,
    analysis_evidence,
    analysis_run,
    check_run,
    record_analysis,
    record_evidence,
    run_to_parsing,
    set_hypothesis_status,
    start_investigation,
    submit,
    task,
)
from sim2real.contracts import (
    AnalysisOutcome,
    AnalysisRecorded,
    Assessment,
    AssessmentRecorded,
    CalculationRequested,
    CalculationState,
    CalculationStateChanged,
    CalculationInputsExamined,
    Completeness,
    EvidenceQuality,
    ExaminationVerdict,
    EvidenceRecorded,
    HypothesisProposed,
    HypothesisStatus,
    InvestigationStatus,
    InvestigationStatusChanged,
    PermissionAction,
    PlanRevised,
    Prediction,
    PredictionRegistered,
    Priority,
    QualityCheck,
    RoundStarted,
    SchedulerState,
    UserInterventionRecorded,
)
from sim2real.errors import InvalidTransition, RuleViolation
from sim2real.state import ResearchLedger, replay
from sim2real.state.projection import hypothesis_evidence
from sim2real.store import IdempotencyConflict, SQLiteEventStore


def predict(ledger, inv, hyp, prediction_id="pred-1", calculation_ids=()):
    return ledger.record(inv, [draft(AGENT, PredictionRegistered(
        prediction_id=prediction_id, hypothesis_id=hyp,
        prediction=Prediction(
            statement="Correcting CO frontier orbitals favours the atop site.",
            observable=SITE_ENERGY,
            expected_outcome="E(atop) - E(fcc) becomes negative",
            falsified_if="E(atop) - E(fcc) stays positive",
            calculation_ids=calculation_ids,
        ),
    ))])


def assess(ledger, inv, hyp, assessment, *evidence_ids, completeness=Completeness.FINAL):
    return ledger.record(inv, [draft(JUDGE, AssessmentRecorded(
        hypothesis_id=hyp, assessment=assessment, completeness=completeness,
        evidence_ids=evidence_ids, rationale="test assessment",
    ))])


@pytest.fixture
def inv(ledger):
    return start_investigation(ledger)


@pytest.fixture
def hyp(ledger, inv):
    return add_hypothesis(ledger, inv)


def request(ledger, inv, hyp, **task_kwargs):
    return ledger.request_calculation(inv, AGENT, hyp, task(**task_kwargs), "discriminating test")


def test_full_investigation_cycle_replays_to_identical_state(ledger, store, tmp_path, inv, hyp):
    calc = request(ledger, inv, hyp, xc="PBE0").calculation_id
    predict(ledger, inv, hyp, calculation_ids=(calc,))
    set_hypothesis_status(ledger, inv, hyp, HypothesisStatus.WAITING_FOR_JOB, awaiting=(calc,))
    run_to_parsing(ledger, inv, calc)
    record_evidence(ledger, inv, calc)
    set_hypothesis_status(ledger, inv, hyp, HypothesisStatus.ANALYZING, HypothesisStatus.ASSESSED)
    result = assess(ledger, inv, hyp, Assessment.SUPPORTED, "ev-1")

    state = result.state
    h = state.hypotheses[hyp]
    c = state.calculations[calc]
    assert (h.status, h.assessment, h.awaiting_calculation_ids) == (
        HypothesisStatus.ASSESSED, Assessment.SUPPORTED, ())
    assert (c.state, c.scheduler_state, c.scheduler_job_id, c.evidence_ids) == (
        CalculationState.QUALITY_ASSESSED, SchedulerState.COMPLETED, "1001", ["ev-1"])
    assert c.run_check is not None and c.run_check.healthy
    assert [s for _, s in c.history] == [
        CalculationState.SPECIFIED, CalculationState.VALIDATED_INPUT, CalculationState.SUBMISSION_PENDING,
        CalculationState.SUBMITTED, CalculationState.RUNNING, CalculationState.SCHEDULER_TERMINAL,
        CalculationState.PARSING, CalculationState.QUALITY_ASSESSED,
    ]
    assert replay(store.read_stream(inv)) == state
    assert ResearchLedger(SQLiteEventStore(tmp_path / "events.sqlite")).load(inv) == state


def test_rejected_batch_records_nothing(ledger, inv, hyp):
    calc = request(ledger, inv, hyp).calculation_id
    before = ledger.load(inv).version
    with pytest.raises(RuleViolation):
        ledger.record(inv, [
            draft(ENGINE, CalculationStateChanged(calculation_id=calc, state=CalculationState.VALIDATED_INPUT)),
            draft(MONITOR, CalculationStateChanged(calculation_id=calc, state=CalculationState.RUNNING)),
        ])
    assert ledger.load(inv).version == before


class TestIdempotency:
    def test_replayed_scheduler_update_is_recorded_once(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        submit(ledger, inv, calc)
        update = draft(MONITOR, CalculationStateChanged(calculation_id=calc, state=CalculationState.RUNNING),
                       idempotency_key="slurm:1001:RUNNING")
        first = ledger.record(inv, [update])
        second = ledger.record(inv, [update])
        assert not first.deduplicated and second.deduplicated
        assert second.events == first.events
        assert ledger.load(inv).version == first.state.version

    def test_reused_key_with_different_content_is_refused(self, ledger, inv):
        ledger.record(inv, [draft(USER, UserInterventionRecorded(intervention_id="i1", text="a"),
                                  idempotency_key="k")])
        with pytest.raises(IdempotencyConflict):
            ledger.record(inv, [draft(USER, UserInterventionRecorded(intervention_id="i1", text="b"),
                                      idempotency_key="k")])

    def test_creating_an_investigation_twice_is_safe(self, ledger, inv):
        again = ledger.create_investigation(USER, investigation_spec(), plan(), investigation_id=inv)
        assert again.deduplicated and again.state.status is InvestigationStatus.INVESTIGATING


class TestDuplicateCalculations:
    def test_identical_task_is_shared_between_hypotheses(self, ledger, inv, hyp):
        other = add_hypothesis(ledger, inv, "hyp-coverage")
        first = request(ledger, inv, hyp)
        shared = request(ledger, inv, other)
        assert not first.reused and shared.reused
        assert shared.calculation_id == first.calculation_id
        assert shared.result.events[0].type == "calculation_linked"
        state = shared.result.state
        assert state.calculations[first.calculation_id].hypothesis_ids == [hyp, other]
        assert len(state.calculations) == 1

    def test_repeating_a_request_records_nothing(self, ledger, inv, hyp):
        first = request(ledger, inv, hyp)
        again = request(ledger, inv, hyp)
        assert again.reused and again.result.deduplicated and again.result.events == ()
        assert again.result.state.version == first.result.state.version

    def test_direct_duplicate_request_is_a_rule_violation(self, ledger, inv, hyp):
        request(ledger, inv, hyp)
        spec = task()
        with pytest.raises(RuleViolation, match="link it instead"):
            ledger.record(inv, [draft(AGENT, CalculationRequested(
                calculation_id="calc-dup", hypothesis_id=hyp, task=spec,
                task_hash=spec.identity_hash(), rationale="r"))])

    def test_failed_scheduler_run_can_be_attempted_again(self, ledger, inv, hyp):
        failed = request(ledger, inv, hyp).calculation_id
        submit(ledger, inv, failed)
        move(ledger, inv, failed, MONITOR, CalculationState.SCHEDULER_TERMINAL,
             scheduler_state=SchedulerState.NODE_FAIL)
        retry = request(ledger, inv, hyp)
        assert not retry.reused and retry.calculation_id != failed


class TestPreregistration:
    def test_prediction_after_the_job_ended_is_refused(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        submit(ledger, inv, calc)
        move(ledger, inv, calc, MONITOR, CalculationState.SCHEDULER_TERMINAL,
             scheduler_state=SchedulerState.COMPLETED)
        with pytest.raises(RuleViolation, match="too late"):
            predict(ledger, inv, hyp, calculation_ids=(calc,))

    def test_evidence_seen_before_any_prediction_cannot_support(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        run_to_parsing(ledger, inv, calc)
        record_evidence(ledger, inv, calc)
        predict(ledger, inv, hyp)
        with pytest.raises(RuleViolation, match="before any prediction"):
            assess(ledger, inv, hyp, Assessment.SUPPORTED, "ev-1")
        assert assess(ledger, inv, hyp, Assessment.INCONCLUSIVE, "ev-1").state.hypotheses[hyp].assessment \
            is Assessment.INCONCLUSIVE

    def test_predictions_are_immutable(self, ledger, inv, hyp):
        predict(ledger, inv, hyp)
        with pytest.raises(RuleViolation, match="immutable"):
            predict(ledger, inv, hyp)


class TestSchedulerIsNotValidation:
    def test_unknown_scheduler_state_is_not_terminal(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        submit(ledger, inv, calc)
        with pytest.raises(RuleViolation, match="terminal scheduler state"):
            move(ledger, inv, calc, MONITOR, CalculationState.SCHEDULER_TERMINAL,
                 scheduler_state=SchedulerState.UNKNOWN)
        with pytest.raises(RuleViolation, match="implies"):
            move(ledger, inv, calc, MONITOR, CalculationState.RUNNING, scheduler_state=SchedulerState.COMPLETED)

    def test_output_of_a_failed_job_cannot_be_validated(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        run_to_parsing(ledger, inv, calc, scheduler_state=SchedulerState.TIMEOUT)
        with pytest.raises(RuleViolation, match="TIMEOUT"):
            record_evidence(ledger, inv, calc)
        state = record_evidence(ledger, inv, calc, quality=EvidenceQuality.INVALID,
                                checks=(QualityCheck(name="scf_converged", passed=False),)).state
        assert state.calculations[calc].state is CalculationState.QUALITY_ASSESSED

    def test_quality_assessment_needs_a_run_check(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        run_to_parsing(ledger, inv, calc)
        with pytest.raises(RuleViolation, match="run check to be recorded first"):
            move(ledger, inv, calc, VALIDATOR, CalculationState.QUALITY_ASSESSED)

    def test_a_run_is_checked_once_after_it_ended(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        submit(ledger, inv, calc)
        with pytest.raises(RuleViolation, match="checked while PARSING"):
            check_run(ledger, inv, calc)
        move(ledger, inv, calc, MONITOR, CalculationState.RUNNING)
        move(ledger, inv, calc, MONITOR, CalculationState.SCHEDULER_TERMINAL, scheduler_state=SchedulerState.COMPLETED)
        move(ledger, inv, calc, VALIDATOR, CalculationState.PARSING)
        check_run(ledger, inv, calc)
        with pytest.raises(RuleViolation, match="checked while PARSING|already checked"):
            check_run(ledger, inv, calc)

    def test_a_run_gives_several_readings_but_an_unhealthy_one_validates_none(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        run_to_parsing(ledger, inv, calc)
        record_evidence(ledger, inv, calc, "ev-energy")
        state = record_evidence(ledger, inv, calc, "ev-site", observable=PREFERRED_SITE).state
        assert state.calculations[calc].evidence_ids == ["ev-energy", "ev-site"]

        other = request(ledger, inv, hyp, xc="PBE0").calculation_id
        run_to_parsing(ledger, inv, other, job_id="1002")
        check_run(ledger, inv, other, healthy=False)
        with pytest.raises(RuleViolation, match="failed checks"):
            record_evidence(ledger, inv, other, "ev-bad")
        record_evidence(ledger, inv, other, "ev-bad", quality=EvidenceQuality.INVALID,
                        checks=(QualityCheck(name="scf_converged", passed=False),))


class TestJudgement:
    @pytest.fixture
    def provisional(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        predict(ledger, inv, hyp, calculation_ids=(calc,))
        run_to_parsing(ledger, inv, calc)
        record_evidence(ledger, inv, calc, quality=EvidenceQuality.PROVISIONAL)
        return "ev-1"

    def test_final_verdict_needs_validated_evidence(self, ledger, inv, hyp, provisional):
        with pytest.raises(RuleViolation, match="FINAL"):
            assess(ledger, inv, hyp, Assessment.SUPPORTED, provisional)
        state = assess(ledger, inv, hyp, Assessment.SUPPORTED, provisional,
                       completeness=Completeness.PARTIAL).state
        entry = state.hypotheses[hyp].assessments[-1]
        assert (entry.assessment, entry.completeness) == (Assessment.SUPPORTED, Completeness.PARTIAL)

    def test_support_must_cite_evidence(self, ledger, inv, hyp):
        with pytest.raises(RuleViolation, match="must cite"):
            assess(ledger, inv, hyp, Assessment.CHALLENGED)

    def test_invalid_evidence_never_supports(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        predict(ledger, inv, hyp, calculation_ids=(calc,))
        run_to_parsing(ledger, inv, calc)
        record_evidence(ledger, inv, calc, quality=EvidenceQuality.INVALID, checks=())
        with pytest.raises(RuleViolation, match="INVALID"):
            assess(ledger, inv, hyp, Assessment.SUPPORTED, "ev-1", completeness=Completeness.PARTIAL)

    def test_assessment_history_is_kept(self, ledger, inv, hyp, provisional):
        assess(ledger, inv, hyp, Assessment.INCONCLUSIVE, provisional, completeness=Completeness.PARTIAL)
        state = assess(ledger, inv, hyp, Assessment.SUPPORTED, provisional,
                       completeness=Completeness.PARTIAL).state
        assert [a.assessment for a in state.hypotheses[hyp].assessments] == [
            Assessment.INCONCLUSIVE, Assessment.SUPPORTED]


class TestAuthority:
    def test_chief_scientist_cannot_execute_validate_or_judge(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        with pytest.raises(RuleViolation, match="chief_scientist may not"):
            move(ledger, inv, calc, CHIEF, CalculationState.VALIDATED_INPUT)
        with pytest.raises(RuleViolation, match="chief_scientist may not"):
            ledger.record(inv, [draft(CHIEF, EvidenceRecorded(evidence_id="e", evidence=computed_evidence(calc)))])
        with pytest.raises(RuleViolation, match="chief_scientist may not"):
            ledger.record(inv, [draft(CHIEF, AssessmentRecorded(
                hypothesis_id=hyp, assessment=Assessment.INCONCLUSIVE,
                completeness=Completeness.PARTIAL, rationale="r"))])

    def test_agents_cannot_record_computed_evidence_or_move_jobs(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        run_to_parsing(ledger, inv, calc)
        with pytest.raises(RuleViolation, match="direct_computation"):
            ledger.record(inv, [draft(AGENT, EvidenceRecorded(evidence_id="e", evidence=computed_evidence(calc)))])
        with pytest.raises(RuleViolation, match="hypothesis_agent may not"):
            move(ledger, inv, calc, AGENT, CalculationState.QUALITY_ASSESSED)

    def test_each_service_moves_only_its_own_states(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        with pytest.raises(RuleViolation, match="job_monitor may not move a calculation to VALIDATED_INPUT"):
            move(ledger, inv, calc, MONITOR, CalculationState.VALIDATED_INPUT)

    def test_only_users_record_interventions(self, ledger, inv):
        with pytest.raises(RuleViolation):
            ledger.record(inv, [draft(CHIEF, UserInterventionRecorded(intervention_id="i", text="t"))])


class TestUserAuthorization:
    def test_submission_requires_approval_by_default(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        examine(ledger, inv, calc)
        move(ledger, inv, calc, ENGINE, CalculationState.VALIDATED_INPUT)
        move(ledger, inv, calc, MANAGER, CalculationState.SUBMISSION_PENDING)
        with pytest.raises(RuleViolation, match="submit"):
            move(ledger, inv, calc, MANAGER, CalculationState.SUBMITTED, scheduler_job_id="1001")

    def test_submission_approval_can_be_waived(self, ledger):
        inv = start_investigation(ledger, "inv-open", submit_requires_approval=False)
        hyp = add_hypothesis(ledger, inv)
        calc = request(ledger, inv, hyp).calculation_id
        examine(ledger, inv, calc)
        move(ledger, inv, calc, ENGINE, CalculationState.VALIDATED_INPUT)
        move(ledger, inv, calc, MANAGER, CalculationState.SUBMISSION_PENDING)
        move(ledger, inv, calc, MANAGER, CalculationState.SUBMITTED, scheduler_job_id="1001")

    def test_scheduler_job_ids_are_unique(self, ledger, inv, hyp):
        first = request(ledger, inv, hyp).calculation_id
        second = request(ledger, inv, hyp, xc="PBE0").calculation_id
        submit(ledger, inv, first, job_id="1001")
        with pytest.raises(RuleViolation, match="already attached"):
            submit(ledger, inv, second, job_id="1001")

    def test_cancelling_a_submitted_job_needs_a_matching_authorization(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        other = request(ledger, inv, hyp, xc="PBE0").calculation_id
        submit(ledger, inv, calc)
        with pytest.raises(RuleViolation, match="cancel"):
            move(ledger, inv, calc, MANAGER, CalculationState.CANCELLED, reason="deprioritized")
        wrong = authorize(ledger, inv, PermissionAction.CANCEL, other, intervention_id="cancel-other")
        with pytest.raises(RuleViolation, match="cancel"):
            move(ledger, inv, calc, MANAGER, CalculationState.CANCELLED, reason="r", authorization_id=wrong)
        right = authorize(ledger, inv, PermissionAction.CANCEL, calc, intervention_id="cancel-calc")
        state = move(ledger, inv, calc, MANAGER, CalculationState.CANCELLED, reason="user asked",
                     authorization_id=right).state
        assert state.calculations[calc].state is CalculationState.CANCELLED

    def test_unsubmitted_work_can_be_cancelled_with_a_reason(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        with pytest.raises(RuleViolation, match="reason"):
            move(ledger, inv, calc, MANAGER, CalculationState.CANCELLED)
        move(ledger, inv, calc, MANAGER, CalculationState.CANCELLED, reason="superseded by plan v2")


class TestPlanChanges:
    def test_hold_blocks_new_work_but_running_jobs_continue(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        submit(ledger, inv, calc)
        move(ledger, inv, calc, MONITOR, CalculationState.RUNNING)
        ledger.record(inv, [draft(USER, UserInterventionRecorded(
            intervention_id="hold-xc", text="Pause new XC tests, but let running jobs finish."))])
        ledger.record(inv, [draft(CHIEF, PlanRevised(
            version=2, plan=plan(**{hyp: Priority.ON_HOLD}), reason="user paused XC tests",
            intervention_id="hold-xc"))])

        with pytest.raises(RuleViolation, match="on hold"):
            request(ledger, inv, hyp, xc="PBE0")
        state = move(ledger, inv, calc, MONITOR, CalculationState.SCHEDULER_TERMINAL,
                     scheduler_state=SchedulerState.COMPLETED).state
        assert state.calculations[calc].scheduler_state is SchedulerState.COMPLETED
        assert [(p.version, p.intervention_id) for p in state.plans] == [(1, None), (2, "hold-xc")]

    def test_plan_versions_are_sequential_and_refer_to_known_hypotheses(self, ledger, inv, hyp):
        with pytest.raises(RuleViolation, match="must be 2"):
            ledger.record(inv, [draft(CHIEF, PlanRevised(version=3, plan=plan(), reason="r"))])
        with pytest.raises(RuleViolation, match="unknown hypotheses"):
            ledger.record(inv, [draft(CHIEF, PlanRevised(
                version=2, plan=plan(**{"hyp-missing": Priority.HIGH}), reason="r"))])

    def test_workers_may_act_under_an_older_plan_but_not_a_future_one(self, ledger, inv, hyp):
        ledger.record(inv, [draft(CHIEF, PlanRevised(version=2, plan=plan(), reason="r"))])
        old = ledger.record(inv, [draft(AGENT, PredictionRegistered(
            prediction_id="p-old", hypothesis_id=hyp,
            prediction=Prediction(statement="s", observable=SITE_ENERGY, expected_outcome="o", falsified_if="f"),
        ), plan_version=1)])
        assert old.events[0].plan_version == 1
        with pytest.raises(RuleViolation, match="does not exist yet"):
            ledger.record(inv, [draft(CHIEF, InvestigationStatusChanged(
                status=InvestigationStatus.REVIEWING, reason="r"), plan_version=3)])


class TestLifecycleRules:
    def test_calculations_only_while_investigating(self, ledger):
        inv = "inv-early"
        ledger.create_investigation(USER, investigation_spec(), plan(), investigation_id=inv)
        ledger.record(inv, [draft(CHIEF, HypothesisProposed(hypothesis_id="h", spec=hypothesis_spec()))])
        set_hypothesis_status(ledger, inv, "h", HypothesisStatus.SCREENED, HypothesisStatus.ACTIVE,
                              HypothesisStatus.PLANNING_TEST)
        with pytest.raises(RuleViolation, match="INVESTIGATING"):
            request(ledger, inv, "h")

    def test_tests_are_requested_while_planning_a_test(self, ledger, inv):
        ledger.record(inv, [draft(CHIEF, HypothesisProposed(hypothesis_id="h", spec=hypothesis_spec()))])
        with pytest.raises(RuleViolation, match="PLANNING_TEST"):
            request(ledger, inv, "h")

    def test_waiting_requires_a_linked_unfinished_calculation(self, ledger, inv, hyp):
        other = add_hypothesis(ledger, inv, "hyp-other")
        calc = request(ledger, inv, other, xc="PBE0").calculation_id
        with pytest.raises(RuleViolation, match="not linked"):
            set_hypothesis_status(ledger, inv, hyp, HypothesisStatus.WAITING_FOR_JOB, awaiting=(calc,))

    def test_paused_agent_can_resume_after_its_job_finished(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        set_hypothesis_status(ledger, inv, hyp, HypothesisStatus.WAITING_FOR_JOB, awaiting=(calc,))
        set_hypothesis_status(ledger, inv, hyp, HypothesisStatus.PAUSED)
        assert ledger.load(inv).hypotheses[hyp].awaiting_calculation_ids == (calc,)
        run_to_parsing(ledger, inv, calc)
        record_evidence(ledger, inv, calc)
        with pytest.raises(InvalidTransition, match="resume"):
            set_hypothesis_status(ledger, inv, hyp, HypothesisStatus.ANALYZING)
        set_hypothesis_status(ledger, inv, hyp, HypothesisStatus.WAITING_FOR_JOB, awaiting=(calc,))
        set_hypothesis_status(ledger, inv, hyp, HypothesisStatus.ANALYZING)
        assert ledger.load(inv).hypotheses[hyp].status is HypothesisStatus.ANALYZING

    def test_rounds_advance_one_at_a_time_and_stamp_events(self, ledger, inv, hyp):
        with pytest.raises(RuleViolation, match="next round must be 1"):
            ledger.record(inv, [draft(CHIEF, RoundStarted(round=2, reason="r"))])
        ledger.record(inv, [draft(CHIEF, RoundStarted(round=1, reason="first round"))])
        result = predict(ledger, inv, hyp)
        assert result.events[0].round == 1
        assert result.state.predictions["pred-1"].round == 1

    def test_closed_investigation_accepts_no_new_hypotheses(self, ledger, inv):
        ledger.record(inv, [draft(USER, InvestigationStatusChanged(
            status=InvestigationStatus.CANCELLED, reason="stopped"))])
        with pytest.raises(RuleViolation, match="CANCELLED"):
            ledger.record(inv, [draft(CHIEF, HypothesisProposed(hypothesis_id="h2", spec=hypothesis_spec()))])

    def test_first_event_must_create_the_investigation(self, ledger):
        with pytest.raises(RuleViolation, match="first event"):
            ledger.record("inv-x", [draft(USER, UserInterventionRecorded(intervention_id="i", text="t"))])


def test_concurrent_writer_forces_a_retry_on_fresh_state(ledger, store, inv):
    other_writer = ResearchLedger(store)
    builds = []

    def build(state):
        builds.append(state.version)
        if len(builds) == 1:
            other_writer.record(inv, [draft(USER, UserInterventionRecorded(intervention_id="sneaky", text="t"))])
        return [draft(USER, UserInterventionRecorded(intervention_id="mine", text="t"))]

    result = ledger.record_with(inv, build)
    assert builds[1] == builds[0] + 1
    assert result.events[0].stream_version == builds[0] + 2
    assert set(result.state.interventions) == {"sneaky", "mine"}


class TestInputExamination:
    def test_inputs_must_be_examined_before_validation(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        with pytest.raises(RuleViolation, match="requires an examination"):
            move(ledger, inv, calc, ENGINE, CalculationState.VALIDATED_INPUT)

    def test_rejected_inputs_cannot_be_validated(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        examine(ledger, inv, calc, ExaminationVerdict.REJECTED)
        with pytest.raises(RuleViolation, match="rejected"):
            move(ledger, inv, calc, ENGINE, CalculationState.VALIDATED_INPUT)
        state = move(ledger, inv, calc, ENGINE, CalculationState.REJECTED, reason="examination rejected").state
        assert state.calculations[calc].state is CalculationState.REJECTED

    def test_inputs_needing_review_run_only_with_user_approval(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        examine(ledger, inv, calc, ExaminationVerdict.NEEDS_REVIEW)
        with pytest.raises(RuleViolation, match="approve_inputs"):
            move(ledger, inv, calc, ENGINE, CalculationState.VALIDATED_INPUT)
        approval = authorize(ledger, inv, PermissionAction.APPROVE_INPUTS, calc, intervention_id="ok-inputs")
        move(ledger, inv, calc, ENGINE, CalculationState.VALIDATED_INPUT, authorization_id=approval)

    def test_examined_files_must_be_the_agents_files(self, ledger, inv, hyp):
        from factories import RAW_OUTPUT

        written = RAW_OUTPUT.model_copy(update={"label": "control.in"})
        calc = request(ledger, inv, hyp, authored_files={"control.in": written}).calculation_id
        with pytest.raises(RuleViolation, match="not the file the agent wrote"):
            examine(ledger, inv, calc)  # synthetic examination holds a different control.in
        examine(ledger, inv, calc, files={"control.in": written})

    def test_examination_happens_once(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        examine(ledger, inv, calc)
        with pytest.raises(RuleViolation, match="already examined"):
            examine(ledger, inv, calc)

    def test_identical_inputs_from_different_specs_are_duplicates(self, ledger, inv, hyp):
        first = request(ledger, inv, hyp).calculation_id
        second = request(ledger, inv, hyp, xc="PBE0").calculation_id
        same_files = examination(first).files
        examine(ledger, inv, first, files=same_files)
        with pytest.raises(RuleViolation, match="identical to calculation"):
            examine(ledger, inv, second, files=same_files)
        examine(ledger, inv, second, ExaminationVerdict.REJECTED, files=same_files)

    def test_only_the_protocol_engine_examines(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        with pytest.raises(RuleViolation, match="hypothesis_agent may not"):
            ledger.record(inv, [draft(AGENT, CalculationInputsExamined(
                calculation_id=calc, examination=examination(calc), workdir=f"{inv}/{calc}"))])


class TestAnalyses:
    @pytest.fixture
    def checked(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        run_to_parsing(ledger, inv, calc)
        check_run(ledger, inv, calc)
        return calc

    def test_only_the_runner_records_analyses_of_checked_runs(self, ledger, inv, hyp, checked):
        unchecked = request(ledger, inv, hyp, xc="PBE0").calculation_id
        with pytest.raises(RuleViolation, match="has not been checked"):
            record_analysis(ledger, inv, hyp, "an-1", unchecked)
        with pytest.raises(RuleViolation, match="hypothesis_agent may not"):
            ledger.record(inv, [draft(AGENT, AnalysisRecorded(analysis_id="an-1", hypothesis_id=hyp,
                                                              run=analysis_run(checked)))])
        state = record_analysis(ledger, inv, hyp, "an-1", checked).state
        assert state.analyses["an-1"].run.calculation_ids == (checked,)
        with pytest.raises(RuleViolation, match="already recorded"):
            record_analysis(ledger, inv, hyp, "an-1", checked)

    def test_staged_files_must_be_the_stored_ones(self, ledger, inv, hyp, checked):
        forged = RAW_OUTPUT.model_copy(update={"sha256": "c" * 64})
        with pytest.raises(RuleViolation, match="differ from the checked run"):
            record_analysis(ledger, inv, hyp, "an-1", checked, files={"aims.out": forged})
        # A file the run check did not store (say, a cube file) may be staged by its own hash.
        record_analysis(ledger, inv, hyp, "an-2", checked, files={"aims.out": RAW_OUTPUT, "dens.cube": forged})

    def test_analysis_evidence_carries_a_reported_result(self, ledger, inv, hyp, checked):
        record_analysis(ledger, inv, hyp, "an-1", checked)
        record_analysis(ledger, inv, hyp, "an-failed", checked, outcome=AnalysisOutcome.FAILED)

        def record(evidence, actor=VALIDATOR):
            return ledger.record(inv, [draft(actor, EvidenceRecorded(evidence_id=f"ev-{len(evidence.summary)}x",
                                                                     evidence=evidence))])

        with pytest.raises(RuleViolation, match="differs from result gap"):
            record(analysis_evidence("an-1", checked, value=6.5))
        with pytest.raises(RuleViolation, match="reported no result homo"):
            record(analysis_evidence("an-1", checked, result="homo"))
        with pytest.raises(RuleViolation, match="did not complete"):
            record(analysis_evidence("an-failed", checked))
        with pytest.raises(RuleViolation, match="exactly the calculations"):
            record(analysis_evidence("an-1", checked, "calc-other"))
        with pytest.raises(RuleViolation, match="hypothesis_agent may not record analysis"):
            record(analysis_evidence("an-1", checked), actor=AGENT)
        state = record(analysis_evidence("an-1", checked)).state
        (evidence_id,) = state.calculations[checked].evidence_ids
        assert [r.evidence_id for r in hypothesis_evidence(state, hyp)] == [evidence_id]

    def test_analysis_evidence_carries_every_check_the_script_reported(self, ledger, inv, hyp, checked):
        electrons = QualityCheck(name="electron_count", passed=True, detail="14 electrons")
        mismatch = QualityCheck(name="matches_printed_gap", passed=False, detail="off by 0.2 eV")
        record_analysis(ledger, inv, hyp, "an-1", checked, results=(("gap", 7.0, "eV"), ("homo", -9.0, "eV")),
                        checks=(electrons,), result_checks={"gap": (mismatch,)})

        def record(evidence_id, evidence):
            return ledger.record(inv, [draft(VALIDATOR, EvidenceRecorded(evidence_id=evidence_id, evidence=evidence))])

        with pytest.raises(RuleViolation, match="exactly the checks the script reported for gap"):
            record("ev-gap", analysis_evidence("an-1", checked, quality=EvidenceQuality.INCONCLUSIVE,
                                               script_checks=(electrons,)))  # the failed check left out
        with pytest.raises(ValueError, match="cannot have failed quality checks"):
            analysis_evidence("an-1", checked, script_checks=(electrons, mismatch))
        record("ev-gap", analysis_evidence("an-1", checked, quality=EvidenceQuality.INCONCLUSIVE,
                                           script_checks=(electrons, mismatch)))
        # The gap's own failed check does not touch the other result.
        state = record("ev-homo", analysis_evidence("an-1", checked, result="homo", value=-9.0,
                                                    script_checks=(electrons,))).state
        assert state.evidence["ev-homo"].evidence.quality is EvidenceQuality.VALIDATED

    def test_analyses_of_unhealthy_runs_validate_nothing(self, ledger, inv, hyp):
        calc = request(ledger, inv, hyp).calculation_id
        run_to_parsing(ledger, inv, calc)
        check_run(ledger, inv, calc, healthy=False)
        record_analysis(ledger, inv, hyp, "an-1", calc)
        with pytest.raises(RuleViolation, match="failed checks"):
            ledger.record(inv, [draft(VALIDATOR, EvidenceRecorded(evidence_id="ev-x",
                                                                  evidence=analysis_evidence("an-1", calc)))])
