"""Research state rebuilt by replaying the event log, and the rules each event must obey.

``apply`` is the single place where research rules are enforced. The ledger runs every
new event through it before appending, so the log only ever contains events that were
valid when recorded, and replaying the log reproduces the state exactly.

On a ``RuleViolation`` the state passed to ``apply`` must be discarded.
"""

from dataclasses import dataclass, field

from sim2real.contracts.analysis import SCRIPT_CHECK_PREFIX, AnalysisOutcome, AnalysisRun, as_evidence_check
from sim2real.contracts.base import ActorKind
from sim2real.contracts.events import (
    AgentRunRecorded,
    AgentTaskQueued,
    AnalysisRecorded,
    AssessmentRecorded,
    CalculationInputsExamined,
    CalculationLinked,
    CalculationRequested,
    CalculationStateChanged,
    EvidenceRecorded,
    HypothesisProposed,
    HypothesisStatusChanged,
    InvestigationCreated,
    InvestigationStatusChanged,
    ObservableAlignmentRecorded,
    PlanRevised,
    PredictionRegistered,
    RecordedEvent,
    RoundStarted,
    RunChecked,
    UserInterventionRecorded,
)
from sim2real.contracts.evidence import EvidenceKind, EvidenceObject
from sim2real.contracts.examination import ExaminationVerdict, InputExamination
from sim2real.contracts.hypothesis import HypothesisSpec, Prediction
from sim2real.contracts.investigation import (
    Discrepancy,
    InvestigationSpec,
    ObservableAlignment,
    discrepancy,
)
from sim2real.contracts.plan import Authorization, PermissionAction, Priority, ResearchPlan
from sim2real.contracts.status import (
    CALCULATION_MACHINE,
    HYPOTHESIS_MACHINE,
    INVESTIGATION_MACHINE,
    SCHEDULER_TERMINAL_STATES,
    Assessment,
    CalculationState,
    Completeness,
    EvidenceQuality,
    HypothesisStatus,
    InvestigationStatus,
    SchedulerState,
)
from sim2real.contracts.task import ScientificTaskSpec
from sim2real.errors import RuleViolation

_A = ActorKind
_C = CalculationState

EVENT_ACTORS: dict[type, frozenset[ActorKind]] = {
    InvestigationCreated: frozenset({_A.USER, _A.CHIEF_SCIENTIST}),
    InvestigationStatusChanged: frozenset({_A.USER, _A.CHIEF_SCIENTIST, _A.SYSTEM}),
    ObservableAlignmentRecorded: frozenset({_A.USER, _A.CHIEF_SCIENTIST}),
    UserInterventionRecorded: frozenset({_A.USER}),
    PlanRevised: frozenset({_A.USER, _A.CHIEF_SCIENTIST}),
    RoundStarted: frozenset({_A.CHIEF_SCIENTIST, _A.SYSTEM}),
    HypothesisProposed: frozenset({_A.USER, _A.CHIEF_SCIENTIST, _A.HYPOTHESIS_AGENT}),
    HypothesisStatusChanged: frozenset({_A.CHIEF_SCIENTIST, _A.HYPOTHESIS_AGENT, _A.SYSTEM}),
    PredictionRegistered: frozenset({_A.HYPOTHESIS_AGENT}),
    CalculationRequested: frozenset({_A.CHIEF_SCIENTIST, _A.HYPOTHESIS_AGENT}),
    CalculationLinked: frozenset({_A.CHIEF_SCIENTIST, _A.HYPOTHESIS_AGENT}),
    CalculationInputsExamined: frozenset({_A.PROTOCOL_ENGINE}),
    CalculationStateChanged: frozenset({_A.PROTOCOL_ENGINE, _A.JOB_MANAGER, _A.JOB_MONITOR, _A.VALIDATOR}),
    RunChecked: frozenset({_A.VALIDATOR}),
    AnalysisRecorded: frozenset({_A.ANALYSIS_RUNNER}),
    EvidenceRecorded: frozenset({_A.VALIDATOR, _A.HYPOTHESIS_AGENT, _A.USER}),
    AssessmentRecorded: frozenset({_A.EVIDENCE_JUDGE}),
    AgentRunRecorded: frozenset({_A.CHIEF_SCIENTIST, _A.HYPOTHESIS_AGENT, _A.EVIDENCE_JUDGE}),
    AgentTaskQueued: frozenset({_A.USER, _A.CHIEF_SCIENTIST, _A.SYSTEM}),
}
"""Who may record each event type. Nobody but the computing services moves a calculation."""

CALCULATION_STATE_ACTORS: dict[CalculationState, frozenset[ActorKind]] = {
    _C.VALIDATED_INPUT: frozenset({_A.PROTOCOL_ENGINE}),
    _C.REJECTED: frozenset({_A.PROTOCOL_ENGINE}),
    _C.SUBMISSION_PENDING: frozenset({_A.JOB_MANAGER}),
    _C.SUBMITTED: frozenset({_A.JOB_MANAGER}),
    _C.QUEUED: frozenset({_A.JOB_MONITOR}),
    _C.RUNNING: frozenset({_A.JOB_MONITOR}),
    _C.SCHEDULER_TERMINAL: frozenset({_A.JOB_MONITOR}),
    _C.PARSING: frozenset({_A.VALIDATOR}),
    _C.QUALITY_ASSESSED: frozenset({_A.VALIDATOR}),
    _C.CANCELLED: frozenset({_A.JOB_MANAGER}),
}

EVIDENCE_KIND_ACTORS: dict[EvidenceKind, frozenset[ActorKind]] = {
    EvidenceKind.DIRECT_COMPUTATION: frozenset({_A.VALIDATOR}),
    EvidenceKind.ARCHIVED_COMPUTATION: frozenset({_A.VALIDATOR}),
    EvidenceKind.PUBLISHED_EXPERIMENT: frozenset({_A.HYPOTHESIS_AGENT, _A.USER}),
    EvidenceKind.INFERENCE: frozenset({_A.HYPOTHESIS_AGENT, _A.USER}),
    EvidenceKind.INDIRECT_CONSTRAINT: frozenset({_A.HYPOTHESIS_AGENT, _A.USER}),
    EvidenceKind.ANALYSIS: frozenset({_A.VALIDATOR}),
}

_IMPLIED_SCHEDULER_STATE = {_C.QUEUED: SchedulerState.PENDING, _C.RUNNING: SchedulerState.RUNNING}
_LIVE_AT_SCHEDULER = frozenset({_C.SUBMISSION_PENDING, _C.SUBMITTED, _C.QUEUED, _C.RUNNING})
"""SUBMISSION_PENDING counts as live: sbatch may already have accepted the job."""
_OUTCOME_UNSEEN = frozenset({
    _C.SPECIFIED, _C.VALIDATED_INPUT, _C.SUBMISSION_PENDING, _C.SUBMITTED, _C.QUEUED, _C.RUNNING,
})
_CLOSED_HYPOTHESIS = frozenset({HypothesisStatus.DISCARDED, HypothesisStatus.SUPERSEDED})


@dataclass
class PlanRecord:
    version: int
    plan: ResearchPlan
    reason: str
    recorded_at: int
    intervention_id: str | None = None


@dataclass
class InterventionRecord:
    intervention_id: str
    text: str
    authorizations: tuple[Authorization, ...]
    actor_id: str
    recorded_at: int


@dataclass
class PredictionRecord:
    prediction_id: str
    hypothesis_id: str
    prediction: Prediction
    registered_at: int
    round: int


@dataclass
class AssessmentEntry:
    assessment: Assessment
    completeness: Completeness
    evidence_ids: tuple[str, ...]
    rationale: str
    limitations: tuple[str, ...]
    recorded_at: int
    round: int
    plan_version: int


@dataclass
class HypothesisRecord:
    hypothesis_id: str
    spec: HypothesisSpec
    proposed_at: int
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    resume_to: HypothesisStatus | None = None
    awaiting_calculation_ids: tuple[str, ...] = ()
    assessment: Assessment = Assessment.NOT_TESTED
    prediction_ids: list[str] = field(default_factory=list)
    calculation_ids: list[str] = field(default_factory=list)
    assessments: list[AssessmentEntry] = field(default_factory=list)


@dataclass
class CalculationRecord:
    calculation_id: str
    task: ScientificTaskSpec
    task_hash: str
    hypothesis_ids: list[str]
    requested_at: int
    state: CalculationState = CalculationState.SPECIFIED
    scheduler_state: SchedulerState | None = None
    scheduler_job_id: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    """Evidence read from this run directly or computed by analyses that used it."""
    examination: InputExamination | None = None
    workdir: str | None = None
    run_check: RunChecked | None = None
    history: list[tuple[int, CalculationState]] = field(default_factory=list)


@dataclass
class EvidenceRecord:
    evidence_id: str
    evidence: EvidenceObject
    recorded_at: int
    round: int


@dataclass
class AnalysisRecord:
    analysis_id: str
    hypothesis_id: str
    run: AnalysisRun
    actor_id: str
    recorded_at: int
    round: int


@dataclass
class AgentTaskRecord:
    task_id: str
    role: str
    hypothesis_id: str
    instruction: str
    queued_by: str
    queued_at: int
    completed_by: int | None = None
    """Stream version of the run that carried it out."""


@dataclass
class AgentRunRecord:
    session_id: str
    role: str
    actor_id: str
    run: AgentRunRecorded
    recorded_at: int
    round: int


@dataclass
class InvestigationState:
    investigation_id: str
    spec: InvestigationSpec
    alignment: ObservableAlignment
    plans: list[PlanRecord]
    status: InvestigationStatus = InvestigationStatus.CREATED
    resume_to: InvestigationStatus | None = None
    round: int = 0
    version: int = 0
    interventions: dict[str, InterventionRecord] = field(default_factory=dict)
    hypotheses: dict[str, HypothesisRecord] = field(default_factory=dict)
    predictions: dict[str, PredictionRecord] = field(default_factory=dict)
    calculations: dict[str, CalculationRecord] = field(default_factory=dict)
    evidence: dict[str, EvidenceRecord] = field(default_factory=dict)
    analyses: dict[str, AnalysisRecord] = field(default_factory=dict)
    agent_runs: list[AgentRunRecord] = field(default_factory=list)
    tasks: dict[str, AgentTaskRecord] = field(default_factory=dict)

    @property
    def pending_tasks(self) -> list[AgentTaskRecord]:
        return [t for t in self.tasks.values() if t.completed_by is None]

    @property
    def plan(self) -> ResearchPlan:
        return self.plans[-1].plan

    @property
    def model_tokens_used(self) -> int:
        return sum(r.run.input_tokens + r.run.output_tokens + r.run.cache_read_tokens + r.run.cache_write_tokens
                   for r in self.agent_runs)

    @property
    def model_tokens_remaining(self) -> int | None:
        budget = self.spec.constraints.model_token_budget
        return None if budget is None else max(0, budget - self.model_tokens_used)

    @property
    def plan_version(self) -> int:
        return self.plans[-1].version

    def discrepancy(self, simulation: int = 0, experiment: int = 0) -> Discrepancy:
        return discrepancy(
            self.spec.simulations[simulation].quantity,
            self.spec.experiments[experiment].quantity,
            self.alignment,
        )


def reusable_calculation(state: InvestigationState, task_hash: str) -> str | None:
    """An existing calculation with identical scientific content that can be shared.

    Rejected or cancelled calculations, and runs the scheduler ended without success,
    do not count: requesting the same task again is then a fresh attempt.
    """
    for calc in state.calculations.values():
        if calc.task_hash == task_hash and _is_reusable(calc):
            return calc.calculation_id
    return None


def _is_reusable(calc: CalculationRecord) -> bool:
    if calc.state in (_C.REJECTED, _C.CANCELLED):
        return False
    return not (
        calc.scheduler_state in SCHEDULER_TERMINAL_STATES and calc.scheduler_state is not SchedulerState.COMPLETED
    )


def calculation_with_inputs(state: InvestigationState, input_hash: str) -> str | None:
    """A reusable calculation whose examined scientific inputs are identical."""
    for calc in state.calculations.values():
        if calc.examination is not None and calc.examination.input_hash == input_hash and _is_reusable(calc):
            return calc.calculation_id
    return None


def hypothesis_evidence(state: InvestigationState, hypothesis_id: str) -> list[EvidenceRecord]:
    """Evidence from the hypothesis's calculations and from the analyses it ran, in recording order."""
    hyp = state.hypotheses[hypothesis_id]
    calcs = set(hyp.calculation_ids)
    found = []
    for record in state.evidence.values():
        ev = record.evidence
        mine = ev.analysis_id is not None and state.analyses[ev.analysis_id].hypothesis_id == hypothesis_id
        if mine or ev.calculation_id in calcs or calcs.intersection(ev.calculation_ids):
            found.append(record)
    return found


def replay(events: list[RecordedEvent]) -> InvestigationState | None:
    state = None
    for event in events:
        state = apply(state, event)
    return state


def apply(state: InvestigationState | None, event: RecordedEvent) -> InvestigationState:
    payload = event.payload
    if event.actor.kind not in EVENT_ACTORS[type(payload)]:
        raise RuleViolation(f"{event.actor.kind} may not record {payload.type}")

    if isinstance(payload, InvestigationCreated):
        if state is not None:
            raise RuleViolation(f"investigation {state.investigation_id} already exists")
        if (event.stream_version, event.round, event.plan_version) != (1, 0, 1):
            raise RuleViolation("investigation_created must be version 1, round 0, plan 1")
        return InvestigationState(
            investigation_id=event.investigation_id,
            spec=payload.spec,
            alignment=payload.spec.alignment,
            plans=[PlanRecord(1, payload.plan, "initial plan", recorded_at=1)],
            version=1,
        )

    if state is None:
        raise RuleViolation("the first event of an investigation must be investigation_created")
    if event.investigation_id != state.investigation_id:
        raise RuleViolation("event belongs to a different investigation")
    if event.stream_version != state.version + 1:
        raise RuleViolation(f"expected stream version {state.version + 1}, got {event.stream_version}")
    if event.round != state.round:
        raise RuleViolation(f"event is stamped round {event.round}, investigation is in round {state.round}")
    if event.plan_version > state.plan_version:
        raise RuleViolation(f"event cites plan version {event.plan_version}, which does not exist yet")

    _HANDLERS[type(payload)](state, event, payload)
    state.version = event.stream_version
    return state


def _require_open(state: InvestigationState) -> None:
    if state.status in INVESTIGATION_MACHINE.terminal:
        raise RuleViolation(f"investigation is {state.status}")


def _hypothesis(state: InvestigationState, hypothesis_id: str) -> HypothesisRecord:
    try:
        return state.hypotheses[hypothesis_id]
    except KeyError:
        raise RuleViolation(f"unknown hypothesis {hypothesis_id}") from None


def _calculation(state: InvestigationState, calculation_id: str) -> CalculationRecord:
    try:
        return state.calculations[calculation_id]
    except KeyError:
        raise RuleViolation(f"unknown calculation {calculation_id}") from None


def authorization_for(state: InvestigationState, action: PermissionAction, target_id: str) -> str | None:
    """The most recent user intervention authorizing ``action`` on ``target_id``, if any."""
    for record in reversed(list(state.interventions.values())):
        if any(a.action is action and target_id in a.target_ids for a in record.authorizations):
            return record.intervention_id
    return None


def _require_authorization(
    state: InvestigationState, authorization_id: str | None, action: PermissionAction, target_id: str
) -> None:
    intervention = state.interventions.get(authorization_id) if authorization_id else None
    if intervention is None or not any(
        a.action is action and target_id in a.target_ids for a in intervention.authorizations
    ):
        raise RuleViolation(f"{action} of {target_id} requires a user authorization naming it")


def _on_status_changed(state: InvestigationState, event: RecordedEvent, p: InvestigationStatusChanged) -> None:
    state.status, state.resume_to = INVESTIGATION_MACHINE.advance(state.status, p.status, state.resume_to)


def _on_alignment(state: InvestigationState, event: RecordedEvent, p: ObservableAlignmentRecorded) -> None:
    _require_open(state)
    state.alignment = p.alignment


def _on_intervention(state: InvestigationState, event: RecordedEvent, p: UserInterventionRecorded) -> None:
    if p.intervention_id in state.interventions:
        raise RuleViolation(f"intervention {p.intervention_id} already recorded")
    state.interventions[p.intervention_id] = InterventionRecord(
        p.intervention_id, p.text, p.authorizations, event.actor.id, event.stream_version
    )


def _on_plan_revised(state: InvestigationState, event: RecordedEvent, p: PlanRevised) -> None:
    if p.version != state.plan_version + 1:
        raise RuleViolation(f"next plan version must be {state.plan_version + 1}, got {p.version}")
    if p.intervention_id is not None and p.intervention_id not in state.interventions:
        raise RuleViolation(f"unknown intervention {p.intervention_id}")
    unknown = set(p.plan.hypothesis_priorities) - set(state.hypotheses)
    if unknown:
        raise RuleViolation(f"plan prioritizes unknown hypotheses: {sorted(unknown)}")
    state.plans.append(PlanRecord(p.version, p.plan, p.reason, event.stream_version, p.intervention_id))


def _on_round_started(state: InvestigationState, event: RecordedEvent, p: RoundStarted) -> None:
    _require_open(state)
    if p.round != state.round + 1:
        raise RuleViolation(f"next round must be {state.round + 1}, got {p.round}")
    state.round = p.round


def _on_hypothesis_proposed(state: InvestigationState, event: RecordedEvent, p: HypothesisProposed) -> None:
    _require_open(state)
    if p.hypothesis_id in state.hypotheses:
        raise RuleViolation(f"hypothesis {p.hypothesis_id} already exists")
    unknown = (set(p.spec.depends_on) | set(p.spec.interacts_with)) - set(state.hypotheses)
    if unknown:
        raise RuleViolation(f"hypothesis refers to unknown hypotheses: {sorted(unknown)}")
    state.hypotheses[p.hypothesis_id] = HypothesisRecord(p.hypothesis_id, p.spec, event.stream_version)


def _on_hypothesis_status(state: InvestigationState, event: RecordedEvent, p: HypothesisStatusChanged) -> None:
    hyp = _hypothesis(state, p.hypothesis_id)
    resuming = hyp.status in HYPOTHESIS_MACHINE.suspend
    status, resume_to = HYPOTHESIS_MACHINE.advance(hyp.status, p.status, hyp.resume_to)
    for calc_id in p.awaiting_calculation_ids:
        calc = _calculation(state, calc_id)
        if p.hypothesis_id not in calc.hypothesis_ids:
            raise RuleViolation(f"calculation {calc_id} is not linked to {p.hypothesis_id}")
        # A job may finish while its agent is paused; resuming must still be possible.
        if not resuming and calc.state in CALCULATION_MACHINE.terminal:
            raise RuleViolation(f"calculation {calc_id} is already {calc.state}; nothing to wait for")
    hyp.status, hyp.resume_to = status, resume_to
    if status is HypothesisStatus.PAUSED:
        return  # keep the awaited calculations for when it resumes
    hyp.awaiting_calculation_ids = p.awaiting_calculation_ids


def _on_prediction(state: InvestigationState, event: RecordedEvent, p: PredictionRegistered) -> None:
    hyp = _hypothesis(state, p.hypothesis_id)
    if hyp.status in _CLOSED_HYPOTHESIS:
        raise RuleViolation(f"hypothesis {p.hypothesis_id} is {hyp.status}")
    if p.prediction_id in state.predictions:
        raise RuleViolation(f"prediction {p.prediction_id} already registered; predictions are immutable")
    for calc_id in p.prediction.calculation_ids:
        calc = _calculation(state, calc_id)
        if p.hypothesis_id not in calc.hypothesis_ids:
            raise RuleViolation(f"calculation {calc_id} is not linked to {p.hypothesis_id}")
        if calc.state not in _OUTCOME_UNSEEN:
            raise RuleViolation(f"calculation {calc_id} has already ended ({calc.state}); too late to predict it")
    state.predictions[p.prediction_id] = PredictionRecord(
        p.prediction_id, p.hypothesis_id, p.prediction, event.stream_version, event.round
    )
    hyp.prediction_ids.append(p.prediction_id)


def _require_can_add_test(state: InvestigationState, hyp: HypothesisRecord) -> None:
    if hyp.status is not HypothesisStatus.PLANNING_TEST:
        raise RuleViolation(f"hypothesis {hyp.hypothesis_id} is {hyp.status}, not PLANNING_TEST")
    if state.plan.priority(hyp.hypothesis_id) is Priority.ON_HOLD:
        raise RuleViolation(f"hypothesis {hyp.hypothesis_id} is on hold in plan v{state.plan_version}")


def _on_calculation_requested(state: InvestigationState, event: RecordedEvent, p: CalculationRequested) -> None:
    if state.status is not InvestigationStatus.INVESTIGATING:
        raise RuleViolation(f"calculations can only be requested while INVESTIGATING, not {state.status}")
    hyp = _hypothesis(state, p.hypothesis_id)
    _require_can_add_test(state, hyp)
    if p.calculation_id in state.calculations:
        raise RuleViolation(f"calculation {p.calculation_id} already exists")
    existing = reusable_calculation(state, p.task_hash)
    if existing is not None:
        raise RuleViolation(f"an identical task already exists as {existing}; link it instead")
    state.calculations[p.calculation_id] = CalculationRecord(
        p.calculation_id, p.task, p.task_hash, [p.hypothesis_id], event.stream_version,
        history=[(event.stream_version, _C.SPECIFIED)],
    )
    hyp.calculation_ids.append(p.calculation_id)


def _on_calculation_linked(state: InvestigationState, event: RecordedEvent, p: CalculationLinked) -> None:
    calc = _calculation(state, p.calculation_id)
    hyp = _hypothesis(state, p.hypothesis_id)
    _require_can_add_test(state, hyp)
    if p.hypothesis_id in calc.hypothesis_ids:
        raise RuleViolation(f"calculation {p.calculation_id} is already linked to {p.hypothesis_id}")
    calc.hypothesis_ids.append(p.hypothesis_id)
    hyp.calculation_ids.append(p.calculation_id)


def _on_inputs_examined(state: InvestigationState, event: RecordedEvent, p: CalculationInputsExamined) -> None:
    calc = _calculation(state, p.calculation_id)
    exam = p.examination
    if calc.state is not _C.SPECIFIED or calc.examination is not None:
        raise RuleViolation(f"calculation {calc.calculation_id} was already examined or has moved on")
    if exam.protocol != calc.task.protocol:
        raise RuleViolation("examination used a different protocol than the task requested")
    for name, ref in calc.task.authored_files.items():
        examined = exam.files.get(name)
        if examined is None or examined.sha256 != ref.sha256:
            raise RuleViolation(f"examined {name} is not the file the agent wrote")
    structure = calc.task.structure
    if structure is not None and structure.sha256 not in {ref.sha256 for ref in exam.files.values()}:
        raise RuleViolation("the requested structure is not among the examined files")
    if exam.verdict is not ExaminationVerdict.REJECTED:
        duplicate = calculation_with_inputs(state, exam.input_hash)
        if duplicate is not None:
            raise RuleViolation(f"inputs are identical to calculation {duplicate}")
    calc.examination = exam
    calc.workdir = p.workdir


def _on_calculation_state(state: InvestigationState, event: RecordedEvent, p: CalculationStateChanged) -> None:
    calc = _calculation(state, p.calculation_id)
    if event.actor.kind not in CALCULATION_STATE_ACTORS[p.state]:
        raise RuleViolation(f"{event.actor.kind} may not move a calculation to {p.state}")
    new_state, _ = CALCULATION_MACHINE.advance(calc.state, p.state)

    scheduler_state = p.scheduler_state
    if p.state in _IMPLIED_SCHEDULER_STATE:
        implied = _IMPLIED_SCHEDULER_STATE[p.state]
        if scheduler_state not in (None, implied):
            raise RuleViolation(f"{p.state} implies scheduler state {implied}, got {scheduler_state}")
        scheduler_state = implied
    elif p.state is _C.SCHEDULER_TERMINAL:
        if scheduler_state not in SCHEDULER_TERMINAL_STATES:
            raise RuleViolation(f"SCHEDULER_TERMINAL needs a terminal scheduler state, got {scheduler_state}")
    elif scheduler_state is not None:
        raise RuleViolation(f"{p.state} does not take a scheduler state")

    if (p.state is _C.SUBMITTED) != (p.scheduler_job_id is not None):
        raise RuleViolation("a scheduler job id is required for, and only allowed on, SUBMITTED")
    if p.state is _C.VALIDATED_INPUT:
        exam = calc.examination
        if exam is None:
            raise RuleViolation("VALIDATED_INPUT requires an examination of the inputs")
        if exam.verdict is ExaminationVerdict.REJECTED:
            raise RuleViolation("the examination rejected these inputs")
        if exam.verdict is ExaminationVerdict.NEEDS_REVIEW:
            _require_authorization(state, p.authorization_id, PermissionAction.APPROVE_INPUTS, calc.calculation_id)
    if p.state is _C.SUBMITTED:
        if any(c.scheduler_job_id == p.scheduler_job_id for c in state.calculations.values()):
            raise RuleViolation(f"scheduler job {p.scheduler_job_id} is already attached to a calculation")
        if state.spec.constraints.submit_requires_approval:
            _require_authorization(state, p.authorization_id, PermissionAction.SUBMIT, calc.calculation_id)
    if p.state in (_C.REJECTED, _C.CANCELLED) and not (p.reason and p.reason.strip()):
        raise RuleViolation(f"{p.state} needs a reason")
    if p.state is _C.CANCELLED and calc.state in _LIVE_AT_SCHEDULER:
        _require_authorization(state, p.authorization_id, PermissionAction.CANCEL, calc.calculation_id)
    if p.state is _C.QUALITY_ASSESSED and calc.run_check is None:
        raise RuleViolation("QUALITY_ASSESSED requires the run check to be recorded first")

    calc.state = new_state
    if scheduler_state is not None:
        calc.scheduler_state = scheduler_state
    if p.scheduler_job_id is not None:
        calc.scheduler_job_id = p.scheduler_job_id
    calc.history.append((event.stream_version, new_state))


def _on_run_checked(state: InvestigationState, event: RecordedEvent, p: RunChecked) -> None:
    calc = _calculation(state, p.calculation_id)
    if calc.state is not _C.PARSING:
        raise RuleViolation(f"a run is checked while PARSING; {calc.calculation_id} is {calc.state}")
    if calc.run_check is not None:
        raise RuleViolation(f"the run of {calc.calculation_id} was already checked")
    calc.run_check = p


def _on_analysis(state: InvestigationState, event: RecordedEvent, p: AnalysisRecorded) -> None:
    _require_open(state)
    hyp = _hypothesis(state, p.hypothesis_id)
    if hyp.status in _CLOSED_HYPOTHESIS:
        raise RuleViolation(f"hypothesis {p.hypothesis_id} is {hyp.status}")
    if p.analysis_id in state.analyses:
        raise RuleViolation(f"analysis {p.analysis_id} already recorded")
    for staged in p.run.inputs:
        calc = _calculation(state, staged.calculation_id)
        if calc.run_check is None:
            raise RuleViolation(f"the run of {calc.calculation_id} has not been checked; nothing to analyse yet")
        known = {**calc.examination.files, **calc.run_check.outputs}
        changed = [name for name, ref in staged.files.items() if name in known and known[name].sha256 != ref.sha256]
        if changed:
            raise RuleViolation(f"staged files of {calc.calculation_id} differ from the checked run: {changed}")
    state.analyses[p.analysis_id] = AnalysisRecord(p.analysis_id, p.hypothesis_id, p.run, event.actor.id,
                                                   event.stream_version, event.round)


def _require_healthy_for_validation(ev: EvidenceObject, calcs: list[CalculationRecord]) -> None:
    if ev.quality is not EvidenceQuality.VALIDATED:
        return
    for calc in calcs:
        if calc.scheduler_state is not SchedulerState.COMPLETED:
            raise RuleViolation(f"cannot validate output of a job that ended {calc.scheduler_state}")
        if not calc.run_check.healthy:
            raise RuleViolation(f"cannot validate results of {calc.calculation_id}: its run failed checks")


def _on_evidence(state: InvestigationState, event: RecordedEvent, p: EvidenceRecorded) -> None:
    ev = p.evidence
    if event.actor.kind not in EVIDENCE_KIND_ACTORS[ev.kind]:
        raise RuleViolation(f"{event.actor.kind} may not record {ev.kind} evidence")
    if p.evidence_id in state.evidence:
        raise RuleViolation(f"evidence {p.evidence_id} already recorded; evidence is immutable")
    linked: list[CalculationRecord] = []
    if ev.calculation_id is not None:
        calc = _calculation(state, ev.calculation_id)
        if calc.run_check is None or calc.state is not _C.QUALITY_ASSESSED:
            raise RuleViolation(f"evidence for {calc.calculation_id} waits for its run check")
        linked = [calc]
    if ev.kind is EvidenceKind.ANALYSIS:
        analysis = state.analyses.get(ev.analysis_id)
        if analysis is None:
            raise RuleViolation(f"unknown analysis {ev.analysis_id}")
        if analysis.run.outcome is not AnalysisOutcome.COMPLETED:
            raise RuleViolation(f"analysis {ev.analysis_id} did not complete ({analysis.run.outcome})")
        if ev.calculation_ids != analysis.run.calculation_ids:
            raise RuleViolation("analysis evidence must name exactly the calculations the analysis used")
        result = analysis.run.result(ev.analysis_result)
        if result is None:
            raise RuleViolation(f"analysis {ev.analysis_id} reported no result {ev.analysis_result}")
        if ev.value != result.value:
            raise RuleViolation(f"the value differs from result {result.name} of analysis {ev.analysis_id}")
        # Every check the script made on this result travels with it, failed ones included.
        expected = {as_evidence_check(c) for c in analysis.run.checks_for(result)}
        carried = {c for c in ev.quality_checks if c.name.startswith(SCRIPT_CHECK_PREFIX)}
        if carried != expected:
            raise RuleViolation(f"analysis evidence must carry exactly the checks the script reported for "
                                f"{result.name}")
        linked = [state.calculations[c] for c in ev.calculation_ids]
    _require_healthy_for_validation(ev, linked)
    state.evidence[p.evidence_id] = EvidenceRecord(p.evidence_id, ev, event.stream_version, event.round)
    for calc in linked:
        calc.evidence_ids.append(p.evidence_id)


def _on_assessment(state: InvestigationState, event: RecordedEvent, p: AssessmentRecorded) -> None:
    hyp = _hypothesis(state, p.hypothesis_id)
    if len(set(p.evidence_ids)) != len(p.evidence_ids):
        raise RuleViolation("evidence is cited more than once")
    cited = []
    for evidence_id in p.evidence_ids:
        if evidence_id not in state.evidence:
            raise RuleViolation(f"unknown evidence {evidence_id}")
        cited.append(state.evidence[evidence_id])

    if p.assessment is Assessment.NOT_TESTED and cited:
        raise RuleViolation("NOT_TESTED cannot cite evidence")
    if p.assessment in (Assessment.SUPPORTED, Assessment.CHALLENGED):
        if not cited:
            raise RuleViolation(f"{p.assessment} must cite evidence")
        invalid = [r.evidence_id for r in cited if r.evidence.quality is EvidenceQuality.INVALID]
        if invalid:
            raise RuleViolation(f"INVALID evidence cannot support or challenge a hypothesis: {invalid}")
        if p.completeness is Completeness.FINAL:
            unvalidated = [r.evidence_id for r in cited if r.evidence.quality is not EvidenceQuality.VALIDATED]
            if unvalidated:
                raise RuleViolation(f"a FINAL {p.assessment} needs VALIDATED evidence: {unvalidated}")
        first_prediction = min(
            (state.predictions[pid].registered_at for pid in hyp.prediction_ids), default=None
        )
        for record in cited:
            if first_prediction is None or first_prediction > record.recorded_at:
                raise RuleViolation(
                    f"evidence {record.evidence_id} was recorded before any prediction of "
                    f"{p.hypothesis_id}, so it cannot test it"
                )

    hyp.assessment = p.assessment
    hyp.assessments.append(AssessmentEntry(
        p.assessment, p.completeness, p.evidence_ids, p.rationale, p.limitations,
        event.stream_version, event.round, event.plan_version,
    ))


def _on_agent_task(state: InvestigationState, event: RecordedEvent, p: AgentTaskQueued) -> None:
    _require_open(state)
    if p.task_id in state.tasks:
        raise RuleViolation(f"task {p.task_id} already exists")
    _hypothesis(state, p.hypothesis_id)
    state.tasks[p.task_id] = AgentTaskRecord(p.task_id, p.role, p.hypothesis_id, p.instruction, event.actor.id,
                                             event.stream_version)


def _on_agent_run(state: InvestigationState, event: RecordedEvent, p: AgentRunRecorded) -> None:
    if p.task_id is not None:
        task = state.tasks.get(p.task_id)
        if task is None:
            raise RuleViolation(f"unknown task {p.task_id}")
        if task.completed_by is not None:
            raise RuleViolation(f"task {p.task_id} was already carried out")
        if task.role != p.role:
            raise RuleViolation(f"task {p.task_id} is for a {task.role}, not a {p.role}")
        task.completed_by = event.stream_version
    state.agent_runs.append(AgentRunRecord(p.session_id, p.role, event.actor.id, p, event.stream_version, event.round))


_HANDLERS = {
    InvestigationStatusChanged: _on_status_changed,
    ObservableAlignmentRecorded: _on_alignment,
    UserInterventionRecorded: _on_intervention,
    PlanRevised: _on_plan_revised,
    RoundStarted: _on_round_started,
    HypothesisProposed: _on_hypothesis_proposed,
    HypothesisStatusChanged: _on_hypothesis_status,
    PredictionRegistered: _on_prediction,
    CalculationRequested: _on_calculation_requested,
    CalculationLinked: _on_calculation_linked,
    CalculationInputsExamined: _on_inputs_examined,
    CalculationStateChanged: _on_calculation_state,
    RunChecked: _on_run_checked,
    AnalysisRecorded: _on_analysis,
    EvidenceRecorded: _on_evidence,
    AssessmentRecorded: _on_assessment,
    AgentTaskQueued: _on_agent_task,
    AgentRunRecorded: _on_agent_run,
}

__all__ = [
    "AnalysisRecord", "AssessmentEntry", "CalculationRecord", "EvidenceRecord", "HypothesisRecord", "InterventionRecord",
    "InvestigationState", "PlanRecord", "PredictionRecord", "apply", "authorization_for", "calculation_with_inputs",
    "hypothesis_evidence", "replay",
    "reusable_calculation",
]
