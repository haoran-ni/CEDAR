"""Research state rebuilt by replaying the event log, and the rules each event must obey.

``apply`` is the single place where research rules are enforced. The ledger runs every
new event through it before appending, so the log only ever contains events that were
valid when recorded, and replaying the log reproduces the state exactly.

On a ``RuleViolation`` the state passed to ``apply`` must be discarded.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from cedar.contracts.agents import AgentSpec
from cedar.contracts.analysis import SCRIPT_CHECK_PREFIX, AnalysisOutcome, AnalysisRun, as_evidence_check
from cedar.contracts.base import Actor, ActorKind, ArtifactRef
from cedar.contracts.events import (
    APPROVER_ACTIONS,
    AgentRetired,
    AgentRunRecorded,
    AgentsProposed,
    AgentsStarted,
    AgentTaskQueued,
    ProposalWithdrawn,
    AnalysisRecorded,
    ApproverDecision,
    AssessmentRecorded,
    CalculationInputsExamined,
    CalculationLinked,
    CalculationRequested,
    CalculationStateChanged,
    CLOSING_STATUSES,
    ClosingProposed,
    EvidenceRecorded,
    HypothesisProposed,
    HypothesisStatusChanged,
    InvestigationCreated,
    InvestigationStatusChanged,
    MaterialAttached,
    ModeChanged,
    NextActionsRecommended,
    ObservableAlignmentRecorded,
    PlanRevised,
    PredictionRegistered,
    RecordedEvent,
    RetrievalPolicyChanged,
    RoundStarted,
    RunChecked,
    RunRegistered,
    TokenBudgetChanged,
    UserInterventionRecorded,
    WorkbenchPolicyChanged,
    WorkbenchRunRecorded,
)
from cedar.contracts.evidence import EvidenceKind, EvidenceObject
from cedar.contracts.examination import ExaminationVerdict, InputExamination
from cedar.contracts.hypothesis import HypothesisSpec, Prediction
from cedar.contracts.investigation import (
    Discrepancy,
    InvestigationSpec,
    ObservableAlignment,
    RetrievalPolicy,
    discrepancy,
)
from cedar.contracts.plan import Authorization, PermissionAction, Priority, ResearchPlan
from cedar.contracts.status import (
    CALCULATION_MACHINE,
    HYPOTHESIS_MACHINE,
    INVESTIGATION_MACHINE,
    SCHEDULER_TERMINAL_STATES,
    Assessment,
    CalculationState,
    Completeness,
    EvidenceQuality,
    HypothesisStatus,
    InvestigationMode,
    InvestigationStatus,
    SchedulerState,
)
from cedar.contracts.task import ScientificTaskSpec
from cedar.contracts.workbench import WorkbenchRun
from cedar.errors import RuleViolation

_A = ActorKind
_C = CalculationState

EVENT_ACTORS: dict[type, frozenset[ActorKind]] = {
    InvestigationCreated: frozenset({_A.USER, _A.CHIEF_SCIENTIST}),
    InvestigationStatusChanged: frozenset({_A.USER, _A.CHIEF_SCIENTIST, _A.SYSTEM}),
    ClosingProposed: frozenset({_A.CHIEF_SCIENTIST}),
    NextActionsRecommended: frozenset({_A.CHIEF_SCIENTIST}),
    ModeChanged: frozenset({_A.USER, _A.SYSTEM}),
    RetrievalPolicyChanged: frozenset({_A.USER}),
    TokenBudgetChanged: frozenset({_A.USER}),
    WorkbenchPolicyChanged: frozenset({_A.USER}),
    MaterialAttached: frozenset({_A.USER}),
    ApproverDecision: frozenset({_A.APPROVER}),
    ObservableAlignmentRecorded: frozenset({_A.USER, _A.CHIEF_SCIENTIST}),
    UserInterventionRecorded: frozenset({_A.USER}),
    PlanRevised: frozenset({_A.USER, _A.CHIEF_SCIENTIST}),
    RoundStarted: frozenset({_A.CHIEF_SCIENTIST, _A.SYSTEM}),
    HypothesisProposed: frozenset({_A.USER, _A.CHIEF_SCIENTIST}),
    HypothesisStatusChanged: frozenset({_A.CHIEF_SCIENTIST, _A.HYPOTHESIS_AGENT, _A.SYSTEM}),
    PredictionRegistered: frozenset({_A.HYPOTHESIS_AGENT}),
    CalculationRequested: frozenset({_A.CHIEF_SCIENTIST, _A.HYPOTHESIS_AGENT}),
    CalculationLinked: frozenset({_A.CHIEF_SCIENTIST, _A.HYPOTHESIS_AGENT}),
    CalculationInputsExamined: frozenset({_A.PROTOCOL_ENGINE}),
    CalculationStateChanged: frozenset({_A.PROTOCOL_ENGINE, _A.JOB_MANAGER, _A.JOB_MONITOR, _A.VALIDATOR}),
    RunChecked: frozenset({_A.VALIDATOR}),
    RunRegistered: frozenset({_A.VALIDATOR}),
    WorkbenchRunRecorded: frozenset({_A.WORKBENCH}),
    AnalysisRecorded: frozenset({_A.ANALYSIS_RUNNER}),
    EvidenceRecorded: frozenset({_A.VALIDATOR, _A.HYPOTHESIS_AGENT, _A.USER}),
    AssessmentRecorded: frozenset({_A.EVIDENCE_JUDGE}),
    AgentRunRecorded: frozenset({_A.CHIEF_SCIENTIST, _A.HYPOTHESIS_AGENT, _A.EVIDENCE_JUDGE, _A.APPROVER}),
    AgentTaskQueued: frozenset({_A.USER, _A.CHIEF_SCIENTIST, _A.SYSTEM, _A.APPROVER}),
    AgentsProposed: frozenset({_A.CHIEF_SCIENTIST}),
    AgentsStarted: frozenset({_A.USER, _A.SYSTEM, _A.APPROVER}),
    AgentRetired: frozenset({_A.USER, _A.CHIEF_SCIENTIST}),
    ProposalWithdrawn: frozenset({_A.USER, _A.CHIEF_SCIENTIST, _A.APPROVER}),
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
_FOLLOWED = _LIVE_AT_SCHEDULER | {_C.SCHEDULER_TERMINAL, _C.PARSING}
"""Calculations the dispatcher still follows: at the scheduler, or waiting for their run check."""
_OUTCOME_UNSEEN = frozenset({
    _C.SPECIFIED, _C.VALIDATED_INPUT, _C.SUBMISSION_PENDING, _C.SUBMITTED, _C.QUEUED, _C.RUNNING,
})
_CLOSED_HYPOTHESIS = frozenset({HypothesisStatus.DISCARDED, HypothesisStatus.SUPERSEDED})
_NOT_YET_ACTIVE = frozenset({HypothesisStatus.PROPOSED, HypothesisStatus.SCREENED})


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
    registration: RunRegistered | None = None
    """For a run computed outside CEDAR and registered: where it came from, who registered it, what it is."""

    @property
    def registered(self) -> bool:
        return self.registration is not None

    @property
    def shared(self) -> bool:
        """A registered run every agent of the investigation may use."""
        return self.registration is not None and self.registration.hypothesis_id is None

    @property
    def input_files(self) -> dict[str, ArtifactRef]:
        """The files the run ran with: as examined, or as registered."""
        if self.examination is not None:
            return dict(self.examination.files)
        return dict(self.registration.inputs) if self.registration is not None else {}

    @property
    def input_hash(self) -> str | None:
        if self.examination is not None:
            return self.examination.input_hash
        return self.registration.input_hash if self.registration is not None else None

    def visible_to(self, hypothesis_id: str) -> bool:
        """Whether the hypothesis's agent may use this run: its own, or a registered run shared with every agent."""
        return hypothesis_id in self.hypothesis_ids or self.shared


@dataclass
class EvidenceRecord:
    evidence_id: str
    evidence: EvidenceObject
    recorded_at: int
    round: int


@dataclass
class WorkbenchRecord:
    run_id: str
    hypothesis_id: str
    agent_id: str
    run: WorkbenchRun
    recorded_at: int
    round: int


@dataclass(frozen=True)
class WorkbenchPolicyRecord:
    """The researcher's setting of the agents' workbench."""

    enabled: bool
    intervention_id: str
    recorded_at: int


@dataclass
class AnalysisRecord:
    analysis_id: str
    hypothesis_id: str
    run: AnalysisRun
    actor_id: str
    recorded_at: int
    round: int


class ProposalStatus(StrEnum):
    PENDING = "pending"
    """Waiting for the researcher's review."""
    STARTED = "started"
    SUPERSEDED = "superseded"
    """A newer proposal replaced it before it started."""
    WITHDRAWN = "withdrawn"
    """Taken back before it started, e.g. when the researcher halted the investigation."""


@dataclass
class ProposalRecord:
    proposal_id: str
    agents: tuple[AgentSpec, ...]
    rationale: str
    proposed_by: str
    recorded_at: int
    proposed_time: datetime
    round: int
    status: ProposalStatus = ProposalStatus.PENDING
    feedback_ids: list[str] = field(default_factory=list)
    """Researcher messages recorded while it waited; any one of them stops it starting by itself."""
    closed_at: int | None = None
    """Stream version at which it started, was superseded or was withdrawn."""


@dataclass(frozen=True)
class ClosingRecord:
    """The Chief Scientist's proposal to close the investigation, waiting for the researcher."""

    closing_id: str
    status: InvestigationStatus
    reason: str
    proposed_by: str
    recorded_at: int


@dataclass(frozen=True)
class ModeRecord:
    """A switch between human-in-the-loop and full automation."""

    mode: InvestigationMode
    reason: str
    actor: Actor
    recorded_at: int
    max_core_hours: float | None = None
    max_tokens: int | None = None
    intervention_id: str | None = None


@dataclass(frozen=True)
class RetrievalRecord:
    """The researcher's setting of the agents' web access."""

    policy: RetrievalPolicy
    allowed_domains: tuple[str, ...]
    intervention_id: str | None
    """None for the setting the investigation was opened with."""
    recorded_at: int


@dataclass(frozen=True)
class BudgetRecord:
    """The researcher's setting of the investigation's total token budget; None for no limit."""

    max_tokens: int | None
    intervention_id: str
    recorded_at: int


@dataclass(frozen=True)
class MaterialRecord:
    """A file the researcher attached for the agents to view."""

    material_id: str
    name: str
    artifact: ArtifactRef
    note: str | None
    intervention_id: str
    recorded_at: int


@dataclass(frozen=True)
class ApprovalRecord:
    """The approver's decision, in full automation, on proposed agents or a submission."""

    decision_id: str
    action: PermissionAction
    target_id: str
    approved: bool
    reason: str
    recorded_at: int
    core_hours: float | None = None
    """For an approved submission: the core-hours its job script requests, counted against the allowance."""


@dataclass(frozen=True)
class RecommendationRecord:
    """The Chief Scientist's candidate next steps and its choice among them."""

    recommendation: NextActionsRecommended
    recorded_by: str
    recorded_at: int
    round: int

    @property
    def recommendation_id(self) -> str:
        return self.recommendation.recommendation_id


@dataclass
class AgentRecord:
    agent_id: str
    hypothesis_id: str
    spec: AgentSpec
    proposal_id: str
    started_at: int
    started_by: Actor
    authorization_id: str | None
    """The researcher's command that started it; None if the review window passed."""
    retired_at: int | None = None
    retire_reason: str | None = None

    @property
    def active(self) -> bool:
        return self.retired_at is None


@dataclass
class AgentTaskRecord:
    task_id: str
    role: str
    hypothesis_id: str | None
    agent_id: str | None
    instruction: str
    queued_by: str
    queued_at: int
    completed_by: int | None = None
    """Stream version of the run that carried it out."""
    recommendation_id: str | None = None
    """The Chief Scientist's recommendation it carries out, if any."""
    dropped_at: int | None = None
    """Stream version at which its agent was retired before carrying it out."""


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
    proposals: dict[str, ProposalRecord] = field(default_factory=dict)
    agents: dict[str, AgentRecord] = field(default_factory=dict)
    pending_closing: ClosingRecord | None = None
    """A proposal to close the investigation that waits for the researcher's /approve close."""
    recommendations: dict[str, RecommendationRecord] = field(default_factory=dict)
    modes: list[ModeRecord] = field(default_factory=list)
    approvals: dict[str, ApprovalRecord] = field(default_factory=dict)
    retrieval_changes: list[RetrievalRecord] = field(default_factory=list)
    budget_changes: list[BudgetRecord] = field(default_factory=list)
    materials: dict[str, MaterialRecord] = field(default_factory=dict)
    workbench_runs: dict[str, WorkbenchRecord] = field(default_factory=dict)
    workbench_changes: list[WorkbenchPolicyRecord] = field(default_factory=list)

    @property
    def workbench_enabled(self) -> bool:
        """Whether agents may run commands in their workspaces: yes, unless the researcher turned it off."""
        return self.workbench_changes[-1].enabled if self.workbench_changes else True

    @property
    def mode(self) -> InvestigationMode:
        return self.modes[-1].mode if self.modes else InvestigationMode.HUMAN_IN_THE_LOOP

    @property
    def retrieval(self) -> RetrievalRecord:
        """The agents' web access now: the researcher's latest setting, or the one the investigation opened with."""
        if self.retrieval_changes:
            return self.retrieval_changes[-1]
        constraints = self.spec.constraints
        return RetrievalRecord(constraints.external_retrieval, constraints.allowed_domains, None, 1)

    @property
    def automation(self) -> ModeRecord | None:
        """The current stretch of full automation, with its limits; None in human-in-the-loop."""
        return self.modes[-1] if self.mode is InvestigationMode.FULL_AUTOMATION else None

    @property
    def automation_tokens_left(self) -> int | None:
        automation = self.automation
        if automation is None:
            return None
        used = sum(_run_tokens(r.run) for r in self.agent_runs if r.recorded_at > automation.recorded_at)
        return automation.max_tokens - used

    @property
    def automation_core_hours_left(self) -> float | None:
        automation = self.automation
        if automation is None:
            return None
        approved = sum(a.core_hours or 0 for a in self.approvals.values()
                       if a.approved and a.action is PermissionAction.SUBMIT and a.recorded_at > automation.recorded_at)
        return automation.max_core_hours - approved

    def decided_now(self, target_id: str) -> ApprovalRecord | None:
        """The approver's decision on a proposal or calculation in the current stretch of full automation."""
        automation = self.automation
        if automation is None:
            return None
        return next((a for a in reversed(list(self.approvals.values()))
                     if a.target_id == target_id and a.recorded_at > automation.recorded_at), None)

    @property
    def pending_tasks(self) -> list[AgentTaskRecord]:
        return [t for t in self.tasks.values() if t.completed_by is None and t.dropped_at is None]

    @property
    def pending_proposal(self) -> ProposalRecord | None:
        """The proposal waiting for the researcher's review, if any; there is at most one."""
        return next((p for p in self.proposals.values() if p.status is ProposalStatus.PENDING), None)

    def active_agent(self, hypothesis_id: str) -> AgentRecord | None:
        """The hypothesis's agent; a hypothesis has at most one active agent."""
        return next((a for a in self.agents.values() if a.hypothesis_id == hypothesis_id and a.active), None)

    def agent_tokens_used(self, agent_id: str) -> int:
        return sum(_run_tokens(r.run) for r in self.agent_runs if r.actor_id == agent_id)

    @property
    def plan(self) -> ResearchPlan:
        return self.plans[-1].plan

    @property
    def model_tokens_used(self) -> int:
        return sum(_run_tokens(r.run) for r in self.agent_runs)

    @property
    def model_token_budget(self) -> int | None:
        """The total the researcher set with /budget, else the one the investigation opened with; None for none."""
        if self.budget_changes:
            return self.budget_changes[-1].max_tokens
        return self.spec.constraints.model_token_budget

    @property
    def model_tokens_remaining(self) -> int | None:
        budget = self.model_token_budget
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


def _run_tokens(run: AgentRunRecorded) -> int:
    return run.input_tokens + run.output_tokens + run.cache_read_tokens + run.cache_write_tokens


def review_deadline(state: InvestigationState, proposal: ProposalRecord) -> datetime:
    """When proposed agents start by themselves if the researcher has not replied."""
    return proposal.proposed_time + timedelta(minutes=state.spec.constraints.agent_review_minutes)


def agent_id_for(state: InvestigationState, hypothesis_id: str) -> str:
    """The id a new agent for the hypothesis gets: agent-<hypothesis>, numbered if one was retired."""
    base = f"agent-{hypothesis_id}"[:120]
    taken = set(state.agents)
    if base not in taken:
        return base
    return next(f"{base}-{n}" for n in range(2, len(taken) + 3) if f"{base}-{n}" not in taken)


def evidence_owner(state: InvestigationState, evidence: EvidenceObject) -> str | None:
    """The hypothesis whose agent produced the evidence, if an agent did."""
    if evidence.analysis_id is not None:
        return state.analyses[evidence.analysis_id].hypothesis_id
    reader = evidence.interpreted_by
    if reader is not None and reader.id in state.agents:
        return state.agents[reader.id].hypothesis_id
    return None


def reusable_calculation(state: InvestigationState, task_hash: str) -> str | None:
    """An existing calculation with identical scientific content that can be shared.

    Rejected or cancelled calculations, and runs the scheduler ended without success,
    do not count: requesting the same task again is then a fresh attempt.
    """
    for calc in state.calculations.values():
        if calc.task_hash == task_hash and not calc.registered and _is_reusable(calc):
            return calc.calculation_id
    return None


def _is_reusable(calc: CalculationRecord) -> bool:
    if calc.registered:
        return calc.registration.healthy
    if calc.state in (_C.REJECTED, _C.CANCELLED):
        return False
    return not (
        calc.scheduler_state in SCHEDULER_TERMINAL_STATES and calc.scheduler_state is not SchedulerState.COMPLETED
    )


def calculation_with_inputs(state: InvestigationState, input_hash: str, hypothesis_ids=()) -> str | None:
    """A reusable calculation whose scientific inputs are identical. A registered run counts only for the
    hypotheses that may use it, so no agent is kept from a run it cannot see."""
    for calc in state.calculations.values():
        if calc.input_hash != input_hash or not _is_reusable(calc):
            continue
        if not calc.registered or any(calc.visible_to(h) for h in hypothesis_ids):
            return calc.calculation_id
    return None


def hypothesis_calculations(state: InvestigationState, hypothesis_id: str) -> list[CalculationRecord]:
    """The runs the hypothesis's agent may use: its own, then the registered runs shared with every agent."""
    own = [state.calculations[c] for c in state.hypotheses[hypothesis_id].calculation_ids]
    return own + [c for c in state.calculations.values() if c.shared]


def hypothesis_evidence(state: InvestigationState, hypothesis_id: str) -> list[EvidenceRecord]:
    """Evidence from the hypothesis's calculations and from the analyses it ran, in recording order."""
    hyp = state.hypotheses[hypothesis_id]
    calcs = set(hyp.calculation_ids)
    found = []
    for record in state.evidence.values():
        ev = record.evidence
        mine = evidence_owner(state, ev) == hypothesis_id
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


def _require_own(state: InvestigationState, actor: Actor, hypothesis_id: str) -> None:
    """A hypothesis agent acts only on its own hypothesis, and only while it is active."""
    if actor.kind is not ActorKind.HYPOTHESIS_AGENT:
        return
    agent = state.agents.get(actor.id)
    if agent is None or not agent.active or agent.hypothesis_id != hypothesis_id:
        raise RuleViolation(f"{actor.id} is not the active agent of {hypothesis_id}")


def _calculation(state: InvestigationState, calculation_id: str) -> CalculationRecord:
    try:
        return state.calculations[calculation_id]
    except KeyError:
        raise RuleViolation(f"unknown calculation {calculation_id}") from None


def authorization_for(state: InvestigationState, action: PermissionAction, target_id: str) -> str | None:
    """The most recent authorization of ``action`` on ``target_id``: the researcher's own command, or, in full
    automation, the approver's approval in the current stretch; None if there is neither."""
    for record in reversed(list(state.interventions.values())):
        if any(a.action is action and target_id in a.target_ids for a in record.authorizations):
            return record.intervention_id
    for approval in reversed(list(state.approvals.values())):
        if approval.action is action and approval.target_id == target_id and _approval_counts(state, approval):
            return approval.decision_id
    return None


def _approval_counts(state: InvestigationState, approval: ApprovalRecord) -> bool:
    """An approver's approval counts in the stretch of full automation it was given in. A submission it allowed
    that is already under way is still recorded, so switching modes never strands a job at the scheduler."""
    if not approval.approved:
        return False
    calc = state.calculations.get(approval.target_id)
    if approval.action is PermissionAction.SUBMIT and calc is not None and calc.state is _C.SUBMISSION_PENDING:
        return True
    automation = state.automation
    return automation is not None and approval.recorded_at > automation.recorded_at


def _require_authorization(
    state: InvestigationState, authorization_id: str | None, action: PermissionAction, target_id: str
) -> None:
    intervention = state.interventions.get(authorization_id) if authorization_id else None
    if intervention is not None and any(
        a.action is action and target_id in a.target_ids for a in intervention.authorizations
    ):
        return
    approval = state.approvals.get(authorization_id) if authorization_id else None
    if approval is None or approval.action is not action or approval.target_id != target_id or not _approval_counts(
        state, approval
    ):
        raise RuleViolation(f"{action} of {target_id} requires a user authorization naming it")


def _on_status_changed(state: InvestigationState, event: RecordedEvent, p: InvestigationStatusChanged) -> None:
    closing = state.pending_closing
    if p.status in CLOSING_STATUSES:
        if event.actor.kind is not ActorKind.USER:
            raise RuleViolation("only the researcher closes an investigation, with /approve close, once the Chief "
                                "Scientist has proposed it")
        if closing is None or closing.status is not p.status:
            raise RuleViolation(f"no proposal to close the investigation as {p.status} waits for approval")
        _require_authorization(state, p.authorization_id, PermissionAction.CLOSE_INVESTIGATION, closing.closing_id)
        followed = [c.calculation_id for c in state.calculations.values() if c.state in _FOLLOWED]
        if followed:  # the dispatcher follows open investigations only
            raise RuleViolation(f"{', '.join(followed)} still at the scheduler or waiting for a run check; wait for "
                                "them, or cancel them, before closing")
    elif p.authorization_id is not None:
        raise RuleViolation("only closing the investigation carries the researcher's authorization")
    state.status, state.resume_to = INVESTIGATION_MACHINE.advance(state.status, p.status, state.resume_to)
    state.pending_closing = None  # closed as proposed; any other change withdraws the proposal


def _on_mode_changed(state: InvestigationState, event: RecordedEvent, p: ModeChanged) -> None:
    _require_open(state)
    if event.actor.kind is ActorKind.USER:
        if p.intervention_id not in state.interventions:
            raise RuleViolation("the researcher's switch of mode cites their recorded command")
    elif p.mode is not InvestigationMode.HUMAN_IN_THE_LOOP:
        raise RuleViolation("only the researcher switches full automation on")
    state.modes.append(ModeRecord(p.mode, p.reason, event.actor, event.stream_version, p.max_core_hours,
                                  p.max_tokens, p.intervention_id))


def _on_retrieval_policy(state: InvestigationState, event: RecordedEvent, p: RetrievalPolicyChanged) -> None:
    _require_open(state)
    if p.intervention_id not in state.interventions:
        raise RuleViolation("the researcher's setting of web access cites their recorded command")
    state.retrieval_changes.append(RetrievalRecord(p.policy, p.allowed_domains, p.intervention_id,
                                                   event.stream_version))


def _on_token_budget(state: InvestigationState, event: RecordedEvent, p: TokenBudgetChanged) -> None:
    _require_open(state)
    if p.intervention_id not in state.interventions:
        raise RuleViolation("the researcher's token budget cites their recorded command")
    state.budget_changes.append(BudgetRecord(p.max_tokens, p.intervention_id, event.stream_version))


def _on_workbench_policy(state: InvestigationState, event: RecordedEvent, p: WorkbenchPolicyChanged) -> None:
    _require_open(state)
    if p.intervention_id not in state.interventions:
        raise RuleViolation("the researcher's setting of the workbench cites their recorded command")
    state.workbench_changes.append(WorkbenchPolicyRecord(p.enabled, p.intervention_id, event.stream_version))


def _on_workbench_run(state: InvestigationState, event: RecordedEvent, p: WorkbenchRunRecorded) -> None:
    _require_open(state)
    _hypothesis(state, p.hypothesis_id)
    if p.run_id in state.workbench_runs:
        raise RuleViolation(f"workbench run {p.run_id} already recorded")
    agent = state.agents.get(p.agent_id)
    if agent is None or agent.hypothesis_id != p.hypothesis_id:
        raise RuleViolation(f"{p.agent_id} is not an agent of {p.hypothesis_id}")
    # A command already running when its agent was retired, or the workbench turned off, is still recorded.
    state.workbench_runs[p.run_id] = WorkbenchRecord(p.run_id, p.hypothesis_id, p.agent_id, p.run,
                                                     event.stream_version, event.round)


def workbench_run_with(state: InvestigationState, hypothesis_id: str, sha256: str) -> str | None:
    """The hypothesis's latest workbench run that wrote a file with this content, if one did."""
    for record in reversed(list(state.workbench_runs.values())):
        if record.hypothesis_id == hypothesis_id and sha256 in record.run.changed.values():
            return record.run_id
    return None


def _on_material(state: InvestigationState, event: RecordedEvent, p: MaterialAttached) -> None:
    _require_open(state)
    if p.intervention_id not in state.interventions:
        raise RuleViolation("an attached material cites the researcher's recorded command")
    if p.material_id in state.materials:
        raise RuleViolation(f"material {p.material_id} already exists")
    state.materials[p.material_id] = MaterialRecord(p.material_id, p.name, p.artifact, p.note, p.intervention_id,
                                                    event.stream_version)


def _on_approver_decision(state: InvestigationState, event: RecordedEvent, p: ApproverDecision) -> None:
    _require_open(state)
    if state.automation is None:
        raise RuleViolation("the approver decides only while the investigation is in full automation")
    if p.decision_id in state.approvals:
        raise RuleViolation(f"decision {p.decision_id} already exists")
    if state.decided_now(p.target_id) is not None:
        raise RuleViolation(f"the approver has already decided on {p.target_id} in this stretch of full automation")
    core_hours = None
    if p.action is PermissionAction.START_AGENTS:
        proposal = state.pending_proposal
        if proposal is None or proposal.proposal_id != p.target_id:
            raise RuleViolation(f"proposal {p.target_id} is not waiting to start")
        if proposal.feedback_ids:
            raise RuleViolation(f"the researcher replied ({', '.join(proposal.feedback_ids)}); proposal "
                                f"{p.target_id} waits for them")
    else:
        calc = state.calculations.get(p.target_id)
        if calc is None or calc.state is not _C.VALIDATED_INPUT:
            raise RuleViolation(f"calculation {p.target_id} is not waiting to be submitted")
        if not state.spec.constraints.submit_requires_approval or authorization_for(
                state, PermissionAction.SUBMIT, p.target_id) is not None:
            raise RuleViolation(f"calculation {p.target_id} needs no decision")
        if p.approved:
            core_hours = calc.examination.requested_core_hours
            if core_hours is None:
                raise RuleViolation(f"the core-hours {p.target_id} requests are not known; it waits for the "
                                    "researcher")
            left = state.automation_core_hours_left
            if core_hours > left:
                raise RuleViolation(f"{p.target_id} requests {core_hours:g} core-hours; {max(left, 0):g} of full "
                                    f"automation's {state.automation.max_core_hours:g} remain, so it waits for the "
                                    "researcher")
    state.approvals[p.decision_id] = ApprovalRecord(p.decision_id, p.action, p.target_id, p.approved, p.reason,
                                                    event.stream_version, core_hours)


def _on_next_actions(state: InvestigationState, event: RecordedEvent, p: NextActionsRecommended) -> None:
    _require_open(state)
    if p.recommendation_id in state.recommendations:
        raise RuleViolation(f"recommendation {p.recommendation_id} already exists")
    unknown = sorted({h for c in p.candidates for h in c.hypothesis_ids} - set(state.hypotheses))
    if unknown:
        raise RuleViolation(f"unknown hypotheses {', '.join(unknown)}; propose them before recommending tests of them")
    state.recommendations[p.recommendation_id] = RecommendationRecord(p, event.actor.id, event.stream_version,
                                                                      event.round)


def _require_recommendation(state: InvestigationState, recommendation_id: str | None) -> None:
    if recommendation_id is not None and recommendation_id not in state.recommendations:
        raise RuleViolation(f"unknown recommendation {recommendation_id}")


def _on_closing_proposed(state: InvestigationState, event: RecordedEvent, p: ClosingProposed) -> None:
    if state.status is not InvestigationStatus.REVIEWING:
        raise RuleViolation(f"closing is proposed while REVIEWING, not {state.status}; move the investigation to "
                            "REVIEWING first")
    state.pending_closing = ClosingRecord(p.closing_id, p.status, p.reason, event.actor.id, event.stream_version)


def _on_alignment(state: InvestigationState, event: RecordedEvent, p: ObservableAlignmentRecorded) -> None:
    _require_open(state)
    state.alignment = p.alignment


def _on_intervention(state: InvestigationState, event: RecordedEvent, p: UserInterventionRecorded) -> None:
    if p.intervention_id in state.interventions:
        raise RuleViolation(f"intervention {p.intervention_id} already recorded")
    state.interventions[p.intervention_id] = InterventionRecord(
        p.intervention_id, p.text, p.authorizations, event.actor.id, event.stream_version
    )
    proposal = state.pending_proposal
    if proposal is not None and not p.authorizations and not p.text.startswith("/"):  # a reply, not a command
        proposal.feedback_ids.append(p.intervention_id)


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
    _require_own(state, event.actor, p.hypothesis_id)
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
    _require_own(state, event.actor, p.hypothesis_id)
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
    _require_own(state, event.actor, p.hypothesis_id)
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
    _require_own(state, event.actor, p.hypothesis_id)
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
        duplicate = calculation_with_inputs(state, exam.input_hash, calc.hypothesis_ids)
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


def _on_run_registered(state: InvestigationState, event: RecordedEvent, p: RunRegistered) -> None:
    _require_open(state)
    if p.calculation_id in state.calculations:
        raise RuleViolation(f"calculation {p.calculation_id} already exists")
    requester = p.requested_by
    if requester.kind is ActorKind.USER and p.intervention_id not in state.interventions:
        raise RuleViolation("a run the researcher registers cites their recorded command")
    hypothesis_ids = []
    if p.hypothesis_id is not None:
        hyp = _hypothesis(state, p.hypothesis_id)
        if hyp.status in _CLOSED_HYPOTHESIS:
            raise RuleViolation(f"hypothesis {p.hypothesis_id} is {hyp.status}")
        _require_own(state, requester, p.hypothesis_id)
        hyp.calculation_ids.append(p.calculation_id)
        hypothesis_ids = [p.hypothesis_id]
    state.calculations[p.calculation_id] = CalculationRecord(
        p.calculation_id, p.task, p.task_hash, hypothesis_ids, event.stream_version, state=_C.QUALITY_ASSESSED,
        run_check=p.as_run_check(), history=[(event.stream_version, _C.QUALITY_ASSESSED)], registration=p)


def _on_analysis(state: InvestigationState, event: RecordedEvent, p: AnalysisRecorded) -> None:
    _require_open(state)
    hyp = _hypothesis(state, p.hypothesis_id)
    if hyp.status in _CLOSED_HYPOTHESIS:
        raise RuleViolation(f"hypothesis {p.hypothesis_id} is {hyp.status}")
    if p.analysis_id in state.analyses:
        raise RuleViolation(f"analysis {p.analysis_id} already recorded")
    for staged in p.run.inputs:
        calc = _calculation(state, staged.calculation_id)
        if not calc.visible_to(p.hypothesis_id):
            raise RuleViolation(f"calculation {calc.calculation_id} is not linked to {p.hypothesis_id}; an analysis "
                                "uses only its own hypothesis's runs and shared registered runs")
        if calc.run_check is None:
            raise RuleViolation(f"the run of {calc.calculation_id} has not been checked; nothing to analyse yet")
        known = {**calc.input_files, **calc.run_check.outputs}
        changed = [name for name, ref in staged.files.items() if name in known and known[name].sha256 != ref.sha256]
        if changed:
            raise RuleViolation(f"staged files of {calc.calculation_id} differ from the checked run: {changed}")
    state.analyses[p.analysis_id] = AnalysisRecord(p.analysis_id, p.hypothesis_id, p.run, event.actor.id,
                                                   event.stream_version, event.round)


def _require_healthy_for_validation(ev: EvidenceObject, calcs: list[CalculationRecord]) -> None:
    if ev.quality is not EvidenceQuality.VALIDATED:
        return
    for calc in calcs:
        if not calc.registered and calc.scheduler_state is not SchedulerState.COMPLETED:
            raise RuleViolation(f"cannot validate output of a job that ended {calc.scheduler_state}")
        if not calc.run_check.healthy:
            raise RuleViolation(f"cannot validate results of {calc.calculation_id}: its run failed checks")


def _require_reader_owns(state: InvestigationState, ev: EvidenceObject, linked: list[CalculationRecord]) -> None:
    """An agent's reading or analysis result concerns its own hypothesis's runs."""
    reader = ev.interpreted_by
    if reader is None or reader.kind is not ActorKind.HYPOTHESIS_AGENT:
        return
    agent = state.agents.get(reader.id)
    if agent is None or not agent.active:
        raise RuleViolation(f"{reader.id} is not an active agent")
    if ev.analysis_id is not None:
        if state.analyses[ev.analysis_id].hypothesis_id != agent.hypothesis_id:
            raise RuleViolation(f"analysis {ev.analysis_id} was not run for {agent.hypothesis_id}")
    elif not all(calc.visible_to(agent.hypothesis_id) for calc in linked):
        raise RuleViolation(f"{reader.id} reads only calculations linked to {agent.hypothesis_id} and shared "
                            "registered runs")


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
        if calc.registered != (ev.kind is EvidenceKind.ARCHIVED_COMPUTATION):
            raise RuleViolation(f"a reading of {calc.calculation_id} is "
                                + ("archived computation evidence: it is a registered run" if calc.registered else
                                   "direct computation evidence: CEDAR ran it"))
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
    _require_reader_owns(state, ev, linked)
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
    if p.role in ("chief_scientist", "approver"):
        if event.actor.kind is not ActorKind.SYSTEM:
            raise RuleViolation("only the dispatcher queues " + ("the Chief Scientist's reviews" if p.role ==
                                                                 "chief_scientist" else "the approver's work"))
        if p.role == "approver" and state.automation is None:
            raise RuleViolation("the approver works only while the investigation is in full automation")
    else:
        _hypothesis(state, p.hypothesis_id)
    if p.agent_id is not None:
        agent = state.agents.get(p.agent_id)
        if agent is None or not agent.active:
            raise RuleViolation(f"no active agent {p.agent_id}")
        if agent.hypothesis_id != p.hypothesis_id:
            raise RuleViolation(f"{p.agent_id} works on {agent.hypothesis_id}, not {p.hypothesis_id}")
    _require_recommendation(state, p.recommendation_id)
    state.tasks[p.task_id] = AgentTaskRecord(p.task_id, p.role, p.hypothesis_id, p.agent_id, p.instruction,
                                             event.actor.id, event.stream_version,
                                             recommendation_id=p.recommendation_id)


def _on_agent_run(state: InvestigationState, event: RecordedEvent, p: AgentRunRecorded) -> None:
    if p.task_id is not None:
        task = state.tasks.get(p.task_id)
        if task is None:
            raise RuleViolation(f"unknown task {p.task_id}")
        if task.completed_by is not None:
            raise RuleViolation(f"task {p.task_id} was already carried out")
        if task.role != p.role:
            raise RuleViolation(f"task {p.task_id} is for a {task.role}, not a {p.role}")
        if task.agent_id is not None and event.actor.id != task.agent_id:
            raise RuleViolation(f"task {p.task_id} is for {task.agent_id}, not {event.actor.id}")
        # A run already going when its agent was retired is still recorded, so its tokens count.
        task.completed_by = event.stream_version
    state.agent_runs.append(AgentRunRecord(p.session_id, p.role, event.actor.id, p, event.stream_version, event.round))


def _require_can_have_agent(state: InvestigationState, hypothesis_id: str) -> None:
    hyp = _hypothesis(state, hypothesis_id)
    if hyp.status in _NOT_YET_ACTIVE or hyp.status in _CLOSED_HYPOTHESIS:
        raise RuleViolation(f"hypothesis {hypothesis_id} is {hyp.status}; agents work on activated hypotheses")
    agent = state.active_agent(hypothesis_id)
    if agent is not None:
        raise RuleViolation(f"hypothesis {hypothesis_id} already has an agent, {agent.agent_id}")


def _on_agents_proposed(state: InvestigationState, event: RecordedEvent, p: AgentsProposed) -> None:
    if state.status is not InvestigationStatus.INVESTIGATING:
        raise RuleViolation(f"agents are proposed while INVESTIGATING, not {state.status}")
    if p.proposal_id in state.proposals:
        raise RuleViolation(f"proposal {p.proposal_id} already exists")
    _require_recommendation(state, p.recommendation_id)
    for spec in p.agents:
        _require_can_have_agent(state, spec.hypothesis_id)
    earlier = state.pending_proposal
    if earlier is not None:
        earlier.status, earlier.closed_at = ProposalStatus.SUPERSEDED, event.stream_version
    state.proposals[p.proposal_id] = ProposalRecord(p.proposal_id, p.agents, p.rationale, event.actor.id,
                                                    event.stream_version, event.occurred_at, event.round)


def _on_agents_started(state: InvestigationState, event: RecordedEvent, p: AgentsStarted) -> None:
    proposal = state.proposals.get(p.proposal_id)
    if proposal is None or proposal.status is not ProposalStatus.PENDING:
        raise RuleViolation(f"proposal {p.proposal_id} is not waiting to start")
    if state.status is not InvestigationStatus.INVESTIGATING:
        raise RuleViolation(f"agents start while INVESTIGATING, not {state.status}")
    if event.actor.kind in (ActorKind.USER, ActorKind.APPROVER):
        _require_authorization(state, p.authorization_id, PermissionAction.START_AGENTS, p.proposal_id)
        if event.actor.kind is ActorKind.APPROVER and proposal.feedback_ids:
            raise RuleViolation(f"the researcher replied; proposal {p.proposal_id} waits for them")
    else:
        if state.automation is not None:
            raise RuleViolation("in full automation the approver, not the review window, starts proposed agents")
        if p.authorization_id is not None:
            raise RuleViolation("only the researcher's own command starts agents early")
        if proposal.feedback_ids:
            raise RuleViolation(f"the researcher replied ({', '.join(proposal.feedback_ids)}); proposal "
                                f"{p.proposal_id} waits for a revision or their command")
        if event.occurred_at < review_deadline(state, proposal):
            raise RuleViolation(f"proposal {p.proposal_id} is still in the researcher's review window")
    if set(p.agent_ids) != {spec.hypothesis_id for spec in proposal.agents}:
        raise RuleViolation("agents must start for exactly the proposed hypotheses")
    ids = list(p.agent_ids.values())
    if len(set(ids)) != len(ids) or set(ids) & set(state.agents):
        raise RuleViolation("each new agent needs an id no agent has had")
    for spec in proposal.agents:
        _require_can_have_agent(state, spec.hypothesis_id)
        agent_id = p.agent_ids[spec.hypothesis_id]
        state.agents[agent_id] = AgentRecord(agent_id, spec.hypothesis_id, spec, p.proposal_id, event.stream_version,
                                             event.actor, p.authorization_id)
    proposal.status, proposal.closed_at = ProposalStatus.STARTED, event.stream_version


def _on_proposal_withdrawn(state: InvestigationState, event: RecordedEvent, p: ProposalWithdrawn) -> None:
    proposal = state.proposals.get(p.proposal_id)
    if proposal is None or proposal.status is not ProposalStatus.PENDING:
        raise RuleViolation(f"proposal {p.proposal_id} is not waiting to start")
    if event.actor.kind is ActorKind.APPROVER:
        decision = state.decided_now(p.proposal_id)
        if decision is None or decision.approved:
            raise RuleViolation(f"the approver withdraws only a proposal it declined; {p.proposal_id} was not")
    proposal.status, proposal.closed_at = ProposalStatus.WITHDRAWN, event.stream_version


def _on_agent_retired(state: InvestigationState, event: RecordedEvent, p: AgentRetired) -> None:
    agent = state.agents.get(p.agent_id)
    if agent is None or not agent.active:
        raise RuleViolation(f"no active agent {p.agent_id}")
    agent.retired_at, agent.retire_reason = event.stream_version, p.reason
    for task in state.pending_tasks:
        if task.agent_id == p.agent_id:
            task.dropped_at = event.stream_version


_HANDLERS = {
    InvestigationStatusChanged: _on_status_changed,
    ClosingProposed: _on_closing_proposed,
    ModeChanged: _on_mode_changed,
    RetrievalPolicyChanged: _on_retrieval_policy,
    TokenBudgetChanged: _on_token_budget,
    WorkbenchPolicyChanged: _on_workbench_policy,
    MaterialAttached: _on_material,
    ApproverDecision: _on_approver_decision,
    NextActionsRecommended: _on_next_actions,
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
    RunRegistered: _on_run_registered,
    WorkbenchRunRecorded: _on_workbench_run,
    AnalysisRecorded: _on_analysis,
    EvidenceRecorded: _on_evidence,
    AssessmentRecorded: _on_assessment,
    AgentTaskQueued: _on_agent_task,
    AgentRunRecorded: _on_agent_run,
    AgentsProposed: _on_agents_proposed,
    AgentsStarted: _on_agents_started,
    AgentRetired: _on_agent_retired,
    ProposalWithdrawn: _on_proposal_withdrawn,
}

__all__ = [
    "AgentRecord", "AnalysisRecord", "AssessmentEntry", "CalculationRecord", "ClosingRecord", "EvidenceRecord", "HypothesisRecord", "InterventionRecord",
    "InvestigationState", "ApprovalRecord", "BudgetRecord", "MaterialRecord", "ModeRecord", "PlanRecord", "RetrievalRecord", "PredictionRecord", "ProposalRecord",
    "ProposalStatus", "RecommendationRecord", "agent_id_for", "apply",
    "authorization_for", "calculation_with_inputs", "evidence_owner", "hypothesis_calculations", "hypothesis_evidence",
    "replay",
    "reusable_calculation", "review_deadline", "WorkbenchPolicyRecord", "WorkbenchRecord", "workbench_run_with",
]
