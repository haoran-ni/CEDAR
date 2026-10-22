"""Event payloads and the envelope under which they are recorded.

The event log is the source of truth: research state is whatever replaying these events
produces. Payloads only add information; nothing edits or deletes an earlier record.
"""

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import AfterValidator, AwareDatetime, Field, JsonValue, StringConstraints, model_validator

from cedar.contracts.agents import AgentReport, AgentSpec
from cedar.contracts.analysis import AnalysisRun
from cedar.contracts.base import (
    Actor,
    ActorKind,
    ArtifactRef,
    Contract,
    FileName,
    Identifier,
    NonEmptyStr,
    RelativePath,
    Sha256Hex,
    canonical_json,
    sha256_hex,
)
from cedar.contracts.evidence import EvidenceObject, QualityCheck, RunReading
from cedar.contracts.examination import Finding, InputExamination, scientific_input_hash
from cedar.contracts.hypothesis import HypothesisSpec, Prediction
from cedar.contracts.investigation import (
    Domain,
    InvestigationSpec,
    ObservableAlignment,
    RetrievalPolicy,
    check_retrieval,
)
from cedar.contracts.next_actions import CandidateAction
from cedar.contracts.plan import Authorization, PermissionAction, ResearchPlan
from cedar.contracts.status import (
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


class InvestigationCreated(Contract):
    type: Literal["investigation_created"] = "investigation_created"
    spec: InvestigationSpec
    plan: ResearchPlan


class InvestigationStatusChanged(Contract):
    type: Literal["investigation_status_changed"] = "investigation_status_changed"
    status: InvestigationStatus
    reason: NonEmptyStr
    authorization_id: Identifier | None = None
    """The researcher's command that closes the investigation; required for COMPLETED and INCONCLUSIVE."""


CLOSING_STATUSES = frozenset({InvestigationStatus.COMPLETED, InvestigationStatus.INCONCLUSIVE})
"""How an investigation is closed: only by the researcher, approving the Chief Scientist's proposal."""


class ClosingProposed(Contract):
    """The Chief Scientist proposes closing the investigation; only the researcher's command closes it.

    A newer proposal replaces one still waiting, and any other status change withdraws it.
    """

    type: Literal["closing_proposed"] = "closing_proposed"
    closing_id: Identifier
    status: InvestigationStatus
    reason: NonEmptyStr
    """Why now: what was found, and what remains open."""

    @model_validator(mode="after")
    def _closes(self) -> "ClosingProposed":
        if self.status not in CLOSING_STATUSES:
            raise ValueError("an investigation is closed as COMPLETED or INCONCLUSIVE")
        return self


class HypothesisStanding(Contract):
    """Where one hypothesis stands in a comparison across hypotheses."""

    hypothesis_id: Identifier
    standing: NonEmptyStr
    """Where it stands now, in the light of the evidence and the other hypotheses."""
    accounts_for: str | None = None
    """How much of the discrepancy it accounts for, in words: the values are those of the evidence it cites, never
    typed here. None when that is not known."""
    evidence_ids: tuple[Identifier, ...] = ()


class SynthesisRecorded(Contract):
    """The Chief Scientist's comparison across hypotheses at a review: where each stands, how they interact, what of
    the discrepancy is explained and what is not, and the new hypotheses the comparison suggests."""

    type: Literal["synthesis_recorded"] = "synthesis_recorded"
    synthesis_id: Identifier
    summary: NonEmptyStr
    hypotheses: tuple[HypothesisStanding, ...] = Field(min_length=1)
    interactions: tuple[NonEmptyStr, ...] = ()
    """Conflicts, confounders and interactions between hypotheses."""
    residual: NonEmptyStr
    """What of the discrepancy remains unexplained."""
    not_separable: tuple[NonEmptyStr, ...] = ()
    """Contributions the evidence cannot tell apart."""
    new_hypotheses: tuple[Identifier, ...] = ()
    """Hypotheses proposed because of this comparison."""

    @model_validator(mode="after")
    def _each_hypothesis_once(self) -> "SynthesisRecorded":
        ids = [h.hypothesis_id for h in self.hypotheses]
        if len(set(ids)) != len(ids):
            raise ValueError("a comparison names each hypothesis once")
        return self


class NextActionsRecommended(Contract):
    """The Chief Scientist's candidate next steps, compared, and the ones it takes now.

    A record of reasoning for the researcher and for evaluation; it authorizes nothing.
    """

    type: Literal["next_actions_recommended"] = "next_actions_recommended"
    recommendation_id: Identifier
    candidates: tuple[CandidateAction, ...] = Field(min_length=1)
    chosen: tuple[Identifier, ...] = ()
    """Candidates taken now; none is a valid choice, e.g. while jobs run."""
    reason: NonEmptyStr
    """Why these over the others, or why none now."""

    @model_validator(mode="after")
    def _chosen_among_candidates(self) -> "NextActionsRecommended":
        ids = [c.candidate_id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate ids must be unique")
        if len(set(self.chosen)) != len(self.chosen) or not set(self.chosen) <= set(ids):
            raise ValueError("chosen names distinct candidates of this recommendation")
        return self


class ModeChanged(Contract):
    """The researcher switches the investigation between human-in-the-loop and full automation; the system
    switches it back, and pauses it, once full automation's spending allowance is used up."""

    type: Literal["mode_changed"] = "mode_changed"
    mode: InvestigationMode
    reason: NonEmptyStr
    max_core_hours: float | None = Field(default=None, gt=0)
    """Full automation: the core-hours the approver may approve from now on, as job scripts request them."""
    max_usd: float | None = Field(default=None, gt=0)
    """Full automation: the dollars all agents together may spend on models from now on."""
    intervention_id: Identifier | None = None
    """The researcher's command, when it was the researcher who switched."""

    @model_validator(mode="after")
    def _limits_only_for_automation(self) -> "ModeChanged":
        automated = self.mode is InvestigationMode.FULL_AUTOMATION
        if automated != (self.max_core_hours is not None) or automated != (self.max_usd is not None):
            raise ValueError("full automation needs a core-hour and a dollar limit; human-in-the-loop takes none")
        return self


class RetrievalPolicyChanged(Contract):
    """The researcher sets whether the investigation's agents may use the web. Hypothesis agents and the Chief
    Scientist then get the provider's web search and fetch; the Evidence Judge and the approver never do."""

    type: Literal["retrieval_policy_changed"] = "retrieval_policy_changed"
    policy: RetrievalPolicy
    allowed_domains: tuple[Domain, ...] = ()
    intervention_id: Identifier
    """The researcher's command."""

    @model_validator(mode="after")
    def _domains_with_curated_only(self) -> "RetrievalPolicyChanged":
        check_retrieval(self.policy, self.allowed_domains)
        return self


class BudgetChanged(Contract):
    """The researcher sets what the investigation's agents may spend on models in total, in dollars, or removes the
    limit. Everything spent since it opened counts, at the price of each kind of token."""

    type: Literal["budget_changed"] = "budget_changed"
    max_usd: float | None = Field(default=None, gt=0)
    """None: no limit."""
    intervention_id: Identifier
    """The researcher's command."""


class WorkbenchPolicyChanged(Contract):
    """The researcher turns the agents' workbench (sandboxed commands in their workspaces) on or off. It is on
    unless the researcher turned it off."""

    type: Literal["workbench_policy_changed"] = "workbench_policy_changed"
    enabled: bool
    intervention_id: Identifier
    """The researcher's command."""


MaterialName = Annotated[str, StringConstraints(min_length=1, max_length=255, pattern=r"^[^/\x00]+$")]


class MaterialAttached(Contract):
    """A file the researcher gives the investigation's agents, such as a figure, a paper or measured data. Every
    agent of the investigation may view it. It is context, not evidence."""

    type: Literal["material_attached"] = "material_attached"
    material_id: Identifier
    name: MaterialName
    """The file's name as the researcher had it."""
    artifact: ArtifactRef
    note: str | None = None
    """What the researcher said about it."""
    intervention_id: Identifier
    """The researcher's command."""


APPROVER_ACTIONS = frozenset({PermissionAction.START_AGENTS, PermissionAction.SUBMIT})
"""What the approver decides in full automation. Inputs needing review, cancellations and closing stay with the
researcher."""


class ApproverDecision(Contract):
    """In full automation, the approver's decision on the researcher's behalf: whether proposed agents start, or a
    calculation is submitted."""

    type: Literal["approver_decision"] = "approver_decision"
    decision_id: Identifier
    action: PermissionAction
    target_id: Identifier
    """The proposal or calculation decided on."""
    approved: bool
    reason: NonEmptyStr

    @model_validator(mode="after")
    def _approver_action(self) -> "ApproverDecision":
        if self.action not in APPROVER_ACTIONS:
            raise ValueError(f"the approver does not decide {self.action.value}; that stays with the researcher")
        return self


class ObservableAlignmentRecorded(Contract):
    type: Literal["observable_alignment_recorded"] = "observable_alignment_recorded"
    alignment: ObservableAlignment


class UserInterventionRecorded(Contract):
    type: Literal["user_intervention_recorded"] = "user_intervention_recorded"
    intervention_id: Identifier
    text: NonEmptyStr
    """The user's instruction, verbatim."""
    authorizations: tuple[Authorization, ...] = ()


class PlanRevised(Contract):
    type: Literal["plan_revised"] = "plan_revised"
    version: int = Field(ge=2)
    plan: ResearchPlan
    reason: NonEmptyStr
    intervention_id: Identifier | None = None


class RoundStarted(Contract):
    type: Literal["round_started"] = "round_started"
    round: int = Field(ge=1)
    reason: NonEmptyStr


class HypothesisProposed(Contract):
    type: Literal["hypothesis_proposed"] = "hypothesis_proposed"
    hypothesis_id: Identifier
    spec: HypothesisSpec


class HypothesisStatusChanged(Contract):
    type: Literal["hypothesis_status_changed"] = "hypothesis_status_changed"
    hypothesis_id: Identifier
    status: HypothesisStatus
    reason: NonEmptyStr
    awaiting_calculation_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def _awaiting_only_when_waiting(self) -> "HypothesisStatusChanged":
        waiting = self.status is HypothesisStatus.WAITING_FOR_JOB
        if waiting != bool(self.awaiting_calculation_ids):
            raise ValueError("awaiting_calculation_ids is required for, and only allowed on, WAITING_FOR_JOB")
        return self


class PredictionRegistered(Contract):
    type: Literal["prediction_registered"] = "prediction_registered"
    prediction_id: Identifier
    hypothesis_id: Identifier
    prediction: Prediction


class CalculationRequested(Contract):
    type: Literal["calculation_requested"] = "calculation_requested"
    calculation_id: Identifier
    hypothesis_id: Identifier
    task: ScientificTaskSpec
    task_hash: Sha256Hex
    rationale: NonEmptyStr

    @model_validator(mode="after")
    def _hash_matches_task(self) -> "CalculationRequested":
        if self.task_hash != self.task.identity_hash():
            raise ValueError("task_hash does not match the task specification")
        return self


class CalculationLinked(Contract):
    """A hypothesis reuses an existing calculation instead of requesting a duplicate."""

    type: Literal["calculation_linked"] = "calculation_linked"
    calculation_id: Identifier
    hypothesis_id: Identifier
    rationale: NonEmptyStr


class CalculationInputsExamined(Contract):
    """The exact files a calculation will run, and what the examiner found in them."""

    type: Literal["calculation_inputs_examined"] = "calculation_inputs_examined"
    calculation_id: Identifier
    examination: InputExamination
    workdir: RelativePath
    """Run directory, relative to the configured runs root."""


class CalculationStateChanged(Contract):
    type: Literal["calculation_state_changed"] = "calculation_state_changed"
    calculation_id: Identifier
    state: CalculationState
    scheduler_state: SchedulerState | None = None
    scheduler_job_id: Identifier | None = None
    reason: str | None = None
    authorization_id: Identifier | None = None
    """User intervention that authorizes this change, where one is required."""


class RunChecked(Contract):
    """The validator's health check of a finished run, with the run's output files stored."""

    type: Literal["run_checked"] = "run_checked"
    calculation_id: Identifier
    outputs: dict[FileName, ArtifactRef] = {}
    checks: tuple[QualityCheck, ...] = Field(min_length=1)
    failure_quality: EvidenceQuality
    """The best quality a result from this run can have when a check failed."""
    readings: tuple[RunReading, ...] = ()
    """Values the protocol read by itself, e.g. the final total energy."""
    limitations: tuple[NonEmptyStr, ...] = ()
    method: dict[str, JsonValue] = {}

    @model_validator(mode="after")
    def _failure_is_not_success(self) -> "RunChecked":
        if self.failure_quality not in (EvidenceQuality.INVALID, EvidenceQuality.INCONCLUSIVE):
            raise ValueError("a failed run check can make results INVALID or INCONCLUSIVE, nothing better")
        return self

    @property
    def healthy(self) -> bool:
        return all(c.passed for c in self.checks)


def _check_absolute(value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or path.as_posix() != value or "\x00" in value:
        raise ValueError(f"expected a normalized absolute path: {value!r}")
    return value


REGISTRANTS = frozenset({ActorKind.USER, ActorKind.CHIEF_SCIENTIST, ActorKind.HYPOTHESIS_AGENT})


class RunRegistered(Contract):
    """A run computed outside CEDAR, registered with its files so that readings and analyses of it can become
    evidence. Nothing runs.

    The validator stores the run's inputs and outputs by content hash, checks the run's health from its output, and
    records the settings its inputs actually used as the task. A run registered for a hypothesis belongs to that
    hypothesis; one the researcher or the Chief Scientist registers for none is shared by every agent.
    """

    type: Literal["run_registered"] = "run_registered"
    calculation_id: Identifier
    hypothesis_id: Identifier | None = None
    """None: shared by every agent of the investigation."""
    requested_by: Actor
    """Who asked for it to be registered: the researcher, the Chief Scientist or a hypothesis agent."""
    description: NonEmptyStr
    """What the run is, in the words of whoever registered it."""
    source: Annotated[str, AfterValidator(_check_absolute)]
    """The directory it was registered from."""
    task: ScientificTaskSpec
    """The settings the run's inputs used, classified as a task declares them."""
    task_hash: Sha256Hex
    code_version: NonEmptyStr
    """The version the run's output reports, or "unknown"."""
    inputs: dict[FileName, ArtifactRef] = Field(min_length=1)
    input_hash: Sha256Hex
    notes: tuple[Finding, ...] = ()
    """What the validator noticed in the inputs, e.g. keywords whose effect it does not classify."""
    outputs: dict[FileName, ArtifactRef] = Field(min_length=1)
    checks: tuple[QualityCheck, ...] = Field(min_length=1)
    failure_quality: EvidenceQuality
    readings: tuple[RunReading, ...] = ()
    limitations: tuple[NonEmptyStr, ...] = ()
    method: dict[str, JsonValue] = {}
    intervention_id: Identifier | None = None
    """The researcher's command, when it was the researcher who registered it."""

    @model_validator(mode="after")
    def _consistent(self) -> "RunRegistered":
        if self.task_hash != self.task.identity_hash():
            raise ValueError("task_hash does not match the task specification")
        if self.input_hash != scientific_input_hash(self.task.protocol, self.code_version, self.inputs):
            raise ValueError("input_hash does not match the registered inputs")
        if self.failure_quality not in (EvidenceQuality.INVALID, EvidenceQuality.INCONCLUSIVE):
            raise ValueError("a failed run check can make results INVALID or INCONCLUSIVE, nothing better")
        if self.requested_by.kind not in REGISTRANTS:
            raise ValueError("runs are registered by the researcher, the Chief Scientist or a hypothesis agent")
        if self.requested_by.kind is ActorKind.HYPOTHESIS_AGENT and self.hypothesis_id is None:
            raise ValueError("an agent registers runs for its own hypothesis")
        if (self.requested_by.kind is ActorKind.USER) != (self.intervention_id is not None):
            raise ValueError("a run the researcher registers cites their command, and only theirs does")
        return self

    @property
    def healthy(self) -> bool:
        return all(c.passed for c in self.checks)

    def as_run_check(self) -> "RunChecked":
        """The registration's health check, in the form every checked run has."""
        return RunChecked(calculation_id=self.calculation_id, outputs=self.outputs, checks=self.checks,
                          failure_quality=self.failure_quality, readings=self.readings,
                          limitations=self.limitations, method=self.method)


class WorkbenchRunRecorded(Contract):
    """A command an agent ran in its workspace, in the workbench's sandbox. Context, never evidence."""

    type: Literal["workbench_run_recorded"] = "workbench_run_recorded"
    run_id: Identifier
    hypothesis_id: Identifier
    agent_id: Identifier
    run: WorkbenchRun


class AnalysisRecorded(Contract):
    """An analysis script the runner executed for a hypothesis, and what came of it."""

    type: Literal["analysis_recorded"] = "analysis_recorded"
    analysis_id: Identifier
    hypothesis_id: Identifier
    run: AnalysisRun


class AgentsProposed(Contract):
    """Hypothesis agents the Chief Scientist proposes; the researcher reviews them before they start.

    A newer proposal replaces one still waiting.
    """

    type: Literal["agents_proposed"] = "agents_proposed"
    proposal_id: Identifier
    agents: tuple[AgentSpec, ...] = Field(min_length=1)
    rationale: NonEmptyStr
    """Why these agents, and why the other open hypotheses get none."""
    recommendation_id: Identifier | None = None
    """The Chief Scientist's recommendation this carries out, if any."""

    @model_validator(mode="after")
    def _one_agent_per_hypothesis(self) -> "AgentsProposed":
        hypotheses = [a.hypothesis_id for a in self.agents]
        if len(set(hypotheses)) != len(hypotheses):
            raise ValueError("a proposal gives each hypothesis at most one agent")
        return self


class AgentsStarted(Contract):
    """A proposal taken up: by the researcher's command, or because the review window passed without a reply."""

    type: Literal["agents_started"] = "agents_started"
    proposal_id: Identifier
    agent_ids: dict[Identifier, Identifier] = Field(min_length=1)
    """The new agents' ids, by hypothesis."""
    authorization_id: Identifier | None = None
    """The researcher's command that started them; None when the review window passed."""


class ProposalWithdrawn(Contract):
    """A proposal taken back before it started; its agents never start."""

    type: Literal["proposal_withdrawn"] = "proposal_withdrawn"
    proposal_id: Identifier
    reason: NonEmptyStr


class AgentRetired(Contract):
    """An agent will get no more work; its record and evidence remain."""

    type: Literal["agent_retired"] = "agent_retired"
    agent_id: Identifier
    reason: NonEmptyStr


AGENT_TASK_ROLES = ("hypothesis_agent", "evidence_judge", "chief_scientist", "approver")


class AgentTaskQueued(Contract):
    """Work for an agent, run by the dispatcher outside any conversation.

    A hypothesis agent's task names the agent and its hypothesis; the Evidence Judge's names
    the hypothesis to assess; the Chief Scientist's (a review of the agents' reports) and the
    approver's (decisions waiting in full automation) name neither.
    """

    type: Literal["agent_task_queued"] = "agent_task_queued"
    task_id: Identifier
    role: Literal["hypothesis_agent", "evidence_judge", "chief_scientist", "approver"]
    hypothesis_id: Identifier | None = None
    agent_id: Identifier | None = None
    instruction: NonEmptyStr
    recommendation_id: Identifier | None = None
    """The Chief Scientist's recommendation this carries out, if any."""
    continues: Identifier | None = None
    """The task this one carries on with, after the run carrying it out was cut off."""

    @model_validator(mode="after")
    def _names_what_its_role_needs(self) -> "AgentTaskQueued":
        needs = {"hypothesis_agent": (True, True), "evidence_judge": (True, False),
                 "chief_scientist": (False, False), "approver": (False, False)}[self.role]
        if (self.hypothesis_id is not None, self.agent_id is not None) != needs:
            raise ValueError(f"a {self.role} task names "
                             + {(True, True): "its agent and hypothesis", (True, False): "a hypothesis and no agent",
                                (False, False): "neither a hypothesis nor an agent"}[needs])
        return self


class AgentRunRecorded(Contract):
    """One invocation of an agent: which model served it, what it used, how it ended."""

    type: Literal["agent_run_recorded"] = "agent_run_recorded"
    session_id: Identifier
    role: Identifier
    task_id: Identifier | None = None
    """The queued task this run carried out, if any."""
    provider: NonEmptyStr
    requested_model: NonEmptyStr
    served_models: tuple[NonEmptyStr, ...] = ()
    """Every model that answered; more than one means a fallback served part of the run."""
    turns: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(ge=0)
    cache_write_tokens: int = Field(ge=0)
    outcome: NonEmptyStr
    report: str | None = None
    """The agent's final message: its stated summary and justification, not hidden reasoning."""
    structured: AgentReport | None = None
    """A hypothesis agent's report in fields, when it ended its task with one; ``report`` is then its text."""
    dropped_reasoning_blocks: int = Field(default=0, ge=0)
    """Earlier reasoning blocks the provider left out because the conversation changed under them."""
    web_searches: int = Field(default=0, ge=0)
    web_fetches: int = Field(default=0, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    """What the run cost, at the prices known for its model; None when they are not known."""
    started_version: int | None = Field(default=None, ge=0)
    """The stream version the run started from, the one its briefing showed; what was recorded later it had not
    been told of."""
    report_by_cedar: bool = False
    """The run ended without the agent's own report, so ``report`` is CEDAR's account of it, built from the record."""


class EvidenceRecorded(Contract):
    type: Literal["evidence_recorded"] = "evidence_recorded"
    evidence_id: Identifier
    evidence: EvidenceObject


class AssessmentRecorded(Contract):
    type: Literal["assessment_recorded"] = "assessment_recorded"
    hypothesis_id: Identifier
    assessment: Assessment
    completeness: Completeness
    evidence_ids: tuple[Identifier, ...] = ()
    rationale: NonEmptyStr
    limitations: tuple[NonEmptyStr, ...] = ()


EventPayload = Annotated[
    InvestigationCreated
    | InvestigationStatusChanged
    | ClosingProposed
    | NextActionsRecommended
    | SynthesisRecorded
    | ModeChanged
    | RetrievalPolicyChanged
    | BudgetChanged
    | WorkbenchPolicyChanged
    | MaterialAttached
    | ApproverDecision
    | ObservableAlignmentRecorded
    | UserInterventionRecorded
    | PlanRevised
    | RoundStarted
    | HypothesisProposed
    | HypothesisStatusChanged
    | PredictionRegistered
    | CalculationRequested
    | CalculationLinked
    | CalculationInputsExamined
    | CalculationStateChanged
    | RunChecked
    | RunRegistered
    | WorkbenchRunRecorded
    | AnalysisRecorded
    | EvidenceRecorded
    | AssessmentRecorded
    | AgentsProposed
    | AgentsStarted
    | ProposalWithdrawn
    | AgentRetired
    | AgentTaskQueued
    | AgentRunRecorded,
    Field(discriminator="type"),
]


class EventDraft(Contract):
    """An event proposed by an actor, before the ledger stamps and validates it."""

    actor: Actor
    payload: EventPayload
    plan_version: int | None = Field(default=None, ge=1)
    """Plan version the actor was working under; defaults to the current one."""
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=256)
    causation_id: Identifier | None = None


class RecordedEvent(Contract):
    schema_version: Literal[1] = 1
    event_id: Identifier
    investigation_id: Identifier
    stream_version: int = Field(ge=1)
    occurred_at: AwareDatetime
    actor: Actor
    round: int = Field(ge=0)
    plan_version: int = Field(ge=1)
    idempotency_key: str | None = None
    causation_id: Identifier | None = None
    payload: EventPayload

    @property
    def type(self) -> str:
        return self.payload.type


def content_hash(actor: Actor, payload: EventPayload) -> str:
    """Hash of who said what; used to tell a retried event from a conflicting one."""
    return sha256_hex(canonical_json({
        "actor": actor.model_dump(mode="json"),
        "payload": payload.model_dump(mode="json"),
    }))
