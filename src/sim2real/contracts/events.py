"""Event payloads and the envelope under which they are recorded.

The event log is the source of truth: research state is whatever replaying these events
produces. Payloads only add information; nothing edits or deletes an earlier record.
"""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from sim2real.contracts.agents import AgentSpec
from sim2real.contracts.analysis import AnalysisRun
from sim2real.contracts.base import (
    Actor,
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
from sim2real.contracts.evidence import EvidenceObject, QualityCheck, RunReading
from sim2real.contracts.examination import InputExamination
from sim2real.contracts.hypothesis import HypothesisSpec, Prediction
from sim2real.contracts.investigation import InvestigationSpec, ObservableAlignment
from sim2real.contracts.plan import Authorization, ResearchPlan
from sim2real.contracts.status import (
    Assessment,
    CalculationState,
    Completeness,
    EvidenceQuality,
    HypothesisStatus,
    InvestigationStatus,
    SchedulerState,
)
from sim2real.contracts.task import ScientificTaskSpec


class InvestigationCreated(Contract):
    type: Literal["investigation_created"] = "investigation_created"
    spec: InvestigationSpec
    plan: ResearchPlan


class InvestigationStatusChanged(Contract):
    type: Literal["investigation_status_changed"] = "investigation_status_changed"
    status: InvestigationStatus
    reason: NonEmptyStr


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


AGENT_TASK_ROLES = ("hypothesis_agent", "evidence_judge", "chief_scientist")


class AgentTaskQueued(Contract):
    """Work for an agent, run by the dispatcher outside any conversation.

    A hypothesis agent's task names the agent and its hypothesis; the Evidence Judge's names
    the hypothesis to assess; the Chief Scientist's (a review of the agents' reports) names
    neither.
    """

    type: Literal["agent_task_queued"] = "agent_task_queued"
    task_id: Identifier
    role: Literal["hypothesis_agent", "evidence_judge", "chief_scientist"]
    hypothesis_id: Identifier | None = None
    agent_id: Identifier | None = None
    instruction: NonEmptyStr

    @model_validator(mode="after")
    def _names_what_its_role_needs(self) -> "AgentTaskQueued":
        needs = {"hypothesis_agent": (True, True), "evidence_judge": (True, False),
                 "chief_scientist": (False, False)}[self.role]
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
    dropped_reasoning_blocks: int = Field(default=0, ge=0)
    """Earlier reasoning blocks the provider left out because the conversation changed under them."""


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
