"""Event payloads and the envelope under which they are recorded.

The event log is the source of truth: research state is whatever replaying these events
produces. Payloads only add information; nothing edits or deletes an earlier record.
"""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from sim2real.contracts.base import (
    Actor,
    Contract,
    Identifier,
    NonEmptyStr,
    RelativePath,
    Sha256Hex,
    canonical_json,
    sha256_hex,
)
from sim2real.contracts.evidence import EvidenceObject
from sim2real.contracts.examination import InputExamination
from sim2real.contracts.hypothesis import HypothesisSpec, Prediction
from sim2real.contracts.investigation import InvestigationSpec, ObservableAlignment
from sim2real.contracts.plan import Authorization, ResearchPlan
from sim2real.contracts.status import (
    Assessment,
    CalculationState,
    Completeness,
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


AGENT_TASK_ROLES = ("hypothesis_agent", "evidence_judge")


class AgentTaskQueued(Contract):
    """Work for an agent, run by the dispatcher outside any conversation."""

    type: Literal["agent_task_queued"] = "agent_task_queued"
    task_id: Identifier
    role: Literal["hypothesis_agent", "evidence_judge"]
    hypothesis_id: Identifier
    instruction: NonEmptyStr


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
    | EvidenceRecorded
    | AssessmentRecorded
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
