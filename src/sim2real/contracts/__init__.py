"""Typed contracts shared by agents, the computing engine and the research record."""

from sim2real.contracts.base import (
    Actor,
    ActorKind,
    ArtifactRef,
    Contract,
    ExternalSource,
    Observable,
    Quantity,
    Uncertainty,
    UncertaintyKind,
)
from sim2real.contracts.events import (
    AssessmentRecorded,
    CalculationLinked,
    CalculationRequested,
    CalculationStateChanged,
    EventDraft,
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
    UserInterventionRecorded,
)
from sim2real.contracts.evidence import EvidenceKind, EvidenceObject, QualityCheck
from sim2real.contracts.hypothesis import HypothesisSpec, Prediction, Quantifiability
from sim2real.contracts.investigation import (
    AlignmentStatus,
    Discrepancy,
    ExperimentProvenance,
    InvestigationConstraints,
    InvestigationSpec,
    ObservableAlignment,
    RetrievalPolicy,
    SimulationProvenance,
    discrepancy,
)
from sim2real.contracts.plan import Authorization, PermissionAction, Priority, ResearchPlan
from sim2real.contracts.status import (
    Assessment,
    CalculationState,
    Completeness,
    EvidenceQuality,
    HypothesisStatus,
    InvestigationStatus,
    SchedulerState,
)
from sim2real.contracts.task import (
    ComparisonReport,
    DifferenceKind,
    FidelityTier,
    ProtocolRef,
    ResourceBudget,
    ScientificTaskSpec,
    compare_tasks,
)

__all__ = [
    "Actor", "ActorKind", "AlignmentStatus", "ArtifactRef", "Assessment", "AssessmentRecorded",
    "Authorization", "CalculationLinked", "CalculationRequested", "CalculationState",
    "CalculationStateChanged", "ComparisonReport", "Completeness", "Contract", "DifferenceKind",
    "Discrepancy", "EventDraft", "EvidenceKind", "EvidenceObject", "EvidenceQuality",
    "EvidenceRecorded", "ExperimentProvenance", "ExternalSource", "FidelityTier",
    "HypothesisProposed", "HypothesisSpec", "HypothesisStatus", "HypothesisStatusChanged",
    "InvestigationConstraints", "InvestigationCreated", "InvestigationSpec", "InvestigationStatus",
    "InvestigationStatusChanged", "Observable", "ObservableAlignment", "ObservableAlignmentRecorded",
    "PermissionAction", "PlanRevised", "Prediction", "PredictionRegistered", "Priority",
    "ProtocolRef", "QualityCheck", "Quantifiability", "Quantity", "RecordedEvent", "ResearchPlan",
    "ResourceBudget", "RetrievalPolicy", "RoundStarted", "SchedulerState", "ScientificTaskSpec",
    "SimulationProvenance", "Uncertainty", "UncertaintyKind", "UserInterventionRecorded",
    "compare_tasks", "discrepancy",
]
