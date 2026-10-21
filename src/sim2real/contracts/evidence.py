"""Evidence objects: results linked to their raw sources, method and quality status."""

from enum import StrEnum

from pydantic import Field, JsonValue, model_validator

from sim2real.contracts.base import (
    ArtifactRef,
    Contract,
    ExternalSource,
    Identifier,
    NonEmptyStr,
    Observable,
    Quantity,
    SourceRef,
)
from sim2real.contracts.status import EvidenceQuality


class EvidenceKind(StrEnum):
    DIRECT_COMPUTATION = "direct_computation"
    """Parsed from a calculation run within this investigation."""
    ARCHIVED_COMPUTATION = "archived_computation"
    """Prerecorded output, disclosed as such and never presented as newly computed."""
    PUBLISHED_EXPERIMENT = "published_experiment"
    INFERENCE = "inference"
    """Derived from other information under stated assumptions."""
    INDIRECT_CONSTRAINT = "indirect_constraint"


COMPUTED_KINDS = frozenset({EvidenceKind.DIRECT_COMPUTATION, EvidenceKind.ARCHIVED_COMPUTATION})


class QualityCheck(Contract):
    name: Identifier
    passed: bool
    detail: str | None = None


class EvidenceObject(Contract):
    kind: EvidenceKind
    summary: NonEmptyStr
    observable: Observable
    value: Quantity | None = None
    calculation_id: Identifier | None = None
    method: dict[str, JsonValue] = {}
    conditions: dict[str, JsonValue] = {}
    quality: EvidenceQuality
    quality_checks: tuple[QualityCheck, ...] = ()
    assumptions: tuple[NonEmptyStr, ...] = ()
    limitations: tuple[NonEmptyStr, ...] = ()
    non_comparability: tuple[NonEmptyStr, ...] = ()
    """Conditions under which this evidence must not be compared with other results."""
    sources: tuple[SourceRef, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> "EvidenceObject":
        if (self.kind is EvidenceKind.DIRECT_COMPUTATION) != (self.calculation_id is not None):
            raise ValueError("calculation_id is required for, and only allowed on, direct_computation evidence")
        if self.kind in COMPUTED_KINDS and not any(isinstance(s, ArtifactRef) for s in self.sources):
            raise ValueError("computed evidence must link at least one raw output artifact")
        if self.kind is EvidenceKind.PUBLISHED_EXPERIMENT and not any(
            isinstance(s, ExternalSource) for s in self.sources
        ):
            raise ValueError("published experimental evidence must cite an external source")
        if self.kind is EvidenceKind.INFERENCE and not self.assumptions:
            raise ValueError("inference must state its assumptions")
        if self.quality is EvidenceQuality.VALIDATED:
            if not self.quality_checks:
                raise ValueError("VALIDATED evidence needs at least one recorded quality check")
            failed = [c.name for c in self.quality_checks if not c.passed]
            if failed:
                raise ValueError(f"VALIDATED evidence cannot have failed quality checks: {failed}")
        return self
