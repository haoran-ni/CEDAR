"""Evidence objects: results linked to their raw sources, method and quality status."""

from enum import StrEnum

from pydantic import Field, JsonValue, model_validator

from sim2real.contracts.base import (
    Actor,
    ArtifactRef,
    Contract,
    ExternalSource,
    FileName,
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

CITATIONS_VERIFIED = "citations_verified"
"""Quality check that every citation was found, verbatim, in the cited stored output."""


class QualityCheck(Contract):
    name: Identifier
    passed: bool
    detail: str | None = None


class OutputCitation(Contract):
    """Lines of a stored raw output that a reported value was read from."""

    artifact: ArtifactRef
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    quote: NonEmptyStr
    """Text of the cited lines, as the reader saw it."""

    @model_validator(mode="after")
    def _ordered(self) -> "OutputCitation":
        if self.line_end < self.line_start:
            raise ValueError("line_end must not precede line_start")
        return self


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
    citations: tuple[OutputCitation, ...] = ()
    """Where in the raw outputs each reported value was read."""
    interpreted_by: Actor | None = None
    """The agent whose reading of the raw output this is; None if read by protocol tools alone."""

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
        if self.kind in COMPUTED_KINDS and self.value is not None and not self.citations:
            raise ValueError("a computed value must cite the output lines it was read from")
        source_hashes = {s.sha256 for s in self.sources if isinstance(s, ArtifactRef)}
        if any(c.artifact.sha256 not in source_hashes for c in self.citations):
            raise ValueError("every cited output must also be listed among the sources")
        if self.kind is EvidenceKind.INFERENCE and not self.assumptions:
            raise ValueError("inference must state its assumptions")
        if self.quality is EvidenceQuality.VALIDATED:
            if not self.quality_checks:
                raise ValueError("VALIDATED evidence needs at least one recorded quality check")
            failed = [c.name for c in self.quality_checks if not c.passed]
            if failed:
                raise ValueError(f"VALIDATED evidence cannot have failed quality checks: {failed}")
            if self.citations and CITATIONS_VERIFIED not in {c.name for c in self.quality_checks}:
                raise ValueError(f"VALIDATED evidence with citations needs a passed {CITATIONS_VERIFIED} check")
        return self


class FileCitation(Contract):
    """Lines of a run's output file that an agent read, named by file within the run directory."""

    file: FileName
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    quote: NonEmptyStr

    @model_validator(mode="after")
    def _ordered(self) -> "FileCitation":
        if self.line_end < self.line_start:
            raise ValueError("line_end must not precede line_start")
        return self


class OutputReading(Contract):
    """An agent's reading of a value from raw output, submitted for verification.

    It becomes evidence only after the validator has found every quote, verbatim, at the
    cited lines of the stored output, and the quoted text contains the value and unit.
    """

    reader: Actor
    value: Quantity
    citations: tuple[FileCitation, ...] = Field(min_length=1)
    summary: NonEmptyStr
    limitations: tuple[NonEmptyStr, ...] = ()
