"""The second layer: deterministic examination of inputs before anything runs.

Agents may write input files and job scripts themselves, or let a protocol generate
them. Either way the protocol's examiner checks the exact files that will run against
the declared task and the protocol's policy, and records what it found.
"""

from collections.abc import Iterable, Mapping
from enum import StrEnum

from pydantic import Field, model_validator

from cedar.contracts.base import (
    ArtifactRef,
    Contract,
    FileName,
    Identifier,
    NonEmptyStr,
    Sha256Hex,
    canonical_json,
    sha256_hex,
)
from cedar.contracts.task import ProtocolRef


class FindingSeverity(StrEnum):
    ERROR = "error"
    """Contradicts the declared task or violates policy; the inputs are rejected."""
    REVIEW = "review"
    """The examiner cannot verify it; the inputs need explicit user approval."""
    WARNING = "warning"
    """Recorded for the reader; does not block."""
    INFO = "info"


class Finding(Contract):
    severity: FindingSeverity
    code: Identifier
    message: NonEmptyStr
    file: FileName | None = None
    line: int | None = Field(default=None, ge=1)


class ExaminationVerdict(StrEnum):
    APPROVED = "APPROVED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    REJECTED = "REJECTED"


def verdict_for(findings: Iterable[Finding]) -> ExaminationVerdict:
    severities = {f.severity for f in findings}
    if FindingSeverity.ERROR in severities:
        return ExaminationVerdict.REJECTED
    if FindingSeverity.REVIEW in severities:
        return ExaminationVerdict.NEEDS_REVIEW
    return ExaminationVerdict.APPROVED


def scientific_input_hash(
    protocol: ProtocolRef,
    code_version: str,
    files: Mapping[str, ArtifactRef],
    execution_files: Iterable[str] = (),
) -> str:
    """Identity of what is computed: protocol, code version and scientific input files.

    Execution files (job scripts) are examined but excluded, so the same calculation
    with a different walltime or partition is still recognized as a duplicate.
    """
    excluded = set(execution_files)
    return sha256_hex(canonical_json({
        "protocol": protocol.model_dump(mode="json"),
        "code_version": code_version,
        "files": {name: ref.sha256 for name, ref in files.items() if name not in excluded},
    }))


class InputExamination(Contract):
    protocol: ProtocolRef
    code_version: NonEmptyStr
    files: dict[FileName, ArtifactRef] = Field(min_length=1)
    """Exactly the files that will run, as stored in the artifact store."""
    execution_files: tuple[FileName, ...] = ()
    input_hash: Sha256Hex
    findings: tuple[Finding, ...] = ()
    verdict: ExaminationVerdict
    requested_core_hours: float | None = Field(default=None, ge=0)
    """What the job script asks of the scheduler (tasks times walltime), if the protocol can read it."""

    @model_validator(mode="after")
    def _consistent(self) -> "InputExamination":
        unknown = set(self.execution_files) - set(self.files)
        if unknown:
            raise ValueError(f"execution files not among the examined files: {sorted(unknown)}")
        expected = scientific_input_hash(self.protocol, self.code_version, self.files, self.execution_files)
        if self.input_hash != expected:
            raise ValueError("input_hash does not match the examined files")
        if self.verdict is not verdict_for(self.findings):
            raise ValueError(f"verdict {self.verdict} does not follow from the findings")
        return self
