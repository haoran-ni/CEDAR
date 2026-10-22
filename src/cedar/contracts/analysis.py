"""Analysis scripts: Python an agent wrote, run on the login node against stored files of finished runs.

A run of a script is recorded with everything needed to repeat it: the script, every input
file by content hash, the interpreter, what it printed, the files it wrote, and the results
and checks it reported. Only the results it reported can become evidence, and they carry the
checks the script made: checks on the whole analysis apply to every result, a result's own
checks to that result alone.
"""

from enum import StrEnum

from pydantic import Field, model_validator

from cedar.contracts.base import ArtifactRef, Contract, Identifier, NonEmptyStr, Quantity, RelativePath
from cedar.contracts.evidence import QualityCheck

SCRIPT_CHECK_PREFIX = "script:"
"""Marks a quality check the analysis script made, as opposed to one the system made."""


def as_evidence_check(check: QualityCheck) -> QualityCheck:
    """A script's check as it appears among an evidence object's quality checks."""
    return check.model_copy(update={"name": SCRIPT_CHECK_PREFIX + check.name})


class AnalysisInput(Contract):
    calculation_id: Identifier
    files: dict[RelativePath, ArtifactRef] = Field(min_length=1)
    """Files of the run, staged read-only for the script as ``inputs/<calculation_id>/<path>``."""


class AnalysisData(Contract):
    """A file from outside the calculations staged for a script, such as measured data the researcher attached."""

    name: RelativePath
    """Staged read-only as ``inputs/data/<name>``."""
    artifact: ArtifactRef
    origin: NonEmptyStr
    """Where it came from: an attached material's id, or the path it was read from."""


def _unique(checks: tuple[QualityCheck, ...], where: str) -> None:
    names = [c.name for c in checks]
    if len(set(names)) != len(names):
        raise ValueError(f"check names must be unique {where}")


class AnalysisResult(Contract):
    name: Identifier
    value: Quantity
    description: str | None = None
    checks: tuple[QualityCheck, ...] = ()
    """Checks the script made that concern this result alone."""

    @model_validator(mode="after")
    def _unique_checks(self) -> "AnalysisResult":
        _unique(self.checks, f"within result {self.name}")
        return self


class AnalysisOutcome(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    """The script stopped with an error, or was stopped by a resource limit."""
    TIMED_OUT = "timed_out"


class AnalysisRun(Contract):
    purpose: NonEmptyStr
    script: ArtifactRef
    inputs: tuple[AnalysisInput, ...] = ()
    data: tuple[AnalysisData, ...] = ()
    """Files from outside the calculations it read."""
    environment: NonEmptyStr
    """Interpreter and scientific library versions the script ran with."""
    outcome: AnalysisOutcome
    exit_code: int | None = None
    duration_seconds: float = Field(ge=0)
    output: ArtifactRef | None = None
    """What the script printed, standard output and errors together."""
    products: dict[RelativePath, ArtifactRef] = {}
    """Files the script wrote in its working directory, ``results.json`` among them."""
    results: tuple[AnalysisResult, ...] = ()
    checks: tuple[QualityCheck, ...] = ()
    """Checks the script made that concern every result it reported."""
    problems: tuple[NonEmptyStr, ...] = ()
    """Why results or checks were not taken, or products not stored."""

    @model_validator(mode="after")
    def _consistent(self) -> "AnalysisRun":
        if (self.results or self.checks) and self.outcome is not AnalysisOutcome.COMPLETED:
            raise ValueError("only a completed analysis reports results and checks")
        names = [r.name for r in self.results]
        if len(set(names)) != len(names):
            raise ValueError("result names must be unique")
        _unique(self.checks, "among the analysis's checks")
        for result in self.results:
            _unique(self.checks + result.checks, f"between the analysis and result {result.name}")
        calcs = [i.calculation_id for i in self.inputs]
        if len(set(calcs)) != len(calcs):
            raise ValueError("each calculation is staged once")
        if not self.inputs and not self.data:
            raise ValueError("an analysis reads the files of a calculation or other staged data")
        names = [d.name for d in self.data]
        if len(set(names)) != len(names):
            raise ValueError("each data file is staged under its own name")
        return self

    @property
    def calculation_ids(self) -> tuple[str, ...]:
        return tuple(i.calculation_id for i in self.inputs)

    def result(self, name: str) -> AnalysisResult | None:
        return next((r for r in self.results if r.name == name), None)

    def checks_for(self, result: AnalysisResult) -> tuple[QualityCheck, ...]:
        """Every check that bears on one result: the analysis's own, then the result's."""
        return self.checks + result.checks
