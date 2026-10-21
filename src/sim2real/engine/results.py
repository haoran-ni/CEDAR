"""Turning a finished run into evidence: run health checks and verification of readings.

Agents may read raw outputs directly (``output_paths``) and submit what they read with
file-and-line citations. A reading becomes evidence only if every quote is found,
verbatim apart from whitespace, at the cited lines of the stored output, and the quoted
text contains the value and its unit. Where the protocol can read the observable itself,
the two readings must agree.
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path

from sim2real.codes.fhiaims.output import numbers_in
from sim2real.contracts.base import Actor, ActorKind, ArtifactRef, Quantity, new_id
from sim2real.contracts.events import CalculationStateChanged, EventDraft, EvidenceRecorded
from sim2real.contracts.evidence import (
    CITATIONS_VERIFIED,
    EvidenceKind,
    EvidenceObject,
    FileCitation,
    OutputCitation,
    OutputReading,
    QualityCheck,
)
from sim2real.contracts.status import CalculationState, EvidenceQuality
from sim2real.engine.jobs import InputsChanged, verify_run_directory
from sim2real.errors import RuleViolation
from sim2real.hpc.slurm import SchedulerBackend
from sim2real.protocols.base import ProtocolRegistry
from sim2real.state.ledger import RecordResult, ResearchLedger
from sim2real.store.artifacts import ArtifactStore

VALIDATOR = Actor(kind=ActorKind.VALIDATOR, id="validator")


def verify_quote(data: bytes, line_start: int, line_end: int, quote: str) -> str | None:
    """None if ``quote`` occurs at the given lines (whitespace-insensitive); otherwise the reason."""
    lines = data.decode(errors="replace").splitlines()
    if line_end > len(lines):
        return f"lines {line_start}-{line_end} are beyond the end of the file ({len(lines)} lines)"
    segment = " ".join(" ".join(line.split()) for line in lines[line_start - 1:line_end])
    if " ".join(quote.split()) not in segment:
        return f"the quote is not found at lines {line_start}-{line_end}"
    return None


def quote_states(quote: str, value: Quantity) -> bool:
    """Whether the quoted text contains the value and its unit."""
    has_number = any(math.isclose(n, value.value, rel_tol=1e-12, abs_tol=1e-12) for n in numbers_in(quote))
    has_unit = re.search(rf"(?<![\w]){re.escape(value.unit)}(?![\w])", quote) is not None
    return has_number and has_unit


@dataclass(frozen=True)
class ValidationOutcome:
    recorded: RecordResult | None
    reading_problems: tuple[str, ...] = ()
    """Why a submitted reading was not accepted; nothing is recorded in that case."""


class ResultValidator:
    def __init__(self, ledger: ResearchLedger, registry: ProtocolRegistry, artifacts: ArtifactStore,
                 backend: SchedulerBackend, runs_root: str | Path, *, actor: Actor = VALIDATOR) -> None:
        self._ledger = ledger
        self._registry = registry
        self._artifacts = artifacts
        self._backend = backend
        self._runs_root = Path(runs_root)
        self._actor = actor

    def output_paths(self, investigation_id: str, calculation_id: str) -> dict[str, Path]:
        """Raw output files of a run that exist now, for agents to read directly."""
        calc, protocol = self._calculation(investigation_id, calculation_id)
        directory = self._runs_root / calc.workdir
        names = list(protocol.output_files)
        if calc.scheduler_job_id:
            names += self._backend.log_file_names(calc.scheduler_job_id)
        return {name: directory / name for name in names if (directory / name).is_file()}

    def assess(self, investigation_id: str, calculation_id: str,
               reading: OutputReading | None = None) -> ValidationOutcome:
        calc, protocol = self._calculation(investigation_id, calculation_id)
        if calc.state not in (CalculationState.SCHEDULER_TERMINAL, CalculationState.PARSING):
            raise RuleViolation(f"calculation {calculation_id} is {calc.state}; its run has not ended")

        outputs = {name: path.read_bytes() for name, path in self.output_paths(investigation_id, calculation_id).items()}
        report = protocol.check_run(calc.task, outputs, calc.scheduler_state)
        checks = list(report.checks)
        try:
            verify_run_directory(self._runs_root / calc.workdir, calc)
            checks.append(QualityCheck(name="inputs_unchanged", passed=True, detail="run directory matches the examination"))
        except InputsChanged as exc:
            checks.append(QualityCheck(name="inputs_unchanged", passed=False, detail=str(exc)))

        value, citations_in, interpreted_by, limitations = None, [], None, list(report.limitations)
        summary = f"{calc.task.observable.name}: no reading of the observable"
        if reading is not None:
            problems = self._check_reading(reading, outputs)
            if problems:
                return ValidationOutcome(None, tuple(problems))
            value, citations_in, interpreted_by = reading.value, list(reading.citations), reading.reader
            summary = reading.summary
            limitations += reading.limitations
            if report.reading is not None:
                agree = report.reading.value.unit == value.unit and math.isclose(
                    report.reading.value.value, value.value, rel_tol=1e-9, abs_tol=1e-9)
                checks.append(QualityCheck(
                    name="agrees_with_protocol_reading", passed=agree,
                    detail=f"protocol read {report.reading.value.value} {report.reading.value.unit} at "
                           f"{report.reading.file}:{report.reading.line_start}"))
        elif report.reading is not None:
            r = report.reading
            own = FileCitation(file=r.file, line_start=r.line_start, line_end=r.line_end, quote=r.quote)
            if not self._check_reading_parts([own], r.value, outputs):
                value, citations_in = r.value, [own]
                summary = f"{calc.task.observable.name} read by {protocol.ref.name} from {r.file} line {r.line_start}"

        if calc.state is CalculationState.SCHEDULER_TERMINAL:
            self._ledger.record(investigation_id, [EventDraft(
                actor=self._actor, idempotency_key=f"{calculation_id}:parsing",
                payload=CalculationStateChanged(calculation_id=calculation_id, state=CalculationState.PARSING))])

        stored = {name: self._artifacts.put_bytes(data, label=name) for name, data in sorted(outputs.items())}
        citations = tuple(OutputCitation(artifact=stored[c.file], line_start=c.line_start, line_end=c.line_end,
                                         quote=c.quote) for c in citations_in)
        if citations:
            checks.append(QualityCheck(name=CITATIONS_VERIFIED, passed=True,
                                       detail=f"{len(citations)} citation(s) found verbatim in the stored output"))
        health_failed = any(not c.passed for c in checks if c.name != "agrees_with_protocol_reading")
        if health_failed:
            quality = report.failure_quality
        elif value is None or not all(c.passed for c in checks):
            quality = EvidenceQuality.INCONCLUSIVE
        else:
            quality = EvidenceQuality.VALIDATED

        sources: tuple[ArtifactRef, ...] = (*stored.values(), *calc.examination.files.values())
        evidence = EvidenceObject(
            kind=EvidenceKind.DIRECT_COMPUTATION, summary=summary, observable=calc.task.observable, value=value,
            calculation_id=calculation_id, method=report.method, quality=quality, quality_checks=tuple(checks),
            limitations=tuple(limitations), sources=sources, citations=citations, interpreted_by=interpreted_by,
        )
        recorded = self._ledger.record(investigation_id, [
            EventDraft(actor=self._actor, idempotency_key=f"{calculation_id}:evidence",
                       payload=EvidenceRecorded(evidence_id=new_id("ev"), evidence=evidence)),
            EventDraft(actor=self._actor, idempotency_key=f"{calculation_id}:quality_assessed",
                       payload=CalculationStateChanged(calculation_id=calculation_id,
                                                       state=CalculationState.QUALITY_ASSESSED)),
        ])
        return ValidationOutcome(recorded)

    def _calculation(self, investigation_id: str, calculation_id: str):
        state = self._ledger.load(investigation_id)
        calc = state.calculations.get(calculation_id)
        if calc is None or calc.examination is None:
            raise RuleViolation(f"calculation {calculation_id} has no examined inputs")
        protocol = self._registry.get(calc.task.protocol)
        if protocol is None:
            raise RuleViolation(f"protocol {calc.task.protocol.name} {calc.task.protocol.version} is not available")
        return calc, protocol

    @classmethod
    def _check_reading(cls, reading: OutputReading, outputs: dict[str, bytes]) -> list[str]:
        return cls._check_reading_parts(reading.citations, reading.value, outputs)

    @staticmethod
    def _check_reading_parts(citations, value: Quantity, outputs: dict[str, bytes]) -> list[str]:
        problems = []
        for c in citations:
            data = outputs.get(c.file)
            if data is None:
                problems.append(f"{c.file} is not an output of this run")
                continue
            if (problem := verify_quote(data, c.line_start, c.line_end, c.quote)) is not None:
                problems.append(f"{c.file}: {problem}")
        if not any(quote_states(c.quote, value) for c in citations):
            problems.append(f"no cited quote states {value.value} {value.unit}")
        return problems
