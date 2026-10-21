"""Turning finished runs into evidence: run checks, verified readings, recorded analysis results.

When a job ends, the validator checks the run's health and stores its output files
(``check_run``). After that, agents take values from the run in two ways:

- a reading (``record_reading``): a value printed in an output, cited by file and lines;
  it counts only if every quote is found, verbatim apart from whitespace, at the cited
  lines of the stored output, and a quote contains the value and its unit;
- an analysis result (``record_analysis_result``): a value an analysis script computed
  and reported, taken from the analysis record, never from the agent's own words, together
  with every check the script reported for it.

Either way the evidence is VALIDATED only if every run it rests on ended COMPLETED and
passed its checks; otherwise it gets the failed run's quality. An analysis result whose
script reported a failed check is INCONCLUSIVE at best.
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path

from sim2real.codes.fhiaims.output import numbers_in
from sim2real.contracts.analysis import AnalysisOutcome, as_evidence_check
from sim2real.contracts.base import Actor, ActorKind, Observable, Quantity, new_id
from sim2real.contracts.events import CalculationStateChanged, EventDraft, EvidenceRecorded, RunChecked
from sim2real.contracts.evidence import (
    CITATIONS_VERIFIED,
    EvidenceKind,
    EvidenceObject,
    OutputCitation,
    OutputReading,
    QualityCheck,
    RunReading,
)
from sim2real.contracts.status import CalculationState, EvidenceQuality, SchedulerState
from sim2real.engine.jobs import InputsChanged, verify_run_directory
from sim2real.errors import RuleViolation
from sim2real.hpc.slurm import SchedulerBackend
from sim2real.protocols.base import ProtocolRegistry
from sim2real.state.ledger import RecordResult, ResearchLedger
from sim2real.state.projection import CalculationRecord
from sim2real.store.artifacts import ArtifactStore

ANALYSIS_LIMITATION = ("Computed by an analysis script the agent wrote; the script has not been reviewed by an "
                       "independent agent or person.")

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
    problems: tuple[str, ...] = ()
    """Why a reading or result was not accepted; nothing is recorded in that case."""


class ResultValidator:
    def __init__(self, ledger: ResearchLedger, registry: ProtocolRegistry, artifacts: ArtifactStore,
                 backend: SchedulerBackend, runs_root: str | Path, *, actor: Actor = VALIDATOR) -> None:
        self._ledger = ledger
        self._registry = registry
        self._artifacts = artifacts
        self._backend = backend
        self._runs_root = Path(runs_root)
        self._actor = actor

    def run_directory(self, investigation_id: str, calculation_id: str) -> Path:
        calc, _ = self._calculation(investigation_id, calculation_id)
        return self._runs_root / calc.workdir

    def output_paths(self, investigation_id: str, calculation_id: str) -> dict[str, Path]:
        """Output files of a run the protocol knows about, plus the scheduler's logs, that exist now."""
        calc, protocol = self._calculation(investigation_id, calculation_id)
        directory = self._runs_root / calc.workdir
        names = list(protocol.output_files)
        if calc.scheduler_job_id:
            names += self._backend.log_file_names(calc.scheduler_job_id)
        return {name: directory / name for name in names if (directory / name).is_file()}

    # -- run checks ---------------------------------------------------------------------

    def check_run(self, investigation_id: str, calculation_id: str) -> RecordResult:
        """Check a finished run's health and store its output files. Safe to call again."""
        calc, protocol = self._calculation(investigation_id, calculation_id)
        if calc.run_check is not None:
            return RecordResult((), self._ledger.load(investigation_id), deduplicated=True)
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
        stored = {name: self._artifacts.put_bytes(data, label=name) for name, data in sorted(outputs.items())}
        readings = tuple(
            RunReading(name=r.name, value=r.value, file=r.file, line_start=r.line_start, line_end=r.line_end,
                       quote=r.quote)
            for r in report.readings
            if r.file in outputs and verify_quote(outputs[r.file], r.line_start, r.line_end, r.quote) is None)

        drafts = []
        if calc.state is CalculationState.SCHEDULER_TERMINAL:
            drafts.append(EventDraft(actor=self._actor, idempotency_key=f"{calculation_id}:parsing",
                                     payload=CalculationStateChanged(calculation_id=calculation_id,
                                                                     state=CalculationState.PARSING)))
        drafts += [
            EventDraft(actor=self._actor, idempotency_key=f"{calculation_id}:run_checked", payload=RunChecked(
                calculation_id=calculation_id, outputs=stored, checks=tuple(checks),
                failure_quality=report.failure_quality, readings=readings, limitations=report.limitations,
                method=report.method)),
            EventDraft(actor=self._actor, idempotency_key=f"{calculation_id}:quality_assessed",
                       payload=CalculationStateChanged(calculation_id=calculation_id,
                                                       state=CalculationState.QUALITY_ASSESSED)),
        ]
        return self._ledger.record(investigation_id, drafts)

    # -- readings -----------------------------------------------------------------------

    def record_reading(self, investigation_id: str, calculation_id: str, reading: OutputReading,
                       observable: Observable) -> ValidationOutcome:
        calc, _ = self._calculation(investigation_id, calculation_id)
        check = self._checked(calc)
        outputs = {name: self._artifacts.read(ref) for name, ref in check.outputs.items()}
        problems = self._check_reading_parts(reading.citations, reading.value, outputs)
        if problems:
            return ValidationOutcome(None, tuple(problems))
        citations = tuple(OutputCitation(artifact=check.outputs[c.file], line_start=c.line_start,
                                         line_end=c.line_end, quote=c.quote) for c in reading.citations)
        checks = (self._health_check(calc),
                  QualityCheck(name=CITATIONS_VERIFIED, passed=True,
                               detail=f"{len(citations)} citation(s) found verbatim in the stored output"))
        evidence = EvidenceObject(
            kind=EvidenceKind.DIRECT_COMPUTATION, summary=reading.summary, observable=observable, value=reading.value,
            calculation_id=calculation_id, method=check.method, quality=self._quality([calc]),
            quality_checks=checks, limitations=(*check.limitations, *reading.limitations),
            sources=(*check.outputs.values(), *calc.examination.files.values()), citations=citations,
            interpreted_by=reading.reader)
        return ValidationOutcome(self._record(investigation_id, evidence))

    # -- analysis results ---------------------------------------------------------------

    def record_analysis_result(self, investigation_id: str, analysis_id: str, result_name: str, *,
                               observable: Observable, summary: str, limitations: tuple[str, ...] = (),
                               reader: Actor) -> ValidationOutcome:
        state = self._ledger.load(investigation_id)
        analysis = state.analyses.get(analysis_id)
        if analysis is None:
            raise RuleViolation(f"unknown analysis {analysis_id}")
        run = analysis.run
        if run.outcome is not AnalysisOutcome.COMPLETED:
            return ValidationOutcome(None, (f"analysis {analysis_id} did not complete ({run.outcome.value})",))
        result = run.result(result_name)
        if result is None:
            names = ", ".join(r.name for r in run.results) or "none"
            return ValidationOutcome(None, (f"analysis {analysis_id} reported no result {result_name!r} (it reported: "
                                            f"{names})",))
        if observable.unit is not None and observable.unit != result.value.unit:
            return ValidationOutcome(None, (f"the observable's unit {observable.unit!r} differs from the result's "
                                            f"unit {result.value.unit!r}",))
        calcs = [state.calculations[c] for c in run.calculation_ids]
        checks = [QualityCheck(name="analysis_completed", passed=True,
                               detail=f"exit code {run.exit_code}; {len(run.results)} result(s) reported")]
        checks += [self._health_check(calc) for calc in calcs]
        script_checks = [as_evidence_check(c) for c in run.checks_for(result)]
        checks += script_checks
        quality = self._quality(calcs)
        if quality is EvidenceQuality.VALIDATED and not all(c.passed for c in script_checks):
            quality = EvidenceQuality.INCONCLUSIVE
        sources = (run.script, *(ref for staged in run.inputs for ref in staged.files.values()),
                   *([run.output] if run.output else []), *run.products.values())
        evidence = EvidenceObject(
            kind=EvidenceKind.ANALYSIS, summary=summary, observable=observable, value=result.value,
            calculation_ids=run.calculation_ids, analysis_id=analysis_id, analysis_result=result.name,
            method={"analysis": analysis_id, "result": result.name, "purpose": run.purpose,
                    "environment": run.environment, "script_sha256": run.script.sha256},
            quality=quality, quality_checks=tuple(checks),
            limitations=(ANALYSIS_LIMITATION, *limitations,
                         *dict.fromkeys(text for calc in calcs for text in calc.run_check.limitations)),
            sources=sources, interpreted_by=reader)
        return ValidationOutcome(self._record(investigation_id, evidence))

    # -- shared -------------------------------------------------------------------------

    def _record(self, investigation_id: str, evidence: EvidenceObject) -> RecordResult:
        return self._ledger.record(investigation_id, [EventDraft(
            actor=self._actor, payload=EvidenceRecorded(evidence_id=new_id("ev"), evidence=evidence))])

    @staticmethod
    def _checked(calc: CalculationRecord):
        if calc.run_check is None:
            raise RuleViolation(f"the run of {calc.calculation_id} has not been checked yet ({calc.state.value})")
        return calc.run_check

    @classmethod
    def _health_check(cls, calc: CalculationRecord) -> QualityCheck:
        check = cls._checked(calc)
        failed = [c.name for c in check.checks if not c.passed]
        return QualityCheck(name=f"run_healthy:{calc.calculation_id}", passed=not failed,
                            detail=f"failed: {', '.join(failed)}" if failed else
                            f"{len(check.checks)} run checks passed")

    @classmethod
    def _quality(cls, calcs: list[CalculationRecord]) -> EvidenceQuality:
        failures = [check.failure_quality for c in calcs
                    if not (check := cls._checked(c)).healthy or c.scheduler_state is not SchedulerState.COMPLETED]
        if not failures:
            return EvidenceQuality.VALIDATED
        return EvidenceQuality.INVALID if EvidenceQuality.INVALID in failures else EvidenceQuality.INCONCLUSIVE

    def _calculation(self, investigation_id: str, calculation_id: str):
        state = self._ledger.load(investigation_id)
        calc = state.calculations.get(calculation_id)
        if calc is None or calc.examination is None:
            raise RuleViolation(f"calculation {calculation_id} has no examined inputs")
        protocol = self._registry.get(calc.task.protocol)
        if protocol is None:
            raise RuleViolation(f"protocol {calc.task.protocol.name} {calc.task.protocol.version} is not available")
        return calc, protocol

    @staticmethod
    def _check_reading_parts(citations, value: Quantity, outputs: dict[str, bytes]) -> list[str]:
        problems = []
        for c in citations:
            data = outputs.get(c.file)
            if data is None:
                problems.append(f"{c.file} is not a stored output of this run (stored: {', '.join(outputs) or 'none'})")
                continue
            if (problem := verify_quote(data, c.line_start, c.line_end, c.quote)) is not None:
                problems.append(f"{c.file}: {problem}")
        if not any(quote_states(c.quote, value) for c in citations):
            problems.append(f"no cited quote states {value.value} {value.unit}")
        return problems
