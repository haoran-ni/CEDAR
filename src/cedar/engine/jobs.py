"""Submitting examined inputs and following jobs through the scheduler adapter.

Submission is journaled: SUBMISSION_PENDING is recorded before the scheduler is called,
and a calculation found in that state is reconciled by job name before anything is
submitted again, so a crash between the two never produces a duplicate job.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from cedar.contracts.base import Actor, ActorKind
from cedar.contracts.events import CalculationStateChanged, EventDraft, RecordedEvent
from cedar.contracts.plan import PermissionAction
from cedar.contracts.status import SCHEDULER_TERMINAL_STATES, CalculationState, SchedulerState
from cedar.errors import RuleViolation
from cedar.hpc.slurm import SchedulerBackend, SchedulerError, default_lookback
from cedar.state.ledger import ResearchLedger
from cedar.state.projection import CalculationRecord, authorization_for

JOB_MANAGER = Actor(kind=ActorKind.JOB_MANAGER, id="job-manager")
JOB_MONITOR = Actor(kind=ActorKind.JOB_MONITOR, id="job-monitor")

_C = CalculationState
_AT_SCHEDULER = frozenset({_C.SUBMITTED, _C.QUEUED, _C.RUNNING})


def job_name_for(calculation_id: str) -> str:
    return f"cedar-{calculation_id}"


class InputsChanged(RuntimeError):
    """A file in the run directory no longer matches what was examined."""


def verify_run_directory(directory: Path, calc: CalculationRecord) -> None:
    for name, ref in calc.examination.files.items():
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != ref.sha256:
            raise InputsChanged(f"{path} differs from the examined {name}")


class JobManager:
    def __init__(self, ledger: ResearchLedger, backend: SchedulerBackend, runs_root: str | Path, *,
                 actor: Actor = JOB_MANAGER) -> None:
        self._ledger = ledger
        self._backend = backend
        self._runs_root = Path(runs_root)
        self._actor = actor

    def submit(self, investigation_id: str, calculation_id: str) -> str:
        """Submit examined inputs and return the scheduler job id. Safe to call again."""
        state = self._ledger.load(investigation_id)
        calc = state.calculations.get(calculation_id)
        if calc is None:
            raise RuleViolation(f"unknown calculation {calculation_id}")
        if calc.scheduler_job_id is not None:
            return calc.scheduler_job_id
        if calc.state not in (_C.VALIDATED_INPUT, _C.SUBMISSION_PENDING):
            raise RuleViolation(f"cannot submit a calculation that is {calc.state}")
        authorization = None
        if state.spec.constraints.submit_requires_approval:
            authorization = authorization_for(state, PermissionAction.SUBMIT, calculation_id)
            if authorization is None:
                raise RuleViolation(f"submitting {calculation_id} needs a user authorization naming it")
        directory = self._runs_root / calc.workdir
        verify_run_directory(directory, calc)
        if not calc.examination.execution_files:
            raise RuleViolation(f"the examined inputs of {calculation_id} contain no job script")

        name = job_name_for(calculation_id)
        if calc.state is _C.SUBMISSION_PENDING:
            existing = self._backend.find_by_name(name, default_lookback())
            if len(existing) > 1:
                raise SchedulerError(f"several scheduler jobs are named {name}; resolve this by hand")
            if existing:
                return self._record_submitted(investigation_id, calculation_id, existing[0].job_id, authorization)
        else:
            self._ledger.record(investigation_id, [self._draft(calculation_id, _C.SUBMISSION_PENDING,
                                                               key=f"{calculation_id}:submission_pending")])
        script = directory / calc.examination.execution_files[0]
        job_id = self._backend.submit(script, directory, name)
        return self._record_submitted(investigation_id, calculation_id, job_id, authorization)

    def reconcile(self, investigation_id: str) -> dict[str, str | None]:
        """Settle calculations left in SUBMISSION_PENDING: {calculation id: job id or None if none exists}."""
        state = self._ledger.load(investigation_id)
        settled: dict[str, str | None] = {}
        for calc in state.calculations.values():
            if calc.state is not _C.SUBMISSION_PENDING:
                continue
            found = self._backend.find_by_name(job_name_for(calc.calculation_id), default_lookback())
            if len(found) == 1:
                authorization = authorization_for(state, PermissionAction.SUBMIT, calc.calculation_id)
                settled[calc.calculation_id] = self._record_submitted(
                    investigation_id, calc.calculation_id, found[0].job_id, authorization)
            else:
                settled[calc.calculation_id] = None
        return settled

    def cancel(self, investigation_id: str, calculation_id: str, reason: str) -> None:
        state = self._ledger.load(investigation_id)
        calc = state.calculations.get(calculation_id)
        if calc is None:
            raise RuleViolation(f"unknown calculation {calculation_id}")
        authorization = None
        if calc.state in _AT_SCHEDULER or calc.state is _C.SUBMISSION_PENDING:
            authorization = authorization_for(state, PermissionAction.CANCEL, calculation_id)
            if authorization is None:
                raise RuleViolation(f"cancelling {calculation_id} needs a user authorization naming it")
            if calc.scheduler_job_id is not None:
                self._backend.cancel(calc.scheduler_job_id)
        self._ledger.record(investigation_id, [self._draft(calculation_id, _C.CANCELLED, reason=reason,
                                                           authorization_id=authorization)])

    def _record_submitted(self, investigation_id: str, calculation_id: str, job_id: str,
                          authorization: str | None) -> str:
        self._ledger.record(investigation_id, [self._draft(
            calculation_id, _C.SUBMITTED, scheduler_job_id=job_id, authorization_id=authorization,
            key=f"{calculation_id}:submitted")])
        return job_id

    def _draft(self, calculation_id: str, state: CalculationState, *, key: str | None = None, **kwargs) -> EventDraft:
        return EventDraft(actor=self._actor, idempotency_key=key, payload=CalculationStateChanged(
            calculation_id=calculation_id, state=state, **kwargs))


@dataclass
class PollResult:
    recorded: list[RecordedEvent] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    """Calculations whose job the scheduler did not report clearly; left unchanged."""


class JobMonitor:
    def __init__(self, ledger: ResearchLedger, backend: SchedulerBackend, *, actor: Actor = JOB_MONITOR) -> None:
        self._ledger = ledger
        self._backend = backend
        self._actor = actor

    def poll(self, investigation_id: str) -> PollResult:
        """One scheduler query for all live jobs; records only real state changes."""
        state = self._ledger.load(investigation_id)
        live = [c for c in state.calculations.values() if c.state in _AT_SCHEDULER and c.scheduler_job_id]
        observations = self._backend.query([c.scheduler_job_id for c in live])
        result = PollResult()
        for calc in live:
            seen = observations.get(calc.scheduler_job_id)
            if seen is None or seen.state is SchedulerState.UNKNOWN:
                result.unknown.append(calc.calculation_id)
                continue
            if seen.state is SchedulerState.PENDING:
                target = _C.QUEUED
            elif seen.state is SchedulerState.RUNNING:
                target = _C.RUNNING
            elif seen.state in SCHEDULER_TERMINAL_STATES:
                target = _C.SCHEDULER_TERMINAL
            else:
                result.unknown.append(calc.calculation_id)
                continue
            if target is calc.state:
                continue
            draft = EventDraft(
                actor=self._actor,
                idempotency_key=f"{calc.calculation_id}:{len(calc.history)}:{target}",
                payload=CalculationStateChanged(
                    calculation_id=calc.calculation_id, state=target,
                    scheduler_state=seen.state if target is _C.SCHEDULER_TERMINAL else None),
            )
            result.recorded.extend(self._ledger.record(investigation_id, [draft]).events)
        return result
